from __future__ import annotations

import queue
import logging
import multiprocessing
from pathlib import Path
from typing import Union, Generator

import numpy as np

from slovorez.core.engine import ModelResource
from slovorez.core.process import SlovorezTokenizer
from slovorez.core.cache_manager import CacheManager
from slovorez.core.tokenizer import FFTokenizer, FTTokenizer
from slovorez.io.loaders import load_json
from slovorez.utils import resolve_path
from slovorezCXX import TokenType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline consts
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE    = 65536   
_DEFAULT_MODEL_BATCH   = 2048    
_DEFAULT_QUEUE_LIMIT   = 16      
_DEFAULT_MAX_WORKERS   = 8       

# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _gpu_worker(
    model_path: str,
    gpu_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    model = ModelResource(model_path)

    while True:
        item = gpu_queue.get()
        if item is None:
            break

        words, encoded = item
        logits = model.predict(encoded)
        result_queue.put((words, logits.astype(np.float16)))


def _cpu_worker(
    task_queue: multiprocessing.Queue,
    gpu_queue: multiprocessing.Queue,
    cache_snapshot: frozenset[str],
    tokenizer_config: dict,
    model_batch: int,
) -> None:
    tokenizer = SlovorezTokenizer.from_config(tokenizer_config)
    local_seen = set()
    pending: list[str] = []

    def _flush():
        if pending:
            encoded = tokenizer.encode_batch(pending)
            gpu_queue.put((list(pending), encoded))
            pending.clear()

    while True:
        batch = task_queue.get()
        if batch is None:
            break

        tokens = batch["text"].lower().split('\0')[:-1]

        for token in tokens:
            if token not in cache_snapshot and token not in local_seen:
                local_seen.add(token)
                pending.append(token)
                if len(pending) >= model_batch:
                    _flush()
    _flush()


def _writer_worker(
    result_queue: multiprocessing.Queue,
    output_path: str,
    cache_snapshot: frozenset[str],
    tokenizer_config: dict,
    model_name: str,
    min_len: int,
    max_len: int,
) -> None:
    cache = CacheManager(
        output_path=output_path,
        initial_keys=cache_snapshot,
        min_len=min_len,
        max_len=max_len,
    )
    tokenizer = SlovorezTokenizer.from_config(tokenizer_config)

    while True:
        item = result_queue.get()
        if item is None: break

        words, logits = item
        results = list(tokenizer.decode_predictions_detail(words, logits, model_name))
        cache.update_cache(results)

    cache.flush()


# ---------------------------------------------------------------------------
# Slovorez Orchestrator
# ---------------------------------------------------------------------------

class Slovorez:
    def __init__(
        self,
        model: ModelResource,
        tokenizer: SlovorezTokenizer,
        cache: CacheManager,
        model_name: str = "unknown",
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._cache = cache
        self._model_name = model_name

    @classmethod
    def from_pretrained(
        cls,
        config_path: Union[str, Path],
        output_path: Union[str, Path, None] = None,
        base_dict_path: Union[str, Path, None] = None,
        device: str = "auto",
    ) -> Slovorez:
        config_path = resolve_path(config_path)
        config = load_json(config_path)

        model_name = config["model_specs"]["name"]
        model_filename = config['model_specs'].get('name', model_name)
        model_path = config_path.parent / f"{model_filename}.onnx"

        initial_keys = None
        if base_dict_path:
            base_dict = load_json(resolve_path(base_dict_path))
            initial_keys = set(base_dict.keys())

        return cls(
            model=ModelResource(str(model_path), device=device),
            tokenizer=SlovorezTokenizer.from_config(config),
            cache=CacheManager(
                output_path=output_path or (config_path.parent / "predictions.jsonl"),
                initial_keys=initial_keys,
                max_len=config["model_specs"]["maxlen"]
            ),
            model_name=model_name
        )

    def predict(self, text: str) -> dict[str, list[tuple[str, int, float]]]:
        tokenizer_cxx = FTTokenizer(text)
        tokenizer_cxx.set_filter(TokenType.RUWORD)
        
        final_results = {}
        batch = tokenizer_cxx.get_batch()
        
        while batch:
            tokens = batch["text"].lower().split('\0')[:-1]
            unseen = self._cache.filter_unseen(tokens)

            if unseen:
                encoded = self._tokenizer.encode_batch(unseen)
                logits = self._model.predict(encoded)

                rich_results = list(self._tokenizer.decode_rich_results(
                    unseen, logits, self._model_name
                ))

                self._cache.update_cache(rich_results)

                for res in rich_results:
                    final_results[res["word"]] = res["morphemes"]
            
            batch = tokenizer_cxx.get_batch()
        
        return final_results

    def process_file(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        """
        Multiprocessing method
        """
        cache_snapshot = frozenset(self._cache._seen)
        
        tokenizer_config = {
            "mapping": {
                "tokenizer_vocab": self._tokenizer.char_vocab,
                "label2id":        self._tokenizer.bies_vocab,
            },
            "model_specs": {"maxlen": self._tokenizer.maxlen},
        }

        gpu_queue    = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        task_queue   = multiprocessing.Queue(_DEFAULT_QUEUE_LIMIT)

        gpu_proc = multiprocessing.Process(
            target=_gpu_worker, 
            args=(str(self._model.model_path), gpu_queue, result_queue)
        )
        gpu_proc.start()

        writer_proc = multiprocessing.Process(
            target=_writer_worker,
            args=(
                result_queue, str(self._cache._path), cache_snapshot, 
                tokenizer_config, self._model_name, self._cache.min_len, self._cache.max_len
            )
        )
        writer_proc.start()

        active_workers = []
        def start_worker():
            if len(active_workers) < max_workers:
                w = multiprocessing.Process(
                    target=_cpu_worker,
                    args=(task_queue, gpu_queue, cache_snapshot, tokenizer_config, model_batch)
                )
                w.start()
                active_workers.append(w)

        start_worker()

        tokenizer_cxx = FFTokenizer(file_path)
        tokenizer_cxx.set_batch_size(batch_size)
        tokenizer_cxx.set_filter(TokenType.RUWORD)

        batch = tokenizer_cxx.get_batch()
        while batch:
            try:
                task_queue.put(batch, block=True, timeout=0.1)
            except queue.Full:
                start_worker()
                task_queue.put(batch, block=True)
            batch = tokenizer_cxx.get_batch()

        for _ in active_workers: task_queue.put(None)
        for w in active_workers: w.join()

        gpu_queue.put(None)
        gpu_proc.join()

        result_queue.put(None)
        writer_proc.join()

        logger.info(f"File {file_path} succesfully processed.")