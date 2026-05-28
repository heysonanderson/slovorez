from __future__ import annotations

import queue
import logging
import multiprocessing
from pathlib import Path
from typing import Union

import numpy as np

from slovorez.core.engine import ModelResource
from slovorez.core.process import SlovorezTokenizer
from slovorez.core.cache import LogWriter, MorphemeRegistry, PersistenceIndex
from slovorez.core.tokenizer import FFTokenizer, FTTokenizer
from slovorez.io.loaders import load_json
from slovorez.utils import resolve_model_dir, resolve_path, MODEL_CONFIG_NAME
from slovorezCXX import TokenType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline consts
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE  = 65536
_DEFAULT_MODEL_BATCH = 2048
_DEFAULT_QUEUE_LIMIT = 16
_DEFAULT_MAX_WORKERS = 8

# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _gpu_worker(
    model_path: str,
    gpu_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    """Inference worker: loads the ONNX model and runs predict in a loop.

    Receives (words, encoded) from ``gpu_queue``, returns (words, logits)
    to ``result_queue``. Logits are cast to float16 to halve IPC payload.
    Terminates on receiving None from the queue.
    """
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
    base_dict_keys: frozenset[str],
    tokenizer_config: dict,
    model_batch: int,
    min_len: int,
    max_len: int,
) -> None:
    """Tokenization worker: filters and encodes word batches for the GPU.

    Applies three filters before a word reaches the GPU:
      1. Length must be within [min_len, max_len].
      2. Must not be in the base dictionary (pre-validated, no inference needed).
      3. Must not have been seen in a prior session (cache_snapshot) or
         earlier in this worker's own run (local_seen).

    ``cache_snapshot`` and ``base_dict_keys`` are frozen at worker spawn time
    and treated as read-only throughout the worker's lifetime.
    """
    tokenizer   = SlovorezTokenizer.from_config(tokenizer_config)
    local_seen: set[str] = set()
    pending:    list[str] = []

    def _flush() -> None:
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
            if (
                min_len <= len(token) <= max_len
                and token not in base_dict_keys
                and token not in cache_snapshot
                and token not in local_seen
            ):
                local_seen.add(token)
                pending.append(token)
                if len(pending) >= model_batch:
                    _flush()

    _flush()


def _writer_worker(
    result_queue: multiprocessing.Queue,
    output_path: str,
    tokenizer_config: dict,
    model_name: str,
) -> None:
    """Writer worker: decodes logits and persists results to disk.

    Uses ``LogWriter`` for buffered JSONL output. No deduplication is
    performed here -- the CPU workers guarantee uniqueness upstream.
    Terminates on receiving None from the queue.
    """
    writer    = LogWriter(output_path)
    tokenizer = SlovorezTokenizer.from_config(tokenizer_config)

    while True:
        item = result_queue.get()
        if item is None:
            break

        words, logits = item
        results = list(tokenizer.decode_predictions_detail(words, logits, model_name))
        writer.write(results)

    writer.flush()


# ---------------------------------------------------------------------------
# Slovorez Orchestrator
# ---------------------------------------------------------------------------

class Slovorez:
    """Morpheme segmentation pipeline for Russian text.

    Composes four components:
      - ``ModelResource``     -- ONNX inference session (GPU/CPU).
      - ``SlovorezTokenizer`` -- char-level encoder and BIES decoder.
      - ``PersistenceIndex``  -- cross-session deduplication (seen-set).
      - ``MorphemeRegistry``  -- in-memory morpheme store (base + validated).
      - ``LogWriter``         -- buffered JSONL output.

    Prefer ``from_pretrained()`` over direct construction.
    """

    def __init__(
        self,
        model: ModelResource,
        tokenizer: SlovorezTokenizer,
        index: PersistenceIndex,
        registry: MorphemeRegistry,
        writer: LogWriter,
        model_name: str = "unknown",
    ):
        self._model      = model
        self._tokenizer  = tokenizer
        self._index      = index
        self._registry   = registry
        self._writer     = writer
        self._model_name = model_name

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        output_path: Union[str, Path, None] = None,
        base_dict_path: Union[str, Path, None] = None,
        device: str = "auto",
    ) -> Slovorez:
        """Load a Slovorez model from a local directory.

        The directory must contain a ``config.json`` file. All other resource
        paths (weights, base dictionary, predictions output) are resolved
        relative to that directory unless explicitly overridden.

        Directory layout (conventional)::

            models/
            └── slovorez-v1/
                ├── config.json          # required
                ├── slovorez-v1.onnx     # weights -- path from config["resources"]["weights"]
                ├── base_dict.json       # optional static dictionary
                └── predictions.jsonl   # default output location

        Args:
            model_name_or_path: path to the model directory (absolute or
                relative to cwd / PROJECT_ROOT). A bare name like
                ``"slovorez-test"`` works if the directory is resolvable.
            output_path: override where predictions are written. Defaults to
                ``config["resources"]["output"]`` resolved inside the model dir.
            base_dict_path: override the static base dictionary path. Defaults
                to ``config["resources"]["base_dict"]`` if present.
            device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.

        Example::

            model = Slovorez.from_pretrained("models/slovorez-v1")
            model = Slovorez.from_pretrained("models/slovorez-v1", device="cuda")
            model = Slovorez.from_pretrained(
                "models/slovorez-v1",
                output_path="runs/experiment-1/predictions.jsonl",
            )
        """
        model_dir   = resolve_model_dir(model_name_or_path)
        config      = load_json(model_dir / MODEL_CONFIG_NAME)
        resources   = config.get("resources", {})
        model_specs = config["model_specs"]
        model_name  = model_specs["name"]

        # --- weights ---------------------------------------------------------
        weights_filename = resources.get("weights", f"{model_name}.onnx")
        model_path = model_dir / weights_filename
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. "
                f"Expected filename from config[\"resources\"][\"weights\"]: "
                f"'{weights_filename}'"
            )

        # --- output path -----------------------------------------------------
        if output_path is not None:
            resolved_output = resolve_path(output_path)
        else:
            default_output_name = resources.get("output", "predictions.jsonl")
            resolved_output = model_dir / default_output_name

        logger.info(f"Predictions will be written to: {resolved_output}")

        # --- morpheme registry (base dict + validated dict) ------------------
        registry = MorphemeRegistry()

        resolved_dict_path: Path | None = None
        if base_dict_path is not None:
            resolved_dict_path = resolve_path(base_dict_path)
        elif "base_dict" in resources:
            candidate = model_dir / resources["base_dict"]
            if candidate.is_file():
                resolved_dict_path = candidate
            else:
                logger.warning(
                    f"base_dict listed in config but not found: {candidate}. Skipping."
                )

        if resolved_dict_path is not None:
            registry = MorphemeRegistry.from_base_dict(resolved_dict_path)

        # --- persistence index (seen-set rebuilt from log file) --------------
        index = PersistenceIndex.from_jsonl(
            resolved_output,
            max_len=model_specs["maxlen"],
        )

        return cls(
            model      = ModelResource(str(model_path), device=device),
            tokenizer  = SlovorezTokenizer.from_config(config),
            index      = index,
            registry   = registry,
            writer     = LogWriter(resolved_output),
            model_name = model_name,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict[str, list[tuple[str, int, float]]]:
        """Segment all Russian words in text into morphemes.

        Words present in the registry (base dict or validated dict) are
        returned directly without inference. Remaining unseen words are
        encoded, run through the model, registered, logged, and returned.

        Args:
            text: raw input string (any language mix is fine -- only Russian
                  words are extracted and segmented).

        Returns:
            Dict mapping each unique Russian word (lowercased) to its
            morpheme list: [(morpheme_text, morpheme_type_id, confidence), ...].
        """
        tokenizer_cxx = FTTokenizer(text)
        tokenizer_cxx.set_filter(TokenType.RUWORD)

        final_results: dict[str, list[tuple[str, int, float]]] = {}
        batch = tokenizer_cxx.get_batch()

        while batch:
            tokens = batch["text"].lower().split('\0')[:-1]

            candidates = [t for t in tokens if t not in self._registry.base_dict_keys]
            unseen = self._index.filter_unseen(candidates)

            if unseen:
                encoded     = self._tokenizer.encode_batch(unseen)
                logits      = self._model.predict(encoded)
                rich_results = list(self._tokenizer.decode_predictions_detail(
                    unseen, logits, self._model_name
                ))
                self._index.mark_seen(unseen)
                self._registry.register(rich_results)
                self._writer.write(rich_results)

            for token in tokens:
                if token not in final_results:
                    morphemes = self._registry.lookup(token)
                    if morphemes is not None:
                        final_results[token] = morphemes

            batch = tokenizer_cxx.get_batch()

        return final_results

    # ------------------------------------------------------------------
    # File processing
    # ------------------------------------------------------------------

    def process_file(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        multiprocessing_mode: bool = False,
    ) -> None:
        """Process a text file and persist all morpheme predictions to disk.

        Args:
            file_path:           path to the input text file.
            batch_size:          number of characters per C++ tokenizer batch.
            model_batch:         maximum words per single model inference call.
            max_workers:         maximum CPU tokenizer workers (multiprocessing only).
            multiprocessing_mode: if True, spawns workers for CPU/GPU parallelism.
                                  if False, runs sequentially in the main thread
                                  (recommended for Windows or small files).
        """
        if multiprocessing_mode:
            self._process_file_multiprocessing(file_path, batch_size, model_batch, max_workers)
        else:
            self._process_file_sequential(file_path, batch_size, model_batch)

    def _process_file_sequential(
        self,
        file_path: Union[str, Path],
        batch_size: int,
        model_batch: int,
    ) -> None:
        tokenizer_cxx = FFTokenizer(file_path)
        tokenizer_cxx.set_batch_size(batch_size)
        tokenizer_cxx.set_filter(TokenType.RUWORD)

        batch = tokenizer_cxx.get_batch()
        while batch:
            tokens     = batch["text"].lower().split('\0')[:-1]
            candidates = {t for t in tokens if t not in self._registry.base_dict_keys}
            unseen     = self._index.filter_unseen(list(candidates))

            for i in range(0, len(unseen), model_batch):
                chunk = unseen[i : i + model_batch]

                encoded      = self._tokenizer.encode_batch(chunk)
                logits       = self._model.predict(encoded)
                rich_results = list(self._tokenizer.decode_predictions_detail(
                    chunk, logits, self._model_name
                ))

                self._index.mark_seen(chunk)
                self._registry.register(rich_results)
                self._writer.write(rich_results)

            batch = tokenizer_cxx.get_batch()

        self._writer.flush()
        logger.info(f"File '{file_path}' successfully processed (sequential).")

    def _process_file_multiprocessing(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        """Process a text file using multiprocessing.

        Spawns separate CPU tokenizer workers, one GPU inference worker, and
        one writer worker. Results are appended to the predictions log file.

        Worker roles:
          - CPU workers (up to max_workers): tokenize, filter, encode -- forward
            to gpu_queue.
          - GPU worker (1): runs model inference -- forwards to result_queue.
          - Writer worker (1): decodes logits and flushes to disk via LogWriter.

        The main process feeds raw C++ batches into task_queue and waits for
        all workers to finish before reloading the index from disk.
        """
        # Snapshot immutable state for worker processes.
        cache_snapshot   = self._index.snapshot()
        base_dict_keys   = self._registry.base_dict_keys
        tokenizer_config = self._tokenizer.to_config()
        min_len          = self._index.min_len
        max_len          = self._index.max_len

        gpu_queue    = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        task_queue   = multiprocessing.Queue(_DEFAULT_QUEUE_LIMIT)

        gpu_proc = multiprocessing.Process(
            target=_gpu_worker,
            args=(str(self._model.model_path), gpu_queue, result_queue),
        )
        gpu_proc.start()

        writer_proc = multiprocessing.Process(
            target=_writer_worker,
            args=(result_queue, str(self._writer.path), tokenizer_config, self._model_name),
        )
        writer_proc.start()

        active_workers: list[multiprocessing.Process] = []

        def _start_worker() -> None:
            if len(active_workers) < max_workers:
                w = multiprocessing.Process(
                    target=_cpu_worker,
                    args=(
                        task_queue, gpu_queue, cache_snapshot, base_dict_keys,
                        tokenizer_config, model_batch, min_len, max_len,
                    ),
                )
                w.start()
                active_workers.append(w)

        _start_worker()

        tokenizer_cxx = FFTokenizer(file_path)
        tokenizer_cxx.set_batch_size(batch_size)
        tokenizer_cxx.set_filter(TokenType.RUWORD)

        batch = tokenizer_cxx.get_batch()
        while batch:
            try:
                task_queue.put(batch, block=True, timeout=0.1)
            except queue.Full:
                _start_worker()
                task_queue.put(batch, block=True)
            batch = tokenizer_cxx.get_batch()

        # Drain workers in order: CPU -> GPU -> writer.
        for _ in active_workers:
            task_queue.put(None)
        for w in active_workers:
            w.join()

        gpu_queue.put(None)
        gpu_proc.join()

        result_queue.put(None)
        writer_proc.join()

        # Sync main-process index with results written by the writer worker.
        self._index.reload_from_jsonl(self._writer.path)
        logger.info(f"File '{file_path}' successfully processed (multiprocessing).")
