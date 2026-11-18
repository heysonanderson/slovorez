from __future__ import annotations

import queue
import threading
import logging
from pathlib import Path
from typing import Union
from slovorez.core.engine import ModelResource, available_cpu_count
from slovorez.core.codec import MorphCodec
from slovorez.core.cache import LogWriter, SeenIndex
from slovorez.core.tokenizer import FFTokenizer, FTTokenizer
from slovorez.config import SlovorezConfig
from slovorez.paths import model_predictions_path
from slovorez.hub import resolve_model_dir, CONFIG_NAME
from slovorezCXX import TokenType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline consts
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE  = 65536
_DEFAULT_MODEL_BATCH = 256
_DEFAULT_QUEUE_LIMIT = 16
_MIN_WORKERS = 1
_DEFAULT_WORKERS = max(_MIN_WORKERS, available_cpu_count() - 2)


def _put_unless_stopped(
        q: queue.Queue, item, stop: threading.Event
)-> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False

def _check_workers_number(max_workers: int) -> None:
    if max_workers < _MIN_WORKERS:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}; use threaded_mode=False for sequential mode")
# ---------------------------------------------------------------------------
# Slovorez Orchestrator
# ---------------------------------------------------------------------------

class Slovorez:
    """Morpheme segmentation pipeline for Russian text.

    Composes four components:
      - ``ModelResource``     -- ONNX inference session (CPU).
      - ``MorphCodec`` -- char-level encoder and BIES decoder.
      - ``SeenIndex``         -- cross-session deduplication (seen-set).
      - ``LogWriter``         -- buffered JSONL output.

    Prefer ``from_pretrained()`` over direct construction.
    """

    def __init__(
        self,
        model: ModelResource,
        tokenizer: MorphCodec,
        index: SeenIndex,
        writer: LogWriter,
        model_name: str = "unknown",
    ):
        self._model      = model
        self._tokenizer  = tokenizer
        self._index      = index
        self._writer     = writer
        self._model_name = model_name

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: Union[str, Path, None] = "slovorez-flash",
        predictions_path: Union[str, Path, None] = None,
        providers: list[str] | None = None,
    ) -> Slovorez:

        model_dir = resolve_model_dir(name_or_path=name_or_path)
        config = SlovorezConfig.from_file(model_dir / CONFIG_NAME)
        weights_path = model_dir / config.resources.weights.file
        if predictions_path is not None:
            predictions_path = Path(predictions_path).resolve()
        else:
            predictions_path = model_predictions_path(config.name, config.version)

        return cls(
            model = ModelResource(weights_path, providers=providers),
            tokenizer = MorphCodec.from_config(config),
            index = SeenIndex(
                minlen = config.io.minlen,
                maxlen = config.io.maxlen,
                base_dict_path = model_dir / config.resources.base_dict.file,
                predictions_path=predictions_path),
            writer = LogWriter(predictions_path),
            model_name = config.name
        )
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
 
    def predict(self, text: str) -> list[dict]:
        """Segment the Russian words of ``text`` into morphemes.

        Words are lowercased and returned in text order, duplicates included.
        Non-Russian tokens are dropped; words longer than the model's ``maxlen``
        (64 for slovorez-flash) are skipped silently. Nothing is written to the
        predictions cache.

        Each record is the same as a line of the predictions jsonl::

            {
                "word": "выделяется",
                "morphemes": [["вы", "PREF", 0.99], ["дел", "ROOT", 0.99], ...],
                "confidence": 0.15,
                "model": "slovorez-flash",
            }

        ``morphemes`` is a list of ``[text, type, score]``; ``type`` is one of
        the model's morpheme types (PREF, ROOT, SUFF, END, POSTFIX, LINK, HYPH),
        ``score`` is the per-morpheme confidence in [0, 1]. ``confidence`` is
        the calibrated confidence of the whole parse. Joining the morpheme
        texts gives back ``word``.

        Args:
            text: any text; only Russian words are segmented.

        Returns:
            One record per Russian word, in text order (format above).

        Raises:
            TypeError: ``text`` is not a ``str``.
            ValueError: ``text`` is empty or whitespace only.
        """

        tokenizer_cxx = FTTokenizer(text + "\n")
        tokenizer_cxx.set_filter(TokenType.RUWORD)
        tokenizer_cxx.set_token_min_len(self._index.minlen)
        tokenizer_cxx.set_token_max_len(self._index.maxlen)

        results = []
        batch = tokenizer_cxx.get_batch_tokens()
        while batch is not None:
            encoded, lengths = self._tokenizer.encode_batch_padded(batch)
            logits = self._model.predict(encoded)
            results_batch = self._tokenizer.decode_predictions_batch(
                batch, logits, lengths, self._model_name
            )
            results.extend(results_batch)
            batch = tokenizer_cxx.get_batch_tokens()

        return results
    
    # ------------------------------------------------------------------
    # File processing
    # ------------------------------------------------------------------

    def process_file(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH,
        max_workers: int = _DEFAULT_WORKERS,
        threaded_mode: bool = False
    ) -> None:
        """Process a text file and persist all morpheme predictions to disk.\n
        If the process falls, its better to recreate the class using the `from_pretrained` method. 
        Args:
            file_path:           path to the input text file.
            batch_size:          number of tokens per C++ tokenizer batch.
            model_batch:         maximum words per single model inference call.
            max_workers:         maximum CPU inferencee workers (threaded mode only).
            threaded_mode: if True, spawns threads for CPU multithreading.
                                  if False, runs sequentially in the main thread
                                  (recommended for Windows or small files).
        """
        if threaded_mode:
            self._process_file_multithread(file_path, batch_size, model_batch, max_workers)
        else:
            self._process_file_sequential(file_path, batch_size, model_batch)

    def _infer_and_persist(self, words) -> None:
        encoded, lengths = self._tokenizer.encode_batch_padded(words)
        logits = self._model.predict(encoded)
        predictions = self._tokenizer.decode_predictions_batch(
            words, logits, lengths, self._model_name
        )
        self._writer.write(predictions)

    def _process_file_sequential(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH
    ) -> None:
        tokenizer_cxx = FFTokenizer(file_path)
        tokenizer_cxx.set_batch_size(batch_size)
        tokenizer_cxx.set_filter(TokenType.RUWORD)
        tokenizer_cxx.set_token_min_len(self._index.minlen)
        tokenizer_cxx.set_token_max_len(self._index.maxlen)

        try:
            pending = []
            batch = tokenizer_cxx.get_batch_tokens()
            while batch:
                unseen = self._index.filter_unseen(set(batch))
                self._index.mark_seen(unseen)
                pending.extend(unseen)
                while len(pending) >= model_batch:
                    pending = sorted(pending, key=len)
                    self._infer_and_persist(pending[:model_batch])
                    pending = pending[model_batch:]
                batch = tokenizer_cxx.get_batch_tokens()
            if pending: 
                self._infer_and_persist(pending)
        finally:
            self._writer.flush()
        logger.info(f"File '{file_path}' successfully processed (sequential).")

    def _process_file_multithread(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH,
        max_workers: int = _DEFAULT_WORKERS
    ) -> None:

        _check_workers_number(max_workers)

        tokenizer_cxx = FFTokenizer(file_path)
        tokenizer_cxx.set_batch_size(batch_size)
        tokenizer_cxx.set_filter(TokenType.RUWORD)
        tokenizer_cxx.set_token_min_len(self._index.minlen)
        tokenizer_cxx.set_token_max_len(self._index.maxlen)
        
        chunk_queue = queue.Queue(maxsize=_DEFAULT_QUEUE_LIMIT)
        writer_queue = queue.Queue()

        self._model.get_session()

        errors = []
        stop = threading.Event()

        pipeline_threads = [
            threading.Thread(
                target=self._pipeline_worker,
                args=(chunk_queue, writer_queue, self._model, self._tokenizer, self._model_name, errors, stop),
                daemon=True
            )
            for _ in range(max_workers)
        ]
        for t in pipeline_threads:
            t.start()

        writer_thread = threading.Thread(
            target=self._writer_worker,
            args=(writer_queue, self._writer, errors, stop),
            daemon=True
        )
        writer_thread.start()

        try:
            pending = []
            batch = tokenizer_cxx.get_batch_tokens()
            while batch and not stop.is_set():
                unseen = self._index.filter_unseen(set(batch))
                self._index.mark_seen(unseen)
                pending.extend(unseen)
                while len(pending) >= model_batch:
                    pending = sorted(pending, key=len)
                    if not _put_unless_stopped(chunk_queue, pending[:model_batch], stop):
                        break
                    pending = pending[model_batch:]
                batch = tokenizer_cxx.get_batch_tokens()
            if pending:
                _put_unless_stopped(chunk_queue, pending, stop)
            for _ in pipeline_threads:
                _put_unless_stopped(chunk_queue, None, stop)
        except BaseException:
            stop.set()
            raise
        finally:
            for t in pipeline_threads:
                t.join()
            writer_queue.put(None)
            writer_thread.join()

        if errors:
            raise errors[0]

        logger.info(f"File '{file_path}' successfully processed (multithread).")

    @staticmethod
    def _pipeline_worker(
            chunk_queue: queue.Queue,
            writer_queue: queue.Queue,
            model: ModelResource,
            tokenizer: MorphCodec,
            model_name: str,
            errors: list[Exception],
            stop: threading.Event
    ) -> None:
        try:
            while not stop.is_set():
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                encoded, lengths = tokenizer.encode_batch_padded(chunk)
                logits = model.predict(encoded)
                results = tokenizer.decode_predictions_batch(chunk, logits, lengths, model_name)
                writer_queue.put(results)
        except Exception as e:
            errors.append(e)
            stop.set()

    @staticmethod
    def _writer_worker(
            writer_queue: queue.Queue,
            writer: LogWriter,
            errors: list[Exception],
            stop: threading.Event
    ) -> None:
        try:
            while True:
                item = writer_queue.get()
                if item is None:
                    break
                results = item
                writer.write(results)
            writer.flush()
        except Exception as e:
            errors.append(e)
            stop.set()