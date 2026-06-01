from __future__ import annotations

import queue
import logging
from pathlib import Path
from typing import Union
from slovorez.core.engine import ModelResource
from slovorez.core.process import SlovorezTokenizer
from slovorez.core.cache import LogWriter, SeenIndex
from slovorez.core.tokenizer import FFTokenizer, FTTokenizer
from slovorez.io.loaders import load_json
from slovorez.utils import resolve_model_dir, resolve_path, MODEL_CONFIG_NAME
from slovorezCXX import TokenType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline consts
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE  = 65536
_DEFAULT_MODEL_BATCH = 256
_DEFAULT_QUEUE_LIMIT = 16
_DEFAULT_MAX_WORKERS = 16

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
        model_name_or_path: Union[str, Path],
        output_path: Union[str, Path, None] = None,
        device: str = "auto",
    ) -> Slovorez:
        """Load a Slovorez model from a local directory.

        The directory must contain a ``config.json`` file. All other resource
        paths (weights, base dictionary, predictions output) are resolved
        relative to that directory unless explicitly overridden.

        Directory layout (conventional)::

            models/
            └── slovorez-test/
                ├── config.json          # required
                ├── slovorez-test.onnx     # weights
                ├── base_dict.json       # optional static dictionary
                └── predictions.jsonl   # default output

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

        return cls(
            model      = ModelResource(str(model_path), device=device),
            tokenizer  = SlovorezTokenizer.from_config(config),
            index      = SeenIndex.from_config(config),
            writer     = LogWriter(resolved_output),
            model_name = model_name,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
 
    def predict(self, text: str, write_to_cache: bool = False) -> list[dict]:
        """Segment Russian words in text into morphemes, preserving word order.
 
        Two-phase approach:
          1. Collect all tokens, infer words, register results.
          2. Return ordered output.
 
        Args:
            text: raw input string. Any language mix is accepted -- only Russian words 
                are extracted and segmented.
            write_to_cache: if ``True``, newly inferred words are persisted 
                to the index and JSONL log so they are reused across calls.
                If ``False``, results not written to disk.
 
        Returns:
            List of result dicts, one per Russian word token in input order.
            Each dict has keys: word, morphemes.
 
        Example::
 
            results = model.predict("Я сижу в своей комнате, в обиталище шума всей квартиры.")
            # [ ...            
            #   {"word": "комнате",   "morphemes": [...]},
            #   {"word": "в",   "morphemes": [...]      },
            #   {"word": "обиталище", "morphemes": [...]},
            #   {"word": "шума",   "morphemes": [...]   },
            #   ...   ]
        """
        tokenizer_cxx = FTTokenizer(text)
        tokenizer_cxx.set_filter(TokenType.RUWORD)
        tokenizer_cxx.set_token_max_len(self._index.max_len)

        all_predictions = []
        batch = tokenizer_cxx.get_batch_tokens(tolower=True)
        while batch is not None:
            encoded      = self._tokenizer.encode_batch(batch)
            logits       = self._model.predict(encoded)
            rich_results = self._tokenizer.decode_predictions_detail(
                batch, logits, self._model_name
            )

            if write_to_cache:
                self._index.mark_seen(batch)
                self._writer.write(rich_results)

            all_predictions.extend(rich_results)
            batch = tokenizer_cxx.get_batch_tokens()

        if write_to_cache:
            self._writer.flush()

        results: list[dict] = []
        for pred in all_predictions:
            results.append({
                "word":       pred["word"],
                "morphemes":  pred["morphemes"],
            })
 
        return results
    
    # ------------------------------------------------------------------
    # File processing
    # ------------------------------------------------------------------

    def process_file(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        threaded_mode: bool = False,
    ) -> None:
        """Process a text file and persist all morpheme predictions to disk.

        Args:
            file_path:           path to the input text file.
            batch_size:          number of characters per C++ tokenizer batch.
            model_batch:         maximum words per single model inference call.
            max_workers:         maximum CPU tokenizer workers (multiprocessing only).
            threaded_mode: if True, spawns workers for CPU/GPU parallelism.
                                  if False, runs sequentially in the main thread
                                  (recommended for Windows or small files).
        """
        if threaded_mode:
            self._process_file_multithread(file_path, batch_size, model_batch, max_workers)
        else:
            self._process_file_sequential(file_path, batch_size, model_batch)

    def _infer_and_persist(self, words) -> None:
        encoded = self._tokenizer.encode_batch(words)
        logits = self._model.predict(encoded)
        predictions = self._tokenizer.decode_predictions_detail(
            words, logits, self._model_name
        )
        self._writer.write(predictions)
        self._index.mark_seen(words)

    def _process_file_sequential(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH
    ) -> None:
        tokenizer_cxx = FFTokenizer(file_path)
        tokenizer_cxx.set_batch_size(batch_size)
        tokenizer_cxx.set_filter(TokenType.RUWORD)
        tokenizer_cxx.set_token_min_len(self._index.min_len)
        tokenizer_cxx.set_token_max_len(self._index.max_len)

        pending = []
        batch = tokenizer_cxx.get_batch_tokens()
        while batch:
            unseen = self._index.filter_unseen(set(batch))
            self._index.mark_seen(unseen)
            pending.extend(unseen)

            while len(pending) >= model_batch:
                self._infer_and_persist(pending[:model_batch])
                pending = pending[model_batch:]
            batch = tokenizer_cxx.get_batch_tokens()

        if pending:
            self._infer_and_persist(pending)

        self._writer.flush()
        logger.info(f"File '{file_path}' successfully processed (sequential).")

    def _process_file_multithread(
        self,
        file_path: Union[str, Path],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_batch: int = _DEFAULT_MODEL_BATCH,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        import threading

        chunk_queue = queue.Queue(maxsize=_DEFAULT_QUEUE_LIMIT)
        writer_queue = queue.Queue()

        self._model.get_session()

        pipeline_threads = [
            threading.Thread(
                target=self._pipeline_worker,
                args=(chunk_queue, writer_queue, self._model, self._tokenizer, self._model_name),
                daemon=True
            )
            for _ in range(max_workers)
        ]
        for t in pipeline_threads:
            t.start()

        writer_thread = threading.Thread(
            target=self._writer_worker,
            args=(writer_queue, self._writer),
            daemon=True
        )
        writer_thread.start()

        tokenizer_cxx = FFTokenizer(file_path)
        tokenizer_cxx.set_batch_size(batch_size)
        tokenizer_cxx.set_filter(TokenType.RUWORD)
        tokenizer_cxx.set_token_min_len(self._index.min_len)
        tokenizer_cxx.set_token_max_len(self._index.max_len)

        pending = []
        batch = tokenizer_cxx.get_batch_tokens()
        while batch:
            unseen = self._index.filter_unseen(set(batch))
            self._index.mark_seen(unseen)
            pending.extend(unseen)
            while len(pending) >= model_batch:
                chunk_queue.put(pending[:model_batch])
                pending = pending[model_batch:]
            batch = tokenizer_cxx.get_batch_tokens()

        if pending:
            chunk_queue.put(pending)

        for _ in pipeline_threads:
            chunk_queue.put(None)
        for t in pipeline_threads:
            t.join()

        writer_queue.put(None)
        writer_thread.join()
        self._writer.flush()

    @staticmethod
    def _pipeline_worker(
            chunk_queue: queue.Queue,
            writer_queue: queue.Queue,
            model: ModelResource,
            tokenizer: SlovorezTokenizer,
            model_name: str
    ) -> None:
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            encoded = tokenizer.encode_batch(chunk)
            logits = model.predict(encoded)
            results = tokenizer.decode_predictions_detail(chunk, logits, model_name)
            writer_queue.put((results))

    @staticmethod
    def _writer_worker(
            writer_queue: queue.Queue,
            writer: LogWriter,
    ) -> None:
        while True:
            item = writer_queue.get()
            if item is None:
                break
            results = item
            writer.write(results)
        writer.flush()