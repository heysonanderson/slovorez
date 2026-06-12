from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union
from slovorez.io.loaders import load_json
from slovorez.utils import resolve_model_dir, resolve_path, MODEL_CONFIG_NAME

_DEFAULT_MAX_TOKEN_LEN = 64
_DEFAULT_MIN_TOKEN_LEN = 4

logger = logging.getLogger(__name__)

_FLUSH_SIZE = 65536

# ===========================================================================
# SeenIndex
# ===========================================================================

class SeenIndex:
    """Tracks which words have already been processed.
Args:
    base_dict_path: path to base_dict.json (optional)
    jsonl_path: path to words.jsonl from previous run (optional)
               If None, only the base_dict is loaded.
    """

    def __init__(self, min_len: int = 1, max_len: int = 64, base_dict_path: Union[str, Path] = None, jsonl_path: Union[str, Path] = None):
        self._seen = self._init_seen(base_dict_path, jsonl_path)
        self.min_len = min_len
        self.max_len = max_len

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict,
        model_dir_path: str | Path | None = None
    ) -> "SeenIndex":
        """Build an index by scanning word keys from an existing JSONL file.
        """
        resources = config.get("resources", {})
        model_specs = config.get("model_specs", {})

        if model_dir_path is not None:
            model_dir = Path(model_dir_path)
        else:
            model_name = model_specs.get("name")
            model_dir = resolve_model_dir(model_name)
    
        default_output_name = resources.get("output")
        base_dict_name = resources.get("base_dict")
        model_base_dict_path = None
        model_words_path = None
        
        if base_dict_name:
            model_base_dict_path = model_dir / base_dict_name
        if default_output_name:
            model_words_path = model_dir / default_output_name

        return cls(
            min_len=model_specs.get("minlen", _DEFAULT_MIN_TOKEN_LEN), 
            max_len=model_specs.get("maxlen", _DEFAULT_MAX_TOKEN_LEN),
            base_dict_path=model_base_dict_path,
            jsonl_path=model_words_path
        )

    def _init_seen(self, base_dict_path, model_words_path):
        jsonl_keys = self._load_jsonl_keys(model_words_path) if model_words_path else set()
        dict_keys = self._load_dict_keys(base_dict_path) if base_dict_path else set()
        return jsonl_keys | dict_keys

    def _load_dict_keys(self, json_path):
        keys = set()
        bp = Path(json_path)
        if not bp.is_file():
            return keys
        base_dict = load_json(bp)
        dict_loaded = len(base_dict)
        keys.update(base_dict.keys())
        logger.info(f"SeenIndex: loaded {dict_loaded:,} keys from {bp.name}")
        return keys

    def _load_jsonl_keys(self, json_path):
        jsonl_loaded = skipped = 0
        keys = set()
        jp = Path(json_path)
        if not jp.is_file():
            return keys
        with open(jp, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.add(json.loads(line)["word"])
                    jsonl_loaded += 1
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        f"Skipping malformed line {lineno} in {jp.name}"
                    )
                    skipped += 1
        logger.info(
            f"SeenIndex: loaded {jsonl_loaded:,} keys from {jp.name}"
            + (f" ({skipped} lines skipped)" if skipped else "")
        )
        return keys
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_unseen(self, words: set[str]) -> list[str]:
        """Return words not yet seen, deduped and sorted by length.

        Length filtering is applied here. Words from external lists (e.g.
        base dictionary) must be filtered out by the caller before this call.

        Args:
            words: lowercased word strings from the current batch.

        Returns:
            Sorted list of new unique words ready for inference.
        """
        unseen = words - self._seen
        return sorted(list(unseen), key=len)

    def mark_seen(self, words: list[str]) -> None:
        """Register words as seen so they are excluded from future batches.

        Args:
            words: lowercased word strings that have been processed.
        """
        self._seen.update(words)

    def snapshot(self) -> frozenset[str]:
        """Return an immutable copy of the seen-set for passing to workers.

        The returned ``frozenset`` is safe to pickle and share across
        ``multiprocessing.Process`` boundaries.
        """
        return frozenset(self._seen)

    def __len__(self) -> int:
        return len(self._seen)


# ===========================================================================
# LogWriter
# ===========================================================================

class LogWriter:
    """Buffered append-only writer for JSONL prediction logs.

    Accumulates result dicts in memory and flushes to disk either when the
    buffer reaches ``_FLUSH_SIZE`` or when ``flush()`` is called explicitly.

    Owns no deduplication logic -- that is ``SeenIndex``'s job.
    Owns no morpheme lookup -- that is ``MorphemeRegistry``'s job.

    Args:
        path: path to the JSONL output file. Parent directories are created
              automatically.

    Example::

        writer = LogWriter("words.jsonl")
        writer.write(results)   # buffered
        writer.flush()          # ensure everything is on disk
    """

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def write(self, results: list[dict]) -> None:
        """Add results to the write buffer, flushing automatically if full.

        Args:
            results: list of prediction dicts -- must be JSON-serialisable.
        """
        self._buffer.extend(results)
        if len(self._buffer) >= _FLUSH_SIZE:
            self._flush_buffer()

    def flush(self) -> None:
        """Write all remaining buffered results to disk immediately.

        Must be called after the last batch to guarantee no data loss.
        """
        if self._buffer:
            self._flush_buffer()
            logger.info(f"LogWriter: final flush to {self._path.name}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_buffer(self) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                for record in self._buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"LogWriter: failed to write to {self._path}: {e}")
            raise
        finally:
            self._buffer.clear()