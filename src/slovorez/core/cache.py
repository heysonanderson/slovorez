from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union
from slovorez.io.loaders import load_json

logger = logging.getLogger(__name__)

_FLUSH_SIZE = 32768

# ===========================================================================
# SeenIndex
# ===========================================================================

class SeenIndex:
    """Tracks which words have already been processed.
Args:
    base_dict_path: path to base_dict.json (optional)
    predictions_path: path to words.jsonl from previous run (optional)
               If None, only the base_dict is loaded.
    """

    def __init__(self, minlen: int = 1, maxlen: int = 64, base_dict_path: str | Path | None = None, predictions_path: str | Path | None = None):
        self._seen  = self._init_seen(base_dict_path, predictions_path)
        self.minlen = minlen
        self.maxlen = maxlen

    def _init_seen(self, base_dict_path, predictions_path):
        jsonl_keys = self._load_jsonl_keys(predictions_path) if predictions_path else set()
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


    def filter_unseen(self, words: set[str]) -> list[str]:
        """Return deduped words not yet seen
        """
        return list(words - self._seen)

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


class LogWriter:
    """Buffered append-only writer for JSONL prediction logs.

    Accumulates result dicts in memory and flushes to disk either when the
    buffer reaches ``_FLUSH_SIZE`` or when ``flush()`` is called explicitly.

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


    def _flush_buffer(self) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                for record in self._buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._buffer.clear()
        except OSError as e:
            logger.error(f"LogWriter: failed to write to {self._path}: {e}")
            raise