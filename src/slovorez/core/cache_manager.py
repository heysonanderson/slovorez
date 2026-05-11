from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class CacheManager:
    """Write-through cache for morpheme segmentation results.

    Keeps an in-memory set of seen words for fast deduplication and
    persists new results to a JSONL file. Writes are batched -- the file
    is opened once per ``update_cache()`` call, not once per word.

    Args:
        output_path:  path to the JSONL file where new results are appended.
        initial_keys: pre-populate the seen-set from an existing dictionary
                      (e.g. the static Tikhonov dict) without loading full records.
        min_len:      minimum word length to consider for inference.
        max_len:      maximum word length to consider for inference.

    Example::

        cache = CacheManager("predictions-raw.jsonl", initial_keys=base_dict.keys())
        unseen = cache.filter_unseen(words)
        ...
        cache.update_cache(results)
        cache.flush()   # обязательно после окончания обработки текста
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        initial_keys: Union[set[str], None] = None,
        min_len: int = 1,
        max_len: int = 64,
    ):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._seen: set[str] = set(initial_keys) if initial_keys else set()
        self.min_len = min_len
        self.max_len = max_len

        self._buffer: list[dict] = []

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_unseen(self, words: list[str]) -> list[str]:
        """Return words not yet in cache, sorted alphabetically.

        Deduplicates within the batch and against the full seen-set.
        Length filtering applied according to min_len / max_len.

        Args:
            words: lowercased word strings from the current batch.

        Returns:
            Sorted list of new unique words ready for inference.
        """
        seen = self._seen
        min_len = self.min_len
        max_len = self.max_len
        unseen = {w for w in words if min_len <= len(w) <= max_len and w not in seen}
        return sorted(unseen, key=len)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def update_cache(self, results: list[dict]) -> None:
        """Add new results to the in-memory seen-set and write buffer.

        Does not write to disk immediately -- call ``flush()`` to persist,
        or rely on the automatic flush when the buffer exceeds ``_FLUSH_SIZE``.

        Args:
            results: list of prediction dicts with at least a "word" key.
        """
        seen = self._seen
        buf = self._buffer

        for r in results:
            seen.add(r["word"])
            buf.append(r)

        if len(buf) >= _FLUSH_SIZE:
            self._write_buffer()

    def flush(self) -> None:
        """Write all remaining buffered results to disk.

        Must be called explicitly after the last batch is processed
        to guarantee no data loss.
        """
        if self._buffer:
            self._write_buffer()
            logger.info(f"Final flush: buffer cleared, total seen: {len(self._seen)}")

    def _write_buffer(self) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                for record in self._buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"Failed to write cache to {self._path}: {e}")
            raise
        finally:
            self._buffer.clear()


_FLUSH_SIZE = 8192
