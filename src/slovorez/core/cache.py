from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_FLUSH_SIZE = 8192

# ---------------------------------------------------------------------------
# Confidence threshold for automatic promotion to validated dict
# ---------------------------------------------------------------------------

_VALIDATED_CONFIDENCE_THRESHOLD = 0.85


# ===========================================================================
# PersistenceIndex
# ===========================================================================

class PersistenceIndex:
    """Tracks which words have already been processed, across sessions.

    The seen-set is the single source of truth for deduplication. It is built
    at startup by scanning the JSONL log file (words only -- morphemes are not
    loaded). New words are registered via ``mark_seen()``.

    This class is intentionally lightweight so it can be serialized to a
    ``frozenset`` and passed to worker processes without carrying any heavy
    state (morpheme data, file handles, etc.).

    Owns no I/O -- writing is delegated to ``LogWriter``.

    Args:
        min_len: minimum word length accepted for inference.
        max_len: maximum word length accepted for inference.

    Example::

        index = PersistenceIndex.from_jsonl("predictions.jsonl")
        snapshot = index.snapshot()           # frozenset -- safe to pickle
        unseen = index.filter_unseen(words)
        index.mark_seen(unseen)
    """

    def __init__(self, min_len: int = 1, max_len: int = 64):
        self._seen:  set[str] = set()
        self.min_len = min_len
        self.max_len = max_len

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        min_len: int = 1,
        max_len: int = 64,
    ) -> PersistenceIndex:
        """Build an index by scanning word keys from an existing JSONL file.

        Only the ``"word"`` field is read per line -- morpheme data is ignored.
        Malformed lines are skipped with a warning so a partially-written file
        does not block startup.

        Args:
            path:    path to the JSONL log file (need not exist yet).
            min_len: forwarded to the constructor.
            max_len: forwarded to the constructor.

        Returns:
            A populated ``PersistenceIndex`` instance.
        """
        index = cls(min_len=min_len, max_len=max_len)
        p = Path(path)

        if not p.is_file():
            return index

        loaded = skipped = 0

        with open(p, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    index._seen.add(json.loads(line)["word"])
                    loaded += 1
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        f"Skipping malformed line {lineno} in {p.name}"
                    )
                    skipped += 1

        logger.info(
            f"PersistenceIndex: loaded {loaded:,} keys from {p.name}"
            + (f" ({skipped} lines skipped)" if skipped else "")
        )
        return index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_unseen(self, words: list[str]) -> list[str]:
        """Return words not yet seen, deduped and sorted by length.

        Length filtering is applied here. Words from external lists (e.g.
        base dictionary) must be filtered out by the caller before this call.

        Args:
            words: lowercased word strings from the current batch.

        Returns:
            Sorted list of new unique words ready for inference.
        """
        seen    = self._seen
        min_len = self.min_len
        max_len = self.max_len
        unseen  = {w for w in words if min_len <= len(w) <= max_len and w not in seen}
        return sorted(unseen, key=len)

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

    def reload_from_jsonl(self, path: Union[str, Path]) -> None:
        """Rescan the JSONL file and merge any new keys into the seen-set.

        Intended to be called after a multiprocessing run completes, so the
        main-process index reflects results written by worker processes.

        Args:
            path: path to the JSONL log file.
        """
        fresh = PersistenceIndex.from_jsonl(path, self.min_len, self.max_len)
        self._seen.update(fresh._seen)

    def __len__(self) -> int:
        return len(self._seen)


# ===========================================================================
# LogWriter
# ===========================================================================

class LogWriter:
    """Buffered append-only writer for JSONL prediction logs.

    Accumulates result dicts in memory and flushes to disk either when the
    buffer reaches ``_FLUSH_SIZE`` or when ``flush()`` is called explicitly.

    Owns no deduplication logic -- that is ``PersistenceIndex``'s job.
    Owns no morpheme lookup -- that is ``MorphemeRegistry``'s job.

    Args:
        path: path to the JSONL output file. Parent directories are created
              automatically.

    Example::

        writer = LogWriter("predictions.jsonl")
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


# ===========================================================================
# MorphemeRegistry
# ===========================================================================

class MorphemeRegistry:
    """In-memory store for morpheme segmentation results.

    Merges two sources of truth:

    * **base_dict** -- a static, pre-validated dictionary loaded from JSON at
      startup. Words in this dict are never sent to the model.
    * **validated_dict** -- accumulated at runtime from model predictions whose
      per-word confidence exceeds ``confidence_threshold``. High-confidence
      results are promoted automatically.

    Lookup returns morphemes for any word present in either dict. The registry
    does *not* participate in deduplication (that is ``PersistenceIndex``'s
    job) and does *not* write to disk (that is ``LogWriter``'s job).

    This class is not meant to be used in worker processes -- it lives only in
    the main process alongside the orchestrator.

    Args:
        confidence_threshold: minimum per-word confidence for automatic
            promotion to ``validated_dict``. Defaults to 0.85.

    Example::

        registry = MorphemeRegistry.from_base_dict("base_dict.json")
        registry.register(results)
        morphemes = registry.lookup("башня")
    """

    def __init__(self, confidence_threshold: float = _VALIDATED_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self._base_dict: dict[str, list[tuple]] = {}
        self._validated_dict: dict[str, list[tuple]] = {}
        self._base_dict_keys: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_base_dict(
        cls,
        path: Union[str, Path],
        confidence_threshold: float = _VALIDATED_CONFIDENCE_THRESHOLD,
    ) -> MorphemeRegistry:
        """Load a static base dictionary from a JSON file.

        The JSON file must be a flat object mapping word strings to their
        pre-validated morpheme lists::

            {"башня": [["баш", 3, 0.99], ["н", 4, 0.98], ["я", 5, 0.99]], ...}

        Args:
            path:                 path to the JSON base dictionary file.
            confidence_threshold: forwarded to the constructor.

        Returns:
            A ``MorphemeRegistry`` instance with the base dict pre-loaded.
        """
        registry = cls(confidence_threshold=confidence_threshold)
        p = Path(path)

        with open(p, "r", encoding="utf-8") as f:
            raw: dict = json.load(f)

        registry._base_dict      = raw
        registry._base_dict_keys = frozenset(raw)  # built exactly once

        logger.info(
            f"MorphemeRegistry: loaded {len(registry._base_dict):,} entries "
            f"from {p.name}"
        )
        return registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def base_dict_keys(self) -> frozenset[str]:
        """Immutable set of base dictionary word keys.

        Use this to pre-filter word batches so base-dict words are never
        sent to the model. The frozenset is built once at load time and
        never reallocated.
        """
        return self._base_dict_keys

    def lookup(self, word: str) -> list[tuple] | None:
        """Return the morpheme list for a word from either dict, or None.

        Checks ``validated_dict`` first (runtime results), then ``base_dict``
        (static). Returns None if the word is absent from both.

        Args:
            word: lowercased word string.

        Returns:
            List of (morpheme_text, morpheme_type_id, confidence) tuples,
            or None if not found.
        """
        return self._validated_dict.get(word) or self._base_dict.get(word)

    def register(self, results: list[dict]) -> None:
        """Ingest new model predictions and promote high-confidence ones.

        A result is promoted to ``validated_dict`` when its ``"confidence"``
        field is >= ``self.confidence_threshold``.

        Args:
            results: list of prediction dicts with at least ``"word"``,
                     ``"morphemes"``, and ``"confidence"`` keys.
        """
        threshold = self.confidence_threshold
        validated = self._validated_dict

        for r in results:
            if r.get("confidence", 0.0) >= threshold:
                validated[r["word"]] = r["morphemes"]

    def __contains__(self, word: str) -> bool:
        """Return True if the word is in either the base or validated dict."""
        return word in self._base_dict or word in self._validated_dict

    def __len__(self) -> int:
        """Total number of entries across both dicts (base + validated)."""
        return len(self._base_dict) + len(self._validated_dict)