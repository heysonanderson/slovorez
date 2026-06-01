from __future__ import annotations

import numpy as np
from typing import Generator

from slovorez.core.vocab import PAD_ID, UNK_ID, PAD_TOKEN, UNK_TOKEN
from slovorez.core.vocab.morpheme import MORPHEME_TYPE_VOCAB


# ---------------------------------------------------------------------------
# BIES prefix codes (used instead of string comparison in the hot loop)
# ---------------------------------------------------------------------------

_B = 0
_I = 1
_E = 2
_S = 3

_PREFIX_MAP = {"B-": _B, "I-": _I, "E-": _E, "S-": _S}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_tag_tables(
    rev_bies_vocab: dict[int, str],
    morpheme_type_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute tag_id -> prefix_code / morpheme_type_id as numpy arrays.

    Replaces the dict-based LUT with two flat arrays that can be indexed
    over an entire (batch, seq_len) tag_ids matrix in a single numpy gather,
    eliminating 1.4M per-character dict lookups from the hot path.

    Called once at tokenizer construction time.

    Args:
        rev_bies_vocab:       reverse BIES vocabulary, tag_id -> tag string.
        morpheme_type_vocab:  morpheme type name -> int id mapping.

    Returns:
        prefix_table: int8 array of shape (max_tag_id + 1,),
                      values are _B / _I / _E / _S codes.
        mtype_table:  int16 array of shape (max_tag_id + 1,),
                      values are morpheme type ids.
    """
    max_id = max(rev_bies_vocab) if rev_bies_vocab else 0
    prefix_table = np.full(max_id + 1, _S, dtype=np.int8)
    mtype_table  = np.zeros(max_id + 1, dtype=np.int16)

    for tag_id, tag_str in rev_bies_vocab.items():
        prefix_table[tag_id] = _PREFIX_MAP.get(tag_str[:2], _S)
        mtype_table[tag_id]  = morpheme_type_vocab.get(tag_str[2:], 0)

    return prefix_table, mtype_table


def _pad_batch(tokenized_list: list[list[int]], maxlen: int = 64) -> np.ndarray:
    current_max = max(len(t) for t in tokenized_list)
    actual_len = min(current_max, maxlen)
    arr = np.zeros((len(tokenized_list), actual_len), dtype=np.int32)
    for i, tokens in enumerate(tokenized_list):
        t_len = min(len(tokens), actual_len)
        arr[i, :t_len] = tokens[:t_len]
    return arr


def _decode_word_bies(
    word: str,
    prefixes: list[int],
    mtypes: list[int],
    confs: list[float],
    repair: bool = True,
) -> tuple[list[tuple[str, int, float]], bool]:
    """Decode pre-resolved BIES tags for a single word into morpheme segments.

    Compared to the previous version this function no longer performs any
    dict lookups: ``prefixes`` and ``mtypes`` are plain Python lists that
    were produced upstream by indexing numpy tables over the full batch
    (one vectorised gather instead of one dict.get per character).

    Morpheme text is sliced from the original word string using ``start_idx``
    / ``i`` offsets -- no intermediate char list or ``"".join()`` allocation.
    Confidence is accumulated as a running sum and divided once at segment
    close, avoiding a separate list.

    Args:
        word:     original word string.
        prefixes: per-character BIES prefix codes (_B/_I/_E/_S), length >= len(word).
        mtypes:   per-character morpheme type ids, same length as prefixes.
        confs:    per-character confidence scores, same length as prefixes.
        repair:   if True, handle malformed BIES sequences gracefully.

    Returns:
        Tuple of (segments, has_errors) where segments is a list of
        (morpheme_text, morpheme_type_id, confidence) tuples.
    """
    segments: list[tuple[str, int, float]] = []
    has_errors = False

    start_idx        = 0
    current_type     = -1
    current_conf_sum = 0.0
    current_len      = 0

    for i in range(len(word)):
        prefix = prefixes[i]
        conf   = confs[i]

        # --- repair: E/I without an open segment -> treat as singleton -------
        if repair and (prefix == _E or prefix == _I) and current_len == 0:
            has_errors = True
            segments.append((word[i], mtypes[i], conf))
            start_idx = i + 1
            continue

        if prefix == _B:
            # Flush any open segment that was never closed (malformed).
            if current_len > 0:
                has_errors = True
                segments.append((word[start_idx:i], current_type, current_conf_sum / current_len))
            start_idx        = i
            current_type     = mtypes[i]
            current_conf_sum = conf
            current_len      = 1

        elif prefix == _I:
            current_conf_sum += conf
            current_len      += 1

        elif prefix == _E:
            current_conf_sum += conf
            current_len      += 1
            segments.append((word[start_idx:i + 1], current_type, current_conf_sum / current_len))
            start_idx        = i + 1
            current_conf_sum = 0.0
            current_len      = 0

        else:  # _S
            # Flush any unclosed segment before appending the singleton.
            if current_len > 0:
                has_errors = True
                segments.append((word[start_idx:i], current_type, current_conf_sum / current_len))
            segments.append((word[i], mtypes[i], conf))
            start_idx        = i + 1
            current_conf_sum = 0.0
            current_len      = 0

    # Flush trailing open segment (B..I without closing E).
    if current_len > 0:
        if repair:
            has_errors = True
        segments.append((word[start_idx:], current_type, current_conf_sum / current_len))

    return segments, has_errors


# ---------------------------------------------------------------------------
# SlovorezTokenizer
# ---------------------------------------------------------------------------

class SlovorezTokenizer:
    """Encodes words to character-index tensors and decodes BIES model outputs.

    Linguistic knowledge (char vocab, BIES tag vocab) is model-specific and
    loaded from the model config JSON via ``from_config()``. Direct construction
    is available for custom vocabs or testing.

    Args:
        char_vocab:  mapping char -> int. Loaded from config["mapping"]["tokenizer_vocab"].
        bies_vocab:  mapping BIES-tag -> int. Loaded from config["mapping"]["label2id"].
        maxlen:      maximum sequence length. Loaded from config["model_specs"]["maxlen"].
        do_lower:    lowercase words before encoding. False is recommended --
                     do lowercasing upstream before tokenization for best throughput.
    """

    def __init__(
        self,
        char_vocab: dict[str, int],
        bies_vocab: dict[str, int],
        maxlen: int = 64,
        do_lower: bool = False,
    ):
        self.char_vocab = char_vocab
        self.bies_vocab = bies_vocab
        self.maxlen     = maxlen
        self.do_lower   = do_lower

        self.rev_char_vocab: dict[int, str] = {v: k for k, v in char_vocab.items()}
        self.rev_bies_vocab: dict[int, str] = {v: k for k, v in bies_vocab.items()}

        self._unk_id = char_vocab.get(UNK_TOKEN, UNK_ID)
        self._pad_id = char_vocab.get(PAD_TOKEN, PAD_ID)

        self._prefix_table, self._mtype_table = _build_tag_tables(
            self.rev_bies_vocab, MORPHEME_TYPE_VOCAB
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> SlovorezTokenizer:
        """Instantiate from a model config dict (loaded from JSON).

        Expected keys:
            config["mapping"]["tokenizer_vocab"],
            config["mapping"]["label2id"],
            config["model_specs"]["maxlen"].

        Example::

            config = load_json("models/slovorez-v1/config.json")
            tokenizer = SlovorezTokenizer.from_config(config)
        """
        mapping = config["mapping"]
        maxlen  = config["model_specs"]["maxlen"]
        return cls(
            char_vocab=mapping["tokenizer_vocab"],
            bies_vocab=mapping["label2id"],
            maxlen=maxlen,
        )

    def to_config(self) -> dict:
        """Serialize the tokenizer state back to a config-compatible dict.

        The returned dict is a valid argument to ``from_config()``, so a
        round-trip is guaranteed::

            tokenizer == SlovorezTokenizer.from_config(tokenizer.to_config())

        Primary use: passing tokenizer state to worker processes without
        exposing internal attributes directly.

        Returns:
            Minimal config dict with keys ``mapping`` and ``model_specs``.
        """
        return {
            "mapping": {
                "tokenizer_vocab": self.char_vocab,
                "label2id":        self.bies_vocab,
            },
            "model_specs": {
                "maxlen": self.maxlen,
            },
        }

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_batch(self, words: list[str]) -> np.ndarray:
        """Encode a list of words into a padded int32 matrix of char indices.

        Returns:
            np.ndarray of shape (len(words), min(max_word_len, maxlen)), dtype=int32.
        """
        get_char = self.char_vocab.get
        unk_id   = self._unk_id
        if self.do_lower:
            words = [w.lower() for w in words]
        char_tokenized = [[get_char(c, unk_id) for c in w] for w in words]
        return _pad_batch(char_tokenized, self.maxlen)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_batch(self, encoded: np.ndarray) -> list[str]:
        """Decode a padded char-index matrix back to strings.

        Args:
            encoded: int array of shape (batch, seq_len).

        Returns:
            List of reconstructed word strings.
        """
        get_char = self.rev_char_vocab.get
        return [
            "".join(get_char(idx, "") for idx in row if idx != self._pad_id)
            for row in encoded
        ]

    def decode_predictions_detail(
        self,
        words: list[str],
        logits: np.ndarray,
        model_name: str,
        repair: bool = True,
    ) -> list[dict]:
        """Decode logits into rich result dicts, one per word.

        Optimisations vs. the previous version:

        1. Tag -> (prefix, morpheme_type) lookup is vectorised: two numpy index
           operations over the full (batch, seq_len) tag_ids matrix replace
           one dict.get + tuple unpack per character in the hot loop.

        2. Per-word confidence is computed in bulk with a length mask, replacing
           a Python ``sum()`` slice + ``round()`` call per word.

        Args:
            words:      original word strings passed to encode_batch().
            logits:     float array of shape (batch, seq_len, num_classes).
            model_name: written into every result dict.
            repair:     passed through to _decode_word_bies.

        Returns:
            List of dicts with keys: word, morphemes, confidence, model,
            repaired, validated.
        """
        tag_ids   = np.argmax(logits, axis=-1)   # (batch, seq_len)
        max_confs = np.max(logits, axis=-1)       # (batch, seq_len)

        prefix_rows = self._prefix_table[tag_ids].tolist()   # list[list[int]]
        mtype_rows  = self._mtype_table[tag_ids].tolist()    # list[list[int]]
        conf_rows   = max_confs.tolist()                     # list[list[float]]

        seq_len = max_confs.shape[1]
        lengths = np.fromiter(
            (len(w) for w in words), dtype=np.int64, count=len(words)
        )
        mask      = np.arange(seq_len)[None, :] < lengths[:, None]
        sums      = (max_confs * mask).sum(axis=1)

        word_confs = np.round(
            np.divide(sums, np.where(lengths > 0, lengths, 1)), 4
        )
        word_confs = np.where(lengths > 0, word_confs, 0.0).tolist()

        results: list[dict] = []
        for i, word in enumerate(words):
            segments, repaired = _decode_word_bies(
                word,
                prefix_rows[i],
                mtype_rows[i],
                conf_rows[i],
                repair=repair,
            )
            results.append({
                "word":       word,
                "morphemes":  segments,
                "confidence": word_confs[i],
                "model":      model_name,
                "repaired":   repaired,
                "validated":  False,
            })

        return results

    def decode_predictions(
        self,
        words: list[str],
        logits: np.ndarray,
    ) -> Generator[list[tuple[str, int, float]], None, None]:
        """Decode raw model logits into morpheme segments, word by word.

        Args:
            words:  original word strings -- the same list passed to encode_batch().
                    Passed by reference, no copy is made.
            logits: float array (batch, seq_len, num_classes) -- raw model output.

        Yields:
            For each word: list of (morpheme_text, morpheme_type_id, confidence).

        Example output for "башня"::

            [("баш", 3, 0.91), ("н", 4, 0.76), ("я", 5, 0.88)]
        """
        tag_ids   = np.argmax(logits, axis=-1)
        max_confs = np.max(logits, axis=-1)

        prefix_rows = self._prefix_table[tag_ids].tolist()
        mtype_rows  = self._mtype_table[tag_ids].tolist()
        conf_rows   = max_confs.tolist()

        for word, prefixes, mtypes, confs in zip(
            words, prefix_rows, mtype_rows, conf_rows
        ):
            yield _decode_word_bies(word, prefixes, mtypes, confs)