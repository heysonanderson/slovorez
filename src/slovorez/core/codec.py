from __future__ import annotations

import numpy as np
from typing import Generator
from pathlib import Path
from typing import Any, Union


from slovorez.config import SlovorezConfig, CalibrationSpec
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
    known_types: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    max_id = max(rev_bies_vocab) if rev_bies_vocab else 0
    prefix_table = np.full(max_id + 1, _S, dtype=np.int8)
    mtype_table  = np.full(max_id + 1, "UNK", dtype=object)

    for tag_id, tag_str in rev_bies_vocab.items():
        prefix_table[tag_id] = _PREFIX_MAP.get(tag_str[:2], _S)
        name = tag_str[2:]
        if name in known_types:
            mtype_table[tag_id] = name

    return prefix_table, mtype_table


def _build_bies_constraints(
    id2tag: dict[int, str], 
    forbidden_ends: set[str] | None = None,
    forbidden_starts: set[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute valid BIES transitions and start/end masks."""
    if forbidden_ends is None:
        forbidden_ends = set(['PAD', 'PREF', 'LINK', 'HYPH'])
    if forbidden_starts is None:
        forbidden_starts = set(['END', 'SUFF', 'PAD', 'LINK', 'HYPH'])

    C = max(id2tag) + 1 if id2tag else 0
    def parse(tag):
        if tag in ("<PAD>", "<UNK>"): return None, None
        return tag.split('-', 1) if '-' in tag else ('S', tag)
    
    P = {i: parse(t) for i, t in id2tag.items()}
    T = np.zeros((C, C), bool)
    start_ok = np.zeros(C, bool)
    end_ok = np.zeros(C, bool)
    
    for i, (p1, t1) in P.items():
        if p1 is None: continue
        

        start_ok[i] = (p1 in ('B', 'S')) and (t1 not in forbidden_starts)
        
        end_ok[i] = (p1 in ('E', 'S')) and (t1 not in forbidden_ends)
        
        for j, (p2, t2) in P.items():
            if p2 is None: continue
            if p1 in ('B', 'I'): 
                T[i, j] = p2 in ('I', 'E') and t2 == t1
            else:                
                T[i, j] = p2 in ('B', 'S')
                
    return T, start_ok, end_ok


def _viterbi(logp: np.ndarray, trans: np.ndarray, start_ok: np.ndarray, end_ok: np.ndarray) -> np.ndarray:
    """Viterbi decoding for a single unpadded word matrix."""
    L, C = logp.shape
    dp = np.full((L, C), -1e9, dtype=np.float32)
    bp = np.zeros((L, C), np.int32)
    
    dp[0] = logp[0] + start_ok
    for t in range(1, L):
        s = dp[t-1][:, None] + trans
        bp[t] = s.argmax(0)
        dp[t] = logp[t] + s[bp[t], np.arange(C)]
        
    path = np.zeros(L, np.int32)
    path[-1] = int((dp[-1] + end_ok).argmax())
    for t in range(L-1, 0, -1): 
        path[t-1] = bp[t, path[t]]
    return path


def _pad_batch(tokenized_list: list[list[int]], maxlen: int = 64) -> np.ndarray:
    if not tokenized_list:
        return np.zeros((0, 0), dtype=np.int32)
    current_max = max(len(t) for t in tokenized_list)
    batch_len = len(tokenized_list)
    actual_len = min(current_max, maxlen)
    arr = np.zeros((batch_len, actual_len), dtype=np.int32)
    lengths = np.zeros(batch_len, dtype=np.int32)
    for i, tokens in enumerate(tokenized_list):
        t_len = min(len(tokens), actual_len)
        if t_len < 1:
            raise ValueError("empty word in batch")
        arr[i, :t_len] = tokens[:t_len]
        lengths[i] = t_len
    return arr, lengths


def _decode_word_bies(
    word: str,
    prefixes: list[int],
    mtypes: list[str],
    confs: list[float]
) -> list[list[str | float]]:
    segments: list[list[str | float]] = []

    start_idx        = 0
    current_type     = "UNK"
    current_conf_sum = 0.0
    current_len      = 0

    for i in range(len(word)):
        prefix = prefixes[i]
        conf   = confs[i]

        if prefix == _B:
            if current_len > 0:
                segments.append([word[start_idx:i], current_type, current_conf_sum / current_len])
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
            segments.append([word[start_idx:i + 1], current_type, current_conf_sum / current_len])
            start_idx        = i + 1
            current_conf_sum = 0.0
            current_len      = 0

        else:  # _S
            if current_len > 0:
                segments.append([word[start_idx:i], current_type, current_conf_sum / current_len])
            segments.append([word[i], mtypes[i], conf])
            start_idx        = i + 1
            current_conf_sum = 0.0
            current_len      = 0

    return segments


def _viterbi_batch(
        logp: np.ndarray,       # B, S, labels
        trans: np.ndarray,      # labels, labels
        start_ok: np.ndarray,   # labels,
        end_ok: np.ndarray,     # labels,
        lengths: np.ndarray     # B, 
) -> np.ndarray: # B, S
    B, S, labels = logp.shape

    if S == 0 or np.any((lengths < 1) | (lengths > S)):
        raise ValueError("lengths must be in the range [1, logp.shape[1]]")
    
    dp = logp[:, 0] + start_ok                                      # B, S[0], label
    final_dp = np.empty_like(dp)   
    single = lengths == 1
    final_dp[single] = dp[single]                                   # B, labels

    bp = np.zeros((B, S, labels), dtype=np.int32)                   # B, S, labels

    for t in range(1, S):
        scores = dp[:, :, None] + trans[None, :, :]                 # (B, labels, 1) + (1, labels, labels) = (B, label, label)
        best = scores.argmax(axis=1)                                # B, labels
        bp[:, t] = best                                             # B, S[t], labels     
        best_scores = scores.max(axis=1)                            # B, label
        dp = best_scores + logp[:, t]                               # (B, labels) + (B, S[t], labels) = (B, labels)
        finished = lengths == t + 1
        final_dp[finished] = dp[finished]                           # B, labels

    path = np.zeros((B, S), dtype=np.int32)
    rows = np.arange(B)
    last = lengths - 1
    final_scores = final_dp + end_ok
    best_last = final_scores.argmax(axis=1)
    # best_path_score = final_scores[rows, best_last]

    path[rows, last] = best_last

    for t in range(S - 1, 0, -1):
        active_rows = rows[lengths > t]

        path[active_rows, t - 1] = bp[
            active_rows, t, path[active_rows, t]
        ]
    return path

# ---------------------------------------------------------------------------
# MorphCodec
# ---------------------------------------------------------------------------

class MorphCodec:

    def __init__(
        self,
        char_vocab: dict[str, int],
        bies_vocab: dict[str, int],
        maxlen: int = 64,
        lowercase: bool = False,
        method: str = "viterbi",
        kind: str = "probs",
        calibration: CalibrationSpec = None
    ):
        self.char_vocab = char_vocab
        self.bies_vocab = bies_vocab
        self.maxlen     = maxlen
        self.lowercase  = lowercase
        self.method     = method
        self.kind       = kind
        self.calibration = calibration

        self.rev_char_vocab: dict[int, str] = {v: k for k, v in char_vocab.items()}
        self.rev_bies_vocab: dict[int, str] = {v: k for k, v in bies_vocab.items()}

        self._unk_id = char_vocab.get(UNK_TOKEN, UNK_ID)
        self._pad_id = char_vocab.get(PAD_TOKEN, PAD_ID)

        self._prefix_table, self._mtype_table = _build_tag_tables(
            self.rev_bies_vocab, MORPHEME_TYPE_VOCAB
        )
        
        # Initialize Viterbi constraint matrices.
        T, start_ok, end_ok = _build_bies_constraints(self.rev_bies_vocab)
        self._trans = np.where(T, 0.0, -1e9).astype(np.float32)
        self._start_ok_mask = np.where(start_ok, 0.0, -1e9).astype(np.float32)
        self._end_ok_mask = np.where(end_ok, 0.0, -1e9).astype(np.float32)

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, Path]) -> SlovorezConfig:
        config = SlovorezConfig.from_pretrained(name_or_path=name_or_path)
        return cls.from_config(config=config)
    
    @classmethod
    def from_config(cls, config: SlovorezConfig) -> MorphCodec:
        return cls(
            char_vocab=config.mapping.tokenizer_vocab,
            bies_vocab=config.mapping.label2id,
            maxlen=config.io.maxlen,
            method=config.decoder.method,
            kind=config.io.outputs[0].kind,
            calibration=config.calibration
        )

    def to_config(self) -> dict:
        return {
            "mapping": {
                "tokenizer_vocab": self.char_vocab,
                "label2id":        self.bies_vocab,
            },
            "io": {
                "maxlen": self.maxlen,
                "outputs": {
                    "name": "morpheme",
                    "kind": self.kind,
                    "dtype": "float32",
                    "shape": [
                        "batch",
                        "length",
                        "labels"
                    ]
                }
            },
            "calibration": None if self.calibration is None else {
                "method": self.calibration.method,
                "a": self.calibration.a
            }
        }

    def encode_batch_padded(self, words: list[str]) -> np.ndarray:
        get_char = self.char_vocab.get
        unk_id   = self._unk_id
        if self.lowercase:
            words = [w.lower() for w in words]
        char_tokenized = [[get_char(c, unk_id) for c in w] for w in words]
        return _pad_batch(char_tokenized, self.maxlen)

    def decode_batch(self, encoded: np.ndarray) -> list[str]:
        get_char = self.rev_char_vocab.get
        return [
            "".join(get_char(idx, "") for idx in row if idx != self._pad_id)
            for row in encoded
        ]

    def decode_predictions_batch(
        self,
        words: list[str],
        logits: np.ndarray,
        lengths: np.ndarray,
        model_name: str
    ) -> list[dict]:
        """Decode logits into rich result dicts with optional Viterbi constraints."""

        if lengths.min() < 1:
            raise ValueError("empty word in batch")
        
        if self.method == "viterbi":
            results: list[dict] = []

            logp = np.log(np.clip(logits.astype(np.float32), 1e-9, 1.0))
            paths = _viterbi_batch(logp, self._trans, self._start_ok_mask, self._end_ok_mask, lengths)

            # per-character confidence: prob of the chosen label at each position
            picked = np.take_along_axis(
                logp, paths[..., None].astype(np.int64), axis=-1
            )[..., 0].astype(np.float64)                       # (B, S)
            char_confs = np.exp(picked)                        # (B, S)

            # word-level confidence = product over real positions (== path_prob)
            inside = np.arange(paths.shape[1])[None, :] < lengths[:, None]
            confs = np.exp(np.where(inside, picked, 0.0).sum(axis=1))

            if self.calibration is not None and self.calibration.method == "power":
                confs = np.float_power(confs, self.calibration.a)

            confs = confs.tolist()

            for i, word in enumerate(words):

                if len(word) != lengths[i]:
                    raise ValueError(
                        f"word length {len(word)} != decoded length {lengths[i]} "
                        f"(word may be truncated by maxlen={self.maxlen})"
                    )

                prefixes = self._prefix_table[paths[i]].tolist()
                mtypes = self._mtype_table[paths[i]].tolist()
                word_char_confs = char_confs[i, :lengths[i]].tolist()

                segments = _decode_word_bies(word, prefixes, mtypes, word_char_confs)

                results.append({
                    "word":       word,
                    "morphemes":  segments,
                    "confidence": confs[i],
                    "model":      model_name
                })
            return results
            
        else:
            tag_ids   = np.argmax(logits, axis=-1)
            max_confs = np.max(logits, axis=-1)

            prefix_rows = self._prefix_table[tag_ids].tolist()
            mtype_rows  = self._mtype_table[tag_ids].tolist()
            conf_rows   = max_confs.tolist()

            seq_len = max_confs.shape[1]
            lengths = np.fromiter((len(w) for w in words), dtype=np.int64, count=len(words))
            mask      = np.arange(seq_len)[None, :] < lengths[:, None]
            sums      = (max_confs * mask).sum(axis=1)

            word_confs = np.round(np.divide(sums, np.where(lengths > 0, lengths, 1)), 4)
            word_confs = np.where(lengths > 0, word_confs, 0.0).tolist()

            results: list[dict] = []
            for i, word in enumerate(words):
                segments = _decode_word_bies(
                    word, prefix_rows[i], mtype_rows[i], conf_rows[i]
                )
                results.append({
                    "word":       word,
                    "morphemes":  segments,
                    "confidence": word_confs[i],
                    "model":      model_name
                })
            return results

    def path_prob(self, scores: np.ndarray, paths: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        if self.kind != "probs":
            raise NotImplementedError("path_prob of a CRF needs log Z (forward algorithm)")
        lp = np.log(np.clip(scores.astype(np.float32), 1e-9, 1.0))
        picked = np.take_along_axis(lp, paths[..., None].astype(np.int64), axis=-1)[..., 0].astype(np.float64)
        inside = np.arange(paths.shape[1])[None, :] < lengths[:, None]
        return np.exp((picked * inside).sum(axis=1))