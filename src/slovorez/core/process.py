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
    max_id = max(rev_bies_vocab) if rev_bies_vocab else 0
    prefix_table = np.full(max_id + 1, _S, dtype=np.int8)
    mtype_table  = np.zeros(max_id + 1, dtype=np.int16)

    for tag_id, tag_str in rev_bies_vocab.items():
        prefix_table[tag_id] = _PREFIX_MAP.get(tag_str[:2], _S)
        mtype_table[tag_id]  = morpheme_type_vocab.get(tag_str[2:], 0)

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
        return (tag.split('-', 1) + ['', ''])[:2] if '-' in tag else ('S', tag)
    
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
    segments: list[tuple[str, int, float]] = []
    has_errors = False

    start_idx        = 0
    current_type     = -1
    current_conf_sum = 0.0
    current_len      = 0

    for i in range(len(word)):
        prefix = prefixes[i]
        conf   = confs[i]

        if repair and (prefix == _E or prefix == _I) and current_len == 0:
            has_errors = True
            segments.append((word[i], mtypes[i], conf))
            start_idx = i + 1
            continue

        if prefix == _B:
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
            if current_len > 0:
                has_errors = True
                segments.append((word[start_idx:i], current_type, current_conf_sum / current_len))
            segments.append((word[i], mtypes[i], conf))
            start_idx        = i + 1
            current_conf_sum = 0.0
            current_len      = 0

    if current_len > 0:
        if repair:
            has_errors = True
        segments.append((word[start_idx:], current_type, current_conf_sum / current_len))

    return segments, has_errors


# ---------------------------------------------------------------------------
# SlovorezTokenizer
# ---------------------------------------------------------------------------

class SlovorezTokenizer:

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
        
        # Инициализация матриц ограничений Витерби
        T, start_ok, end_ok = _build_bies_constraints(self.rev_bies_vocab)
        self._trans = np.where(T, 0.0, -1e9).astype(np.float32)
        self._start_ok_mask = np.where(start_ok, 0.0, -1e9).astype(np.float32)
        self._end_ok_mask = np.where(end_ok, 0.0, -1e9).astype(np.float32)

    @classmethod
    def from_config(cls, config: dict) -> SlovorezTokenizer:
        mapping = config["mapping"]
        maxlen  = config["model_specs"]["maxlen"]
        return cls(
            char_vocab=mapping["tokenizer_vocab"],
            bies_vocab=mapping["label2id"],
            maxlen=maxlen,
        )

    def to_config(self) -> dict:
        return {
            "mapping": {
                "tokenizer_vocab": self.char_vocab,
                "label2id":        self.bies_vocab,
            },
            "model_specs": {
                "maxlen": self.maxlen,
            },
        }

    def encode_batch(self, words: list[str]) -> np.ndarray:
        get_char = self.char_vocab.get
        unk_id   = self._unk_id
        if self.do_lower:
            words = [w.lower() for w in words]
        char_tokenized = [[get_char(c, unk_id) for c in w] for w in words]
        return _pad_batch(char_tokenized, self.maxlen)

    def decode_batch(self, encoded: np.ndarray) -> list[str]:
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
        use_viterbi: bool = False
    ) -> list[dict]:
        """Decode logits into rich result dicts with optional Viterbi constraints."""
        
        if use_viterbi:
            log_probs = np.log(np.clip(logits.astype(np.float32), 1e-9, 1.0))
            results: list[dict] = []
            
            for i, word in enumerate(words):
                L = len(word)
                if L == 0:
                    results.append({
                        "word":       word,
                        "morphemes":  [],
                        "confidence": 0.0,
                        "model":      model_name,
                        "repaired":   False,
                        "validated":  False,
                    })
                    continue

                word_logp = log_probs[i, :L]
                path = _viterbi(word_logp, self._trans, self._start_ok_mask, self._end_ok_mask)
                
                word_logits = logits[i, :L]
                confs = word_logits[np.arange(L), path].tolist()
                
                prefixes = self._prefix_table[path].tolist()
                mtypes    = self._mtype_table[path].tolist()
                
                segments, repaired = _decode_word_bies(word, prefixes, mtypes, confs, repair=repair)
                mean_conf   = float(np.round(np.mean(confs), 4)) if confs else 0.0

                results.append({
                    "word":       word,
                    "morphemes":  segments,
                    "confidence": mean_conf,
                    "model":      model_name,
                    "repaired":   repaired,
                    "validated":  False,
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
                segments, repaired = _decode_word_bies(
                    word, prefix_rows[i], mtype_rows[i], conf_rows[i], repair=repair
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
        use_viterbi: bool = True,
    ) -> Generator[tuple[list[tuple[str, int, float]], bool], None, None]:
        """Decode raw model logits into morpheme segments word by word."""
        
        if use_viterbi:
            log_probs = np.log(np.clip(logits.astype(np.float32), 1e-9, 1.0))
            for i, word in enumerate(words):
                L = len(word)
                if L == 0:
                    yield [], False
                    continue

                word_logp = log_probs[i, :L]
                path = _viterbi(word_logp, self._trans, self._start_ok_mask, self._end_ok_mask)
                
                word_logits = logits[i, :L]
                confs = word_logits[np.arange(L), path].tolist()
                
                prefixes = self._prefix_table[path].tolist()
                mtypes    = self._mtype_table[path].tolist()
                
                yield _decode_word_bies(word, prefixes, mtypes, confs, repair=False)
                
        else:
            tag_ids   = np.argmax(logits, axis=-1)
            max_confs = np.max(logits, axis=-1)

            prefix_rows = self._prefix_table[tag_ids].tolist()
            mtype_rows  = self._mtype_table[tag_ids].tolist()
            conf_rows   = max_confs.tolist()

            for word, prefixes, mtypes, confs in zip(words, prefix_rows, mtype_rows, conf_rows):
                yield _decode_word_bies(word, prefixes, mtypes, confs, repair=True)