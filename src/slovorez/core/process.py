from __future__ import annotations

import numpy as np
from typing import Generator

from slovorez.core.vocab import PAD_ID, UNK_ID, PAD_TOKEN, UNK_TOKEN
from slovorez.core.vocab.morpheme import MORPHEME_TYPE_VOCAB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    tag_ids: np.ndarray,
    confidences: np.ndarray,
    rev_bies_vocab: dict[int, str],
    morpheme_type_vocab: dict[str, int],
    repair: bool = True
) -> tuple[list[tuple[str, int, float]], bool]:
    segments = []
    current_morpheme = ""
    current_type = -1
    current_confs = []
    has_errors = False

    word_len = len(word)
    active_tags = tag_ids[:word_len]
    active_confs = confidences[:word_len]

    for i, (char, tag_id) in enumerate(zip(word, active_tags)):
        tag = rev_bies_vocab.get(tag_id, "S-ROOT")
        prefix = tag[:2]
        m_type_str = tag[2:]
        m_type_id = morpheme_type_vocab.get(m_type_str, 0)
        conf = float(active_confs[i])

        if repair and prefix in ("E-", "I-") and not current_morpheme:
            has_errors = True
            segments.append((char, m_type_id, conf))
            continue

        if prefix == "B-":
            if current_morpheme:
                has_errors = True
                segments.append((current_morpheme, current_type, float(np.mean(current_confs))))
            current_morpheme = char
            current_type = m_type_id
            current_confs = [conf]

        elif prefix == "I-":
            current_morpheme += char
            current_confs.append(conf)

        elif prefix == "E-":
            current_morpheme += char
            current_confs.append(conf)
            segments.append((current_morpheme, current_type, float(np.mean(current_confs))))
            current_morpheme = ""
            current_confs = []

        elif prefix == "S-":
            if current_morpheme:
                has_errors = True
                segments.append((current_morpheme, current_type, float(np.mean(current_confs))))
                current_morpheme = ""
            segments.append((char, m_type_id, conf))

    if current_morpheme:
        if repair: has_errors = True
        segments.append((current_morpheme, current_type, float(np.mean(current_confs))))

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
        char_vocab:  mapping char -- int. Loaded from config["mapping"]["tokenizer_vocab"].
        bies_vocab:  mapping BIES-tag -- int. Loaded from config["mapping"]["label2id"].
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
        self.maxlen = maxlen
        self.do_lower = do_lower

        self.rev_char_vocab: dict[int, str] = {v: k for k, v in char_vocab.items()}
        self.rev_bies_vocab: dict[int, str] = {v: k for k, v in bies_vocab.items()}

        self._unk_id = char_vocab.get(UNK_TOKEN, UNK_ID)
        self._pad_id = char_vocab.get(PAD_TOKEN, PAD_ID)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> SlovorezTokenizer:
        """Instantiate from a model config dict (loaded from JSON).

        Expected keys: config["mapping"]["tokenizer_vocab"],
                       config["mapping"]["label2id"],
                       config["model_specs"]["maxlen"].

        Example::

            config = load_json("slovorez-v1.0.json")
            tokenizer = SlovorezTokenizer.from_config(config)
        """
        mapping = config["mapping"]
        maxlen = config["model_specs"]["maxlen"]
        return cls(
            char_vocab=mapping["tokenizer_vocab"],
            bies_vocab=mapping["label2id"],
            maxlen=maxlen,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_batch(self, words: list[str]) -> np.ndarray:
        """Encode a list of words into a padded int32 matrix of char indices.

        Returns:
            np.ndarray of shape (len(words), min(max_word_len, maxlen)), dtype=int32.
        """
        get_char = self.char_vocab.get
        unk_id = self._unk_id
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
        repair: bool = True
    ) -> Generator[dict, None, None]:
        tag_ids = np.argmax(logits, axis=-1)
        max_confs = np.max(logits, axis=-1)

        for i, word in enumerate(words):
            segments, repaired = _decode_word_bies(
                word, 
                tag_ids[i], 
                max_confs[i], 
                self.rev_bies_vocab, 
                MORPHEME_TYPE_VOCAB,
                repair=repair
            )
            
            word_conf = float(np.mean(max_confs[i, :len(word)]))

            yield {
                "word": word,
                "morphemes": segments,
                "confidence": round(word_conf, 4),
                "model": model_name,
                "repaired": repaired,
                "validated": False
            }

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

        Example output for "башня":

            [("баш", 3, 0.91), ("н", 4, 0.76), ("я", 5, 0.88)]
        """
        tag_ids   = np.argmax(logits, axis=-1)
        max_confs = np.max(logits, axis=-1)

        for word, word_tag_ids, word_confs in zip(words, tag_ids, max_confs):
            yield _decode_word_bies(
                list(word),
                word_tag_ids,
                word_confs,
                self.rev_bies_vocab,
                MORPHEME_TYPE_VOCAB,
            )
