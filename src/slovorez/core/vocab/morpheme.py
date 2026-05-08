from enum import IntEnum
from .base import PAD_ID, UNK_ID, PAD_TOKEN, UNK_TOKEN


class MorphemeType(IntEnum):
    """Morpheme type IDs.
    """
    PAD      = 0
    UNK      = 1
    PREF     = 2
    ROOT     = 3
    SUFF     = 4
    END      = 5
    POSTFIX  = 6
    LINK     = 7
    HYPH     = 8


MORPHEME_TYPE_VOCAB: dict[str, int] = {m.name: m.value for m in MorphemeType}
MORPHEME_TYPE_VOCAB[PAD_TOKEN] = PAD_ID
MORPHEME_TYPE_VOCAB[UNK_TOKEN] = UNK_ID

REV_MORPHEME_TYPE_VOCAB: dict[int, str] = {v: k for k, v in MORPHEME_TYPE_VOCAB.items()}
