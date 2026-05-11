from .base import PAD_ID, UNK_ID, PAD_TOKEN, UNK_TOKEN

# ---------------------------------------------------------------------------
# UPOS (Universal Dependencies)
# ---------------------------------------------------------------------------

UPOS: dict[str, dict[str, int]] = {
    "POS": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "ADJ": 2, "ADP": 3, "ADV": 4, "AUX": 5, "CCONJ": 6, "DET": 7,
        "INTJ": 8, "NOUN": 9, "NUM": 10, "PART": 11, "PRON": 12,
        "PROPN": 13, "PUNCT": 14, "SCONJ": 15, "SYM": 16, "VERB": 17, "X": 18,
    },
    "GENDER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "Gender=Masc": 2, "Gender=Fem": 3, "Gender=Neut": 4,
    },
    "NUMBER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "Number=Sing": 2, "Number=Plur": 3,
    },
    "CASE": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "Case=Nom": 2, "Case=Gen": 3, "Case=Dat": 4, "Case=Acc": 5,
        "Case=Ins": 6, "Case=Loc": 7, "Case=Voc": 8,
    },
}

# ---------------------------------------------------------------------------
# RuCorpora
# ---------------------------------------------------------------------------

RUCORPORA: dict[str, dict[str, int]] = {
    "POS": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "S": 2, "A": 3, "NUM": 4, "ANUM": 5, "V": 6, "ADV": 7,
        "PRAEDIC": 8, "PARENTH": 9, "SPRO": 10, "APRO": 11, "ADVPRO": 12,
        "PRAEDICPRO": 13, "PR": 14, "CONJ": 15, "PART": 16, "INTJ": 17,
        "INIT": 18, "NONLEX": 19,
    },
    "GENDER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "m": 2, "f": 3, "m-f": 4, "n": 5,
    },
    "NUMBER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "sg": 2, "pl": 3,
    },
    "CASE": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "nom": 2, "gen": 3, "dat": 4, "acc": 5, "ins": 6, "loc": 7,
        "gen2": 8, "acc2": 9, "loc2": 10, "voc": 11, "adnum": 12,
    },
}

# ---------------------------------------------------------------------------
# OpenCorpora
# ---------------------------------------------------------------------------

OPENCORPORA: dict[str, dict[str, int]] = {
    "POS": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "NOUN": 2, "ADJF": 3, "ADJS": 4, "COMP": 5, "VERB": 6,
        "INFN": 7, "PRTF": 8, "PRTS": 9, "GRND": 10, "NUMR": 11,
        "ADVB": 12, "NPRO": 13, "PRED": 14, "PREP": 15, "CONJ": 16,
        "PRCL": 17, "INTJ": 18,
    },
    "GENDER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "masc": 2, "femn": 3, "neut": 4, "ms-f": 5,
    },
    "NUMBER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "sing": 2, "plur": 3,
    },
    "CASE": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "nomn": 2, "gent": 3, "datv": 4, "accs": 5, "ablt": 6, "loct": 7,
        "voct": 8, "gen2": 9, "acc2": 10, "loc2": 11,
    },
}

# ---------------------------------------------------------------------------
# Cross-standard mappings -- UPOS
# ---------------------------------------------------------------------------

RUCORPORA_TO_UPOS: dict[str, dict[str, str]] = {
    "POS": {
        "S": "NOUN", "A": "ADJ", "NUM": "NUM", "ANUM": "ADJ", "V": "VERB",
        "ADV": "ADV", "PRAEDIC": "ADV", "PARENTH": "ADV", "SPRO": "PRON",
        "APRO": "DET", "ADVPRO": "ADV", "PRAEDICPRO": "PRON", "PR": "ADP",
        "CONJ": "CCONJ", "PART": "PART", "INTJ": "INTJ", "INIT": "PROPN", "NONLEX": "X",
    },
    "GENDER": {"m": "Gender=Masc", "f": "Gender=Fem", "n": "Gender=Neut", "m-f": "Gender=Masc"},
    "NUMBER": {"sg": "Number=Sing", "pl": "Number=Plur"},
    "CASE": {
        "nom": "Case=Nom", "gen": "Case=Gen", "gen2": "Case=Gen", "dat": "Case=Dat",
        "acc": "Case=Acc", "acc2": "Case=Acc", "ins": "Case=Ins", "loc": "Case=Loc",
        "loc2": "Case=Loc", "voc": "Case=Voc", "adnum": "Case=Gen",
    },
}

OPENCORPORA_TO_UPOS: dict[str, dict[str, str]] = {
    "POS": {
        "NOUN": "NOUN", "ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ",
        "PRTF": "ADJ",  "PRTS": "ADJ", "VERB": "VERB", "INFN": "VERB",
        "GRND": "VERB",  "NUMR": "NUM", "ADVB": "ADV",  "NPRO": "PRON",
        "PRED": "ADV",   "PREP": "ADP", "CONJ": "CCONJ","PRCL": "PART",
        "INTJ": "INTJ",
    },
    "GENDER": {"masc": "Gender=Masc", "femn": "Gender=Fem", "neut": "Gender=Neut", "ms-f": "Gender=Masc"},
    "NUMBER": {"sing": "Number=Sing", "plur": "Number=Plur"},
    "CASE": {
        "nomn": "Case=Nom", "gent": "Case=Gen", "gen2": "Case=Gen", "datv": "Case=Dat",
        "accs": "Case=Acc", "acc2": "Case=Acc", "ablt": "Case=Ins", "loct": "Case=Loc",
        "loc2": "Case=Loc", "voct": "Case=Voc",
    },
}
