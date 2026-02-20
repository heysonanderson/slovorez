from dataclasses import dataclass, field
from typing import Dict, List
from enum import IntEnum


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1

pos = {
    PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
    "ADJ": 2, "ADP": 3, "ADV": 4, "AUX": 5, "CCONJ": 6, "DET": 7, 
    "INTJ": 8, "NOUN": 9, "NUM": 10, "PART": 11, "PRON": 12, 
    "PROPN": 13, "PUNCT": 14, "SCONJ": 15, "SYM": 16, "VERB": 17, "X": 18
}

morphemes = {
    PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
    "PREF": 2, "ROOT": 3, "SUFF": 4, "END": 5, "POSTFIX": 6, "LINK": 7, "HYPH": 8
}

morphemes_vocab = morphemes

morphemes_bies= {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,
    "B-PREF": 2, "I-PREF": 3, "E-PREF": 4, "S-PREF": 5,
    "B-ROOT": 6, "I-ROOT": 7, "E-ROOT": 8, "S-ROOT": 9,
    "B-SUFF": 10, "I-SUFF": 11, "E-SUFF": 12, "S-SUFF": 13,
    "B-END": 14, "I-END": 15, "E-END": 16, "S-END": 17,
    "B-POSTFIX": 18, "I-POSTFIX": 19, "E-POSTFIX": 20, "S-POSTFIX": 21,
    "B-LINK": 22, "I-LINK": 23, "E-LINK": 24, "S-LINK": 25,
    "B-HYPH": 26, "I-HYPH": 27, "E-HYPH": 28, "S-HYPH": 29
}

class PosVocab(IntEnum):
    PAD = 0
    UNK = 1
    ADJ = 2
    ADP = 3
    ADV = 4
    AUX = 5
    CCONJ = 6
    DET = 7
    INTJ = 8
    NOUN = 9
    NUM = 10
    PART = 11
    PRON = 12
    PROPN = 13
    PUNCT = 14
    SCONJ = 15
    SYM = 16
    VERB = 17
    X = 18

@dataclass
class WordAnalysis:
    pos: int
    gender: int
    number: int
    case: int
    morphemes: List[tuple[str, int]]

@dataclass
class Token:
    text: str
    lookup_key: str

@dataclass
class TokenStream:
    name: str
    tokens: List[Token] = field(default_factory=list)




PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1



# --- UPOS

UPOS = {
    "POS": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "ADJ": 2, "ADP": 3, "ADV": 4, "AUX": 5, "CCONJ": 6, "DET": 7,
        "INTJ": 8, "NOUN": 9, "NUM": 10, "PART": 11, "PRON": 12,
        "PROPN": 13, "PUNCT": 14, "SCONJ": 15, "SYM": 16, "VERB": 17, "X": 18
    },
    "GENDER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "Gender=Masc": 2, "Gender=Fem": 3, "Gender=Neut": 4
    },
    "NUMBER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "Number=Sing": 2, "Number=Plur": 3
    },
    "CASE": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "Case=Nom": 2, "Case=Gen": 3, "Case=Dat": 4, "Case=Acc": 5,
        "Case=Ins": 6, "Case=Loc": 7, "Case=Voc": 8
    }
}



# --- RUCORPORA

RUCORPORA = {
    "POS": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "S": 2, "A": 3, "NUM": 4, "ANUM": 5, "V": 6, "ADV": 7,
        "PRAEDIC": 8, "PARENTH": 9, "SPRO": 10, "APRO": 11, "ADVPRO": 12,
        "PRAEDICPRO": 13, "PR": 14, "CONJ": 15, "PART": 16, "INTJ": 17,
        "INIT": 18, "NONLEX": 19
    },
    "GENDER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "m": 2, "f": 3, "m-f": 4, "n": 5
    },
    "NUMBER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "sg": 2, "pl": 3
    },
    "CASE": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "nom": 2, "gen": 3, "dat": 4, "acc": 5, "ins": 6, "loc": 7,
        "gen2": 8, "acc2": 9, "loc2": 10, "voc": 11, "adnum": 12
    }
}



# --- OPENCORPORA

OPENCORPORA = {
    "POS": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "NOUN": 2, "ADJF": 3, "ADJS": 4, "COMP": 5, "VERB": 6,
        "INFN": 7, "PRTF": 8, "PRTS": 9, "GRND": 10, "NUMR": 11,
        "ADVB": 12, "NPRO": 13, "PRED": 14, "PREP": 15, "CONJ": 16,
        "PRCL": 17, "INTJ": 18
    },
    "GENDER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "masc": 2, "femn": 3, "neut": 4, "ms-f": 5
    },
    "NUMBER": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "sing": 2, "plur": 3
    },
    "CASE": {
        PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        "nomn": 2, "gent": 3, "datv": 4, "accs": 5, "ablt": 6, "loct": 7,
        "voct": 8, "gen2": 9, "acc2": 10, "loc2": 11
    }
}



# --- MAPPINGS

RUCORPORA_TO_UPOS = {
    "POS": {
        "S": "NOUN", "A": "ADJ", "NUM": "NUM", "ANUM": "ADJ", "V": "VERB", 
        "ADV": "ADV", "PRAEDIC": "ADV", "PARENTH": "ADV", "SPRO": "PRON", 
        "APRO": "DET", "ADVPRO": "ADV", "PRAEDICPRO": "PRON", "PR": "ADP", 
        "CONJ": "CCONJ", "PART": "PART", "INTJ": "INTJ", "INIT": "PROPN", "NONLEX": "X"
    },
    "GENDER": {"m": "Gender=Masc", "f": "Gender=Fem", "n": "Gender=Neut", "m-f": "Gender=Masc"},
    "NUMBER": {"sg": "Number=Sing", "pl": "Number=Plur"},
    "CASE": {
        "nom": "Case=Nom", "gen": "Case=Gen", "gen2": "Case=Gen", "dat": "Case=Dat",
        "acc": "Case=Acc", "acc2": "Case=Acc", "ins": "Case=Ins", "loc": "Case=Loc",
        "loc2": "Case=Loc", "voc": "Case=Voc", "adnum": "Case=Gen"
    }
}

OPENCORPORA_TO_UPOS = {
    "POS": {
        "NOUN": "NOUN", "ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ", "VERB": "VERB",
        "INFN": "VERB", "PRTF": "VERB", "PRTS": "VERB", "GRND": "VERB", "NUMR": "NUM",
        "ADVB": "ADV", "NPRO": "PRON", "PRED": "ADV", "PREP": "ADP", "CONJ": "CCONJ",
        "PRCL": "PART", "INTJ": "INTJ"
    },
    "GENDER": {"masc": "Gender=Masc", "femn": "Gender=Fem", "neut": "Gender=Neut", "ms-f": "Gender=Masc"},
    "NUMBER": {"sing": "Number=Sing", "plur": "Number=Plur"},
    "CASE": {
        "nomn": "Case=Nom", "gent": "Case=Gen", "gen2": "Case=Gen", "datv": "Case=Dat",
        "accs": "Case=Acc", "acc2": "Case=Acc", "ablt": "Case=Ins", "loct": "Case=Loc",
        "loc2": "Case=Loc", "voct": "Case=Voc"
    }
}

CHAR_VOCAB = {
    # Базовые специальные токены (0-3)
    PAD_TOKEN: PAD_ID,    # 0 - padding
    UNK_TOKEN: UNK_ID,    # 1 - unknown character
    "EOW": 2,    # 2 - end of word
    "MASK": 3,   # 3 - mask token для MLM
    
    # Русские буквы (строчные) (4-36)
    "а": 4, "б": 5, "в": 6, "г": 7, "д": 8, "е": 9,
    "ё": 10, "ж": 11, "з": 12, "и": 13, "й": 14,
    "к": 15, "л": 16, "м": 17, "н": 18, "о": 19,
    "п": 20, "р": 21, "с": 22, "т": 23, "у": 24,
    "ф": 25, "х": 26, "ц": 27, "ч": 28, "ш": 29,
    "щ": 30, "ъ": 31, "ы": 32, "ь": 33, "э": 34,
    "ю": 35, "я": 36,
    
    # Цифры (37-46)
    "0": 37, "1": 38, "2": 39, "3": 40, "4": 41,
    "5": 42, "6": 43, "7": 44, "8": 45, "9": 46,
    
    # Пунктуация и символы (47-55)
    " ": 47,   # пробел
    "-": 48,   # дефис
    ".": 49,   # точка
    ",": 50,   # запятая
    "!": 51,   # восклицательный знак
    "?": 52,   # вопросительный знак
    ":": 53,   # двоеточие
    ";": 54,   # точка с запятой
    "'": 55,   # апостроф
}