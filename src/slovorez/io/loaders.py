import os
import json


from slovorez.core.models import morphemes_vocab, OPENCORPORA_TO_UPOS, UPOS, UNK_ID


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def to_json(json_object, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_object, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_tikhonov_txt(path="./data/dictionaries/tikhonov",tags=False):
    if tags:
        from pymorphy3 import MorphAnalyzer
        m = MorphAnalyzer(lang='ru', result_type=None)

    with open(f"{path}.txt", 'r', encoding='utf-8') as f:

        dictionary = {}
        dictionary_full = {}
        morphemes = {}

        affixes = set()

        for line in f.readlines():
            line = line.strip().split()

            if not line:
                continue

            word = line[0]

            pairs = [ pair.split(":") for pair in line[1].split("/")]

            morphemes_obj = [(k, morphemes_vocab[v]) for k, v in pairs ]

            [affixes.add(k) for k, v in pairs if v in ["PREF", "SUFF", "END", "POSTFIX"] and len(k) < 4 and len(k) > 1]
                
            print(f"{word}: {morphemes_obj}")

            if tags:
                parsed = m.parse(word)
                dictionary[word] = {
                    "morphemes": morphemes_obj,
                    "pos": UPOS["POS"].get(OPENCORPORA_TO_UPOS["POS"].get(parsed[0].tag.POS, None), UNK_ID)
                }
                dictionary_full[word] = {
                    "morphemes": morphemes_obj,
                    "pos": UPOS["POS"].get(OPENCORPORA_TO_UPOS["POS"].get(parsed[0].tag.POS, None), UNK_ID),
                    "gender": UPOS["GENDER"].get(OPENCORPORA_TO_UPOS["GENDER"].get(parsed[0].tag.gender, None), UNK_ID),
                    "number": UPOS["NUMBER"].get(OPENCORPORA_TO_UPOS["NUMBER"].get(parsed[0].tag.number, None), UNK_ID),
                    "case": UPOS["CASE"].get(OPENCORPORA_TO_UPOS["CASE"].get(parsed[0].tag.case, None), UNK_ID),
                }
            morphemes[word] = {
                "morphemes": morphemes_obj,
            }
        
        if tags:
            to_json(dictionary, f"{path}.json")
            to_json(dictionary_full, f"{path}-full.json")

        to_json(morphemes, f"{path}-morphemes.json")
        to_json(list(affixes), f"{path}-affixes.json")