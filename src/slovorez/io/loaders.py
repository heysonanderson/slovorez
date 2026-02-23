import os
import json


from slovorez.core.models import morphemes_vocab, OPENCORPORA_TO_UPOS, UPOS, UNK_ID


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def to_json(json_object, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_object, f, indent=2, ensure_ascii=False)

def load_json(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)