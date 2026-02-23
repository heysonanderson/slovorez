from slovorez.io.loaders import *
from slovorez.core.models import morphemes_vocab, OPENCORPORA_TO_UPOS, UPOS, UNK_ID


def create_morphemes_list(path="./data/dictionaries/ml"):
    morphemes = load_json(f"{path}-morphemes.json")
    morphemes_list = {}
    morphemes_list = {k: {} for k in morphemes_vocab.keys()}

    reverse_morphemes_vocab = {v: k for k, v in morphemes_vocab.items()}
    
    for word, word_data in morphemes.items():
        pairs = word_data["morphemes"]
        for morpheme, morph_type in pairs:
            morph_type = reverse_morphemes_vocab[morph_type]

            if morpheme not in morphemes_list[morph_type]:
                morphemes_list[morph_type][morpheme] = {
                    "count": 0
                }

            morphemes_list[morph_type][morpheme]["count"] += 1
            root_variants = {}

            if morph_type == "ROOT":
                if "words" not in morphemes_list[morph_type][morpheme]:
                    morphemes_list[morph_type][morpheme]["words"] = []
                morphemes_list[morph_type][morpheme]["words"].append(word)
                
                for i in range(2, len(morpheme)):
                    first_part = morpheme[:i]
                    sec_part = morpheme[i:]
                    if (is_morpheme_type(first_part, "ROOT", morphemes_list) and 
                        is_morpheme_type(sec_part, "SUFF", morphemes_list) and 
                        len(first_part) > 1):
                        if morpheme not in root_variants:
                            root_variants[morpheme] = []
                        root_variants[morpheme].append([(first_part, "ROOT"), (sec_part, "SUFF")])
                    
                    if (is_morpheme_type(first_part, "PREF", morphemes_list) and 
                        is_morpheme_type(sec_part, "ROOT", morphemes_list) and 
                        len(sec_part) > 1):
                        if morpheme not in root_variants:
                            root_variants[morpheme] = []
                        root_variants[morpheme].append([(first_part, "PREF"), (sec_part, "ROOT")])
                    
                    if (is_morpheme_type(first_part, "ROOT", morphemes_list) and 
                        is_morpheme_type(sec_part, "ROOT", morphemes_list) and 
                        len(sec_part) > 2 and len(first_part) > 2):
                        if morpheme not in root_variants:
                            root_variants[morpheme] = []
                        root_variants[morpheme].append([(first_part, "ROOT"), (sec_part, "ROOT")])
        
        if root_variants:
            for root in root_variants.keys():
                morphemes_list["ROOT"][root]["variants"] = root_variants[root]

    counts = []
    for n in morphemes_list.keys():
        counts.append(f"{n}: {len(morphemes_list[n])}")

    counts = []
    for n in morphemes_list.keys():
        counts.append(f"{n}: {len(morphemes_list[n])}")

    print("\n".join(counts))

    morphemes_list["ROOT"] = dict(sorted(morphemes_list["ROOT"].items(), key=lambda item: item[0]))
    to_json(morphemes_list, f"{path}-morphemes-list.json")

def is_morpheme_type(morpheme, type, morphemes_list):
    return morpheme in morphemes_list[type].keys()


create_morphemes_list()


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

            morphemes_obj = [ (k, morphemes_vocab[v]) for k, v in pairs ]

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

