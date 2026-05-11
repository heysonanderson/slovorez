import os
import re
import copy
from slovorez.io.loaders import *
from slovorez.core.models import morphemes_vocab, OPENCORPORA_TO_UPOS, UPOS, UNK_ID, rev_morphemes_vocab, rev_pos
from collections import Counter, defaultdict


def log_to_remarks(regmorph, rt, tt, count, corrected_words, remarks_path="./data/dictionaries/static_dictionary/remarks.md"):
    next_num = 1
    if os.path.exists(remarks_path):
        with open(remarks_path, 'r', encoding='utf-8') as f:
            content = f.read()
            nums = re.findall(r'^(\d+)\.', content, re.MULTILINE)
            if nums:
                next_num = max(int(n) for n in nums) + 1

    rt_name = rev_morphemes_vocab[rt]
    tt_name = rev_morphemes_vocab[tt]
    header = f"{next_num}. Исправления ({count}) в '{regmorph}' {rt_name} ---> {tt_name}, результат:\n"

    lines = []
    for word, word_data in corrected_words:
        display_data = copy.deepcopy(word_data)
        display_data["morphemes"] = [[m, rev_morphemes_vocab[t]] for m, t in display_data["morphemes"]]
        if isinstance(display_data["pos"], int):
            display_data["pos"] = rev_pos.get(display_data["pos"], display_data["pos"])
            
        lines.append(f" - {word} {display_data}")

    with open(remarks_path, 'a', encoding='utf-8') as f:
        f.write(header + "\n".join(lines) + "\n\n")

def get_detailed_stats(remarks_path="./data/dictionaries/static_dictionary/remarks.md"):
    if not os.path.exists(remarks_path):
        return "Файл не найден."

    pattern = r"Исправления\s*\((\d+)\)\s*в\s*'([^']+)'\s*(\w+)\s*--->\s*(\w+)"
    
    total_corrected = 0
    transitions = Counter()
    morpheme_stats = Counter()

    with open(remarks_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                count = int(match.group(1))
                morpheme = match.group(2)
                src_type = match.group(3)
                dst_type = match.group(4)

                total_corrected += count
                transitions[f"{src_type} ---> {dst_type}"] += count
                morpheme_stats[morpheme] += count

    print(f"=== ОБЩАЯ СТАТИСТИКА РЕМАРКОВ ===")
    print(f"Всего исправлено морфем: {total_corrected}")
    print(f"\n--- Топ переходов (по типам) ---")
    for trans, cnt in transitions.most_common():
        print(f" {trans:20} | {cnt} шт.")

    print(f"\n--- Топ измененных морфем ---")
    for morph, cnt in morpheme_stats.most_common(10):
        print(f" '{morph}': {cnt} раз")

    return total_corrected, transitions, morpheme_stats

get_detailed_stats()
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

def get_upos_tag(p_variant):
    oc_pos = p_variant.tag.POS
    upos_name = OPENCORPORA_TO_UPOS["POS"].get(oc_pos)
    return UPOS["POS"].get(upos_name, UNK_ID)

def is_initial(variant):
    t = variant.tag
    if t.POS == 'NOUN':
        return t.case == 'nomn' and t.number == 'sing'
    if t.POS in ('ADJF', 'PRTF'):
        return t.case == 'nomn' and t.number == 'sing' and t.gender == 'masc'
    if t.POS == 'INFN':
        return True
    if t.POS in ('ADVB', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'PRED'):
        return True
    return False

def process_morphology(word, morph_analyzer):
    parsed = morph_analyzer.parse(word)

    initial_candidates = [p for p in parsed if is_initial(p)]

    final_list = initial_candidates if initial_candidates else parsed

    pos_scores = defaultdict(float)
    for p in final_list:
        pos_id = get_upos_tag(p)
        pos_scores[pos_id] += p.score

    sorted_pos = sorted(pos_scores.items(), key=lambda x: x[1], reverse=True)
    best_pos_id, best_score = sorted_pos[0]

    if len(sorted_pos) == 1:
        return False, {"pos": best_pos_id, "validated": True}, 1

    second_pos_id, second_score = sorted_pos[1]

    if (best_score - second_score) > 0.02:
        return False, {"pos": best_pos_id, "validated": True}, len(sorted_pos)

    return True, {"pos_variants": sorted_pos, "validated": False}, len(sorted_pos)


def count_ambigious(path="./data/dictionaries/static_dictionary/tikhonov-morphemes-pos", user_val=True):
    from pymorphy3 import MorphAnalyzer
    m = MorphAnalyzer(lang='ru', result_type=None)

    morphemes = load_json(f"{path}.json")

    counter = 0
    very_amb = 0
    allc = 0
    processed_this_session = 0

    try:
        for word, word_data in morphemes.items():
            if word_data.get("validated") is True:
                allc += 1
                continue
                
            allc += 1
            is_ambiguous, result_data, parsed_len = process_morphology(word, m)

            if is_ambiguous:
                print("-" * 30)
                morphs = "|".join(f"{morph}:{rev_morphemes_vocab[morph_type]}" 
                                 for morph, morph_type in word_data["morphemes"])
                
                print(f"Слово [{allc}]: \033[1;32m{word}\033[0m ({morphs})")
                print(f"Варианты (ID, Score): {result_data['pos_variants']}")

                if user_val:
                    while True:
                        user_review = input("Введите ID (или 's' - пропустить, 'q' - сохранить и выйти): ").strip().lower()

                        if user_review == 'q':
                            raise KeyboardInterrupt
                        if user_review == 's':
                            print("Пропущено")
                            break
                        if user_review.isdigit():
                            word_data["pos"] = int(user_review)
                            word_data["validated"] = True
                            very_amb += 1
                            break
                        else:
                            print("\033[91mОшибка: введите числовой ID!\033[0m")

                    processed_this_session += 1
                else:
                    word_data["pos"] = result_data["pos_variants"][0]
                    word_data["validated"] = True
            else:
                if parsed_len > 1:
                    counter += 1
                word_data.update(result_data)
                word_data["validated"] = True

            if processed_this_session > 0 and processed_this_session % 50 == 0:
                to_json(morphemes, f"{path}-pos_autosave.json")
                print(f"\n[Автосохранение выполнено на слове {word}]\n")
                processed_this_session += 1

    except KeyboardInterrupt:
        print("\n\nПроцесс прерван пользователем")
    except Exception as e:
        print(f"\n\nПроизошла ошибка: {e}")
    finally:
        to_json(morphemes, f"{path}-pos.json")
        print(f"Данные сохранены в {path}-pos.json")
        print(f"Всего слов: {allc}, Валидировано вручную: {very_amb}")


def find_morpheme_type_collisions(rt, tt, morphemes):
    collision_sets = {}
    for word, word_data in morphemes.items():
        for (m, t) in word_data["morphemes"]:
            key = rev_morphemes_vocab[t]

            if key not in collision_sets:
                collision_sets[key] = {m}
            else:
                collision_sets[key].add(m)
    
    ignore_set = set()
    # if rt == 3 and tt ==  4:
    #     ignore_set = set(["тор", "те", "к", "лин", "ку", 'смен', 'ом', 'ин', 'ком', 'уч', 'м'])
    # if rt == 4 and tt == 3:
    #     ignore_set = set(['т','н','а'])

    collisions_list = list( collision_sets[rev_morphemes_vocab[rt]] & collision_sets[rev_morphemes_vocab[tt]] - ignore_set)
    print(f"Collisions count:{len(collisions_list)}")
    return collisions_list

def is_single_root(morphemes_list):
    root_count = 0
    for m, t in morphemes_list:
        if t == 3:
            root_count += 1 
    return root_count == 1

PREF, ROOT, SUFF, END, POSTFIX, LINK, HYPH = 2, 3, 4, 5, 6, 7, 8
def fix_regular_fast(morphemes_dict, regmorph, regmorph_type, pos, true_type, 
                     interfix_root_rule=False, suff_suff_rule=False):

    corrected_count = 0
    corrected_words = []
    
    for i, (word, word_data) in enumerate(morphemes_dict.items()):
        if word_data["pos"] not in pos:
            continue
            
        word_was_changed = False
        morphemes_list = word_data["morphemes"]
        single_root = is_single_root(morphemes_list)
        
        if regmorph_type == ROOT and single_root:
            continue

        prev_m, prev_t = "", -1

        for i, (m, t) in enumerate(morphemes_list):
            if i + 1 < len(morphemes_list):
                next_m, next_t = morphemes_list[i + 1]
            else:
                next_m, next_t = "", -1 
            if i + 2 < len(morphemes_list):
                next2_m, next2_t = morphemes_list[i + 2]
            else:
                next2_m, next2_t = "", -1

            
            is_target = (m == regmorph and t == regmorph_type)

            if not is_target:
                prev_m, prev_t = m, t
                continue

            is_hyphen_root = (prev_t == HYPH and t == ROOT and true_type != PREF)

            is_start_or_pref = (t == ROOT and true_type == SUFF and (i == 0 or prev_t == PREF))

            is_after_link = (t == ROOT and true_type == SUFF and 
                             ((prev_t == SUFF and prev_m in "оеаи") or prev_t == LINK))

            is_root_protection = (single_root and t == ROOT and regmorph_type != ROOT)
            
            
            is_valid_pos_for_interfix = (
                t in [SUFF, PREF, LINK] and 
                true_type == LINK and 
                prev_t in [LINK] and 
                next_t in [ROOT, PREF] and
                # prev_m in ["дв", "тр", ] and
                # next_m in ["дцат", "дцать"] and
                i != len(morphemes_list) - 1 # and m in ['о', 'и', 'а', 'е', 'на']
            )

            is_hyphen_root = (prev_t == HYPH and t == ROOT and true_type != PREF)

            is_start_or_pref = (t == ROOT and true_type == SUFF and i == 0)

            is_root_protection = (single_root and t == ROOT and true_type != ROOT)

            is_valid_pos_for_pref = (t in [ROOT, LINK, SUFF, PREF] and true_type == PREF and next_t in [ROOT, PREF])
            
            is_root_pref = ( t == PREF and true_type == ROOT and next_t == ROOT and prev_t == PREF)

            is_root_suff = ( t == SUFF and true_type == ROOT and i > 0 )

            is_suff_pref = ( t in [PREF, END, ROOT] and true_type == SUFF and prev_t != PREF and m in ["дцат", "дцать"])
            
            is_end_suff = ( t == SUFF and true_type == END and not next_t)

            fix_pref = ( next_m.startswith(("я", "е", "ю", "ё")) )

            should_fix = (
                not is_hyphen_root and           
                not is_start_or_pref and                
                not is_after_link and
                not is_root_protection and
                (is_valid_pos_for_pref if true_type == PREF else True) and
                (is_valid_pos_for_interfix if true_type == LINK else True) and
                (is_root_pref if true_type == ROOT and regmorph_type == PREF else True) and
                (is_root_suff if true_type == ROOT and regmorph_type == SUFF else True) and
                (is_suff_pref if true_type == SUFF else True) and
                (is_end_suff if true_type == END else True) and
                (fix_pref if true_type == PREF and regmorph_type == PREF else True)
                )

            if should_fix:
                morphemes_list[i] = (m, true_type)
                corrected_count += 1
                word_was_changed = True

                if prev_t == SUFF and true_type == ROOT and regmorph_type == SUFF and interfix_root_rule:
                    morphemes_list[i - 1] = (prev_m, LINK)
                if prev_t == LINK and true_type == SUFF and regmorph_type == ROOT and suff_suff_rule:
                    morphemes_list[i - 1] = (prev_m, SUFF)

            prev_m, prev_t = morphemes_list[i]
                
        if word_was_changed:
            corrected_words.append((word, word_data))
            
    return corrected_words, corrected_count, morphemes_dict

# 3,4
# 4,3
# 3,2
def find_n_fix(path="./data/dictionaries/static_dictionary/tikhonov-morphemes-pos", morph_types=[(2, 3)]):
    full_data = load_json(f"{path}.json")
    skipped = []
    pos_list = [i for i in range(0, 20)]

    for rt, tt in morph_types:
        collisions = find_morpheme_type_collisions(2, 2, full_data) # ("надо",) # 
        print(collisions)
        for c in collisions:
            temp_data = copy.deepcopy(full_data)
            
            corrected, count, modified_data = fix_regular_fast(
                temp_data, regmorph=c, regmorph_type=rt, pos=pos_list, true_type=tt
            )

            if count > 0:
                print(f"\nPay attention to {rev_morphemes_vocab[rt]}: '{c}' ({count} was found)")
                for word, data in corrected[:40]:
                    print(f"  {word}: {data['morphemes']} {data['pos']}")

                while True:
                    user_review = input(f"Add correction {rev_morphemes_vocab[rt]} -> {rev_morphemes_vocab[tt]} to '{c}'? (y/n): ").lower().strip()
                    if user_review == 'y':
                        full_data = modified_data
                        to_json(full_data, f"{path}.json")
                        log_to_remarks(c, rt, tt, count, corrected)
                        print("Saved.")
                        break
                    elif user_review == 'n':
                        skipped.append((c, rt))
                        print("Skipped.")
                        break

    to_json(full_data, f"{path}.json")
    print("Done.")
    skipped_str = "\n".join([f"{c}: {rev_morphemes_vocab[t]}" for c, t in skipped])
    print(f"Skipped List:\n{skipped_str}")