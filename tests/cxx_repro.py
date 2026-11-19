"""Воспроизведения ошибок C++-токенизатора (slovorezCXX).

Номера C1..C22 — из ревью C++ от 27.10.2026. Каждый случай запускается
в отдельном процессе: так падение (segfault) не роняет весь прогон,
а мусор из памяти от запуска к запуску разный. Нестабильные случаи
гоняются по нескольку раз.

    python tests/cxx_repro.py            # все случаи
    python tests/cxx_repro.py C4 C16     # только эти
    python tests/cxx_repro.py --md out.md  # плюс отчёт в Markdown

Нужен собранный модуль slovorezCXX (pip install -e .).
Код возврата: 0 — всё совпало с ожиданием, 1 — есть расхождения.

Значки по ходу прогона, по одному на запуск:
    ✅ результат как ожидали     ❌ результат неправильный
    💥 процесс упал (segfault)    ⚠️ ошибка в коде самой проверки (Python)
    🔹 решение по дизайну, просто показываем, что сейчас
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

PRELUDE = r'''
import os, sys, tempfile
from slovorezCXX import FTTokenizer, FFTokenizer, TokenType

def _setup(tk, bs, ru):
    if ru:
        tk.set_filter(TokenType.RUWORD)
    if bs is not None:
        tk.set_batch_size(bs)
    return tk

def toks(tk):
    out = []
    while (b := tk.get_batch()):
        out += b["text"].split("\0")[:-1]
    return out

def toks_t(tk):
    out = []
    while (b := tk.get_batch()):
        out += list(zip(b["text"].split("\0")[:-1], map(int, b["types"])))
    return out

def ft(text, bs=None, ru=True, types=False):
    tk = _setup(FTTokenizer(text), bs, ru)
    return toks_t(tk) if types else toks(tk)

def _write(data, name):
    d = tempfile.mkdtemp()
    p = os.path.join(d, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data.encode("utf-8") if isinstance(data, str) else data)
    return p

def ff(data, bs=None, ru=True, name="t.txt"):
    tk = FFTokenizer(_write(data, name))
    if not tk.is_fopen():
        return "is_fopen() == False"
    return toks(_setup(tk, bs, ru))

def show(x):
    print("@@RESULT@@" + repr(x))
'''


@dataclass
class Case:
    cid: str          # номер находки
    title: str
    code: str         # должен вызвать show(...)
    expect: str       # что должно быть, словами или repr
    ok: Callable[[Any], bool]  # результат (Python-объект) -> совпал ли с ожиданием
    runs: int = 1
    design: bool = False  # решение по дизайну, не баг: в итог не идёт
    results: Counter = field(default_factory=Counter)
    passed: int = 0
    crashes: int = 0      # процесс упал
    pyerrors: int = 0     # упал код проверки, а не C++


# Где искать (строки на коммите 6710f4d; C++ с ревью 27.10 не менялся).
WHERE = {
    "C1": "token.h:38 + lexer.cc:5 (data[128] — 128 символов UTF8Char, не байт); tokenizer.h:11 + tokenizer.cc:5 (tokens[16] — дефисы тоже части); склейка tokenizer.cc:20 пишет в rtoken.data[128]",
    "C2": "pybinder.cc:272-280, :328, :357 — stream держит ссылку, нужен py::keep_alive",
    "C3": "utf8_decoder.h:87-92 (curr_size без предела); unicode_trie.h:22 (lookup без проверки codepoint < 0x110000)",
    "C4": "pybinder.cc:100 — `<=` вместо `<` (в FF, :221, стоит `<`)",
    "C16": "tokenizer.cc:109-113 + :30-34 (пустой контекст) и :63-73 (склейка); pybinder.cc:107-114 (FT), :229-236 (FF) — второй end только при true",
    "C17": "pybinder.cc:61-64, :71-72, :178-179, :186-187 — размер 0, malloc/realloc без проверки",
    "C5": "pybinder.cc:95 (FF :216) — ранний return мимо flush; условие flush :107 (FF :229)",
    "C6": "pybinder.cc:227 — последний fgetc съедает байт следующего батча, :216",
    "C7": "pybinder.cc:122-127, :244-249 — types отдаётся view на внутренний буфер",
    "C8": "pybinder.cc:177 — fopen(..., \"r\") вместо \"rb\"",
    "C9": "pybinder.cc:177 — fopen с UTF-8 путём на Windows (нужен _wfopen)",
    "C10": "tokenizer.cc:54 — дефис приклеивается и не откатывается",
    "C11": "unicode_trie.cc — U+0301, U+0306, U+0308, U+00AD не входят в слово (UNKNWN)",
    "C12": "tokenizer.cc:54 — склейка только по ASCII '-', а U+2010–U+201F размечены PNCTTN",
    "C13": "tokenizer.cc:10 — тип склейки берётся по первой части",
    "N1": "pybinder.cc:48-52 — NUL из текста попадает в text, где NUL — разделитель",
    "N2": "lexer.cc:27-34 — смена алфавита внутри слова рвёт токен",
    "N3": "unicode_trie.cc:182-258 — кириллица только U+0401, U+0410–U+044F, U+0451",
    "N4": "lexer.cc:27-34 — NUMBER и RUWORD разных типов",
    "N5": "tokenizer.cc:54 — склейка только по '-'",
    "N6": "pybinder.cc:80-88 — set_token_min_len/max_len без проверки",
    "C14": "unicode_trie.cc:12 — TAB/VT, похоже, перепутаны; \\r, U+2009, U+202F → UNKNWN",
}

LONG = "а" * 129
CHAIN = "-".join(["а"] * 20)

CASES = [
    # --- порча памяти и падения ---
    Case("C1", "слово длиннее 128 символов (Token.data[128])",
         f'show(ft("начало {LONG} конец"))',
         "'начало' и 'конец' на месте (длинное слово — как решите: целиком или выкинуть)",
         lambda r: isinstance(r, list) and r[:1] == ["начало"] and r[-1:] == ["конец"]),
    Case("C1", "цепочка из 20 частей через дефис (tokens[16])",
         f'show(ft("начало {CHAIN} конец"))',
         "'начало' и 'конец' на месте",
         lambda r: isinstance(r, list) and r[:1] == ["начало"] and r[-1:] == ["конец"]),
    Case("C1", "ссылка со slug из 10 слов через дефис",
         'show(ft("начало https://example.com/kak-pravilno-i-bystro-varit-vkusnyi-borsch-po-receptu-babushki конец"))',
         "'начало' и 'конец' на месте",
         lambda r: isinstance(r, list) and "конец" in r),
    Case("C1", "склейка длиннее 128 символов из коротких частей (2 × 70 букв)",
         'show([(len(t), ty) for t, ty in ft("начало " + "а" * 70 + "-" + "а" * 70 + " конец", ru=False, types=True) if len(t) > 10])',
         "[(141, 16)] — слово целиком и тип RUWORD (или явный отказ)",
         lambda r: r in ([(141, 16)], [])),
    Case("C2", "итерация по .stream временного FTTokenizer",
         'n = sum(1 for _ in FTTokenizer("кот пёс").stream)\nshow(n)',
         "1 батч, процесс не падает",
         lambda r: r == 1, runs=3),
    Case("C2", "итерация по .stream временного FFTokenizer",
         'n = sum(1 for _ in FFTokenizer(_write("кот пёс", "t.txt")).stream)\nshow(n)',
         "1 батч (если держать токенизатор в переменной, так и есть)",
         lambda r: r == 1, runs=3),
    Case("C3", "файл: 30000 continuation-байтов 0x80 перед текстом",
         'show(ff(b"\\x80" * 30000 + "привет мир".encode()))',
         "['привет', 'мир'], процесс не падает (при 400 байтах пока не падает)",
         lambda r: r == ["привет", "мир"], runs=5),
    Case("C3", "файл: F4 90 80 80 (codepoint 0x110000, за пределом Unicode)",
         'show(ff(b"\\xf4\\x90\\x80\\x80" + " привет мир\\nещё строка\\n".encode()))',
         "['привет', 'мир', 'ещё', 'строка'], процесс не падает",
         lambda r: r == ["привет", "мир", "ещё", "строка"], runs=5),
    Case("C4", "чтение за концом текста (<= в get_batch)",
         'show(ft("Привет, мир!", ru=False))',
         "['Привет', ',', ' ', 'мир', '!'] без лишнего токена в конце",
         lambda r: r == ["Привет", ",", " ", "мир", "!"], runs=20),
    Case("C4", "размер батча: set_batch_size(4)",
         'tk = FTTokenizer("а б в г д е ж з и к")\ntk.set_filter(TokenType.RUWORD)\ntk.set_batch_size(4)\n'
         'sizes = []\nwhile (b := tk.get_batch()):\n    sizes.append(len(b["text"].split("\\0")) - 1)\nshow(sizes)',
         "[4, 4, 2]",
         lambda r: r == [4, 4, 2], runs=3),
    Case("C16", "строка кончается на слово через дефис: кот-cat",
         'show(ft("кот-cat"))',
         "['кот-cat']",
         lambda r: r == ["кот-cat"], runs=10),
    Case("C16", "строка кончается на слово через дефис: кто-то",
         'show(ft("я видел кто-то"))',
         "['я', 'видел', 'кто-то']",
         lambda r: r == ["я", "видел", "кто-то"], runs=20),
    Case("C16", "строка кончается на слово через дефис: ту-сто",
         'show(ft("летим на ту-сто"))',
         "['летим', 'на', 'ту-сто']",
         lambda r: r == ["летим", "на", "ту-сто"], runs=20),
    Case("C16", "файл кончается на слово через дефис: из-за",
         'show(ff("пошёл из-за"))',
         "['пошёл', 'из-за']",
         lambda r: r == ["пошёл", "из-за"], runs=10),
    Case("C16", "файл кончается на слово через дефис: бело-сине-красный",
         'show(ff("флаг бело-сине-красный"))',
         "['флаг', 'бело-сине-красный']",
         lambda r: r == ["флаг", "бело-сине-красный"], runs=10),
    Case("C16", "файл из одного слова без перевода строки",
         'show(ff("кот"))',
         "['кот']",
         lambda r: r == ["кот"], runs=5),
    Case("C16", "строка из одного слова: ёлка",
         'show(ft("ёлка"))',
         "['ёлка']",
         lambda r: r == ["ёлка"], runs=30),
    Case("C16", "то же, но с переводом строки в конце (контроль)",
         'show(ff("пошёл из-за\\n"))',
         "['пошёл', 'из-за']",
         lambda r: r == ["пошёл", "из-за"], runs=5),
    Case("C17", "FFTokenizer: set_batch_size(0)",
         'try:\n    show(ff("кот пёс", bs=0))\nexcept Exception as e:\n    show(type(e).__name__)',
         "ошибка (ValueError) или ['кот', 'пёс'], но не молча []",
         lambda r: r in ("ValueError", ["кот", "пёс"])),
    Case("C17", "FTTokenizer: set_batch_size(0) (запись в malloc(0))",
         'try:\n    show(ft("кот пёс", bs=0))\nexcept Exception as e:\n    show(type(e).__name__)',
         "ошибка (ValueError) или ['кот', 'пёс']",
         lambda r: r in ("ValueError", ["кот", "пёс"]), runs=5),
    # --- тихая потеря данных ---
    Case("C5", "число токенов кратно батчу: bs=1",
         'show(ft("а б в", bs=1))',
         "['а', 'б', 'в']",
         lambda r: r == ["а", "б", "в"]),
    Case("C5", "число токенов кратно батчу: bs=2",
         'show(ft("а б в", bs=2))',
         "['а', 'б', 'в']",
         lambda r: r == ["а", "б", "в"]),
    Case("C6", "FFTokenizer теряет байт на границе батча",
         'show(ff(" ".join(["абвгд"] * 6), bs=2))',
         "['абвгд'] * 6",
         lambda r: r == ["абвгд"] * 6),
    Case("C7", "types первого батча после второго get_batch",
         'tk = FTTokenizer("а б 1 2 3 4")\ntk.set_batch_size(3)\nb1 = tk.get_batch()\n'
         'before = list(map(int, b1["types"]))\ntk.get_batch()\nafter = list(map(int, b1["types"]))\n'
         'show((before, after))',
         "типы не меняются (в батче 4 типа вместо 3 — это C4)",
         lambda r: r[0] == r[1]),
    Case("C8", "Ctrl-Z (0x1A) в файле",
         'show(ff("начало \\x1a конец"))',
         "['начало', 'конец']",
         lambda r: r == ["начало", "конец"]),
    Case("C9", "кириллица в пути к файлу",
         'show(ff("кот пёс", name="папка/текст.txt"))',
         "['кот', 'пёс']",
         lambda r: r == ["кот", "пёс"]),
    # --- лингвистика ---
    Case("C10", "висячий дефис перед пробелом",
         'show(ft("слово- дальше"))',
         "['слово', 'дальше']",
         lambda r: r == ["слово", "дальше"]),
    Case("C10", "двойной дефис",
         'show(ft("а--б"))',
         "['а', 'б']",
         lambda r: r == ["а", "б"]),
    Case("C10", "дефис в самом конце текста",
         'show(ft("слово-"))',
         "['слово']",
         lambda r: r == ["слово"], runs=10),
    Case("C11", "ударение U+0301 внутри слова",
         'show(ft("замо\\u0301к дальше"))',
         "['замо\\u0301к', 'дальше'] или ['замок', 'дальше']",
         lambda r: r in (["замо́к", "дальше"], ["замок", "дальше"])),
    Case("C11", "мягкий перенос U+00AD внутри слова",
         'show(ft("при\\u00adвет дальше"))',
         "['при\\xadвет', 'дальше'] или ['привет', 'дальше']",
         lambda r: r in (["при­вет", "дальше"], ["привет", "дальше"])),
    Case("C11", "ё в разложенной форме (е + U+0308, NFD)",
         'show(ft("е\\u0308лка дальше"))',
         "['ёлка', 'дальше'] (или исходная форма), не ['е', 'лка', ...]",
         lambda r: r in (["ёлка", "дальше"], ["ёлка", "дальше"])),
    Case("C11", "й в разложенной форме (и + U+0306, NFD): мой",
         'show(ft("мои\\u0306 дальше"))',
         "['мой', 'дальше'] (или исходная форма), не ['мои', ...]",
         lambda r: r in (["мой", "дальше"], ["мой", "дальше"])),
    Case("C12", "неразрывный дефис U+2011",
         'show(ft("кто\\u2011то дальше"))',
         "['кто‑то', 'дальше'] или ['кто-то', 'дальше']",
         lambda r: r in (["кто‑то", "дальше"], ["кто-то", "дальше"])),
    Case("C12", "дефис U+2010",
         'show(ft("кто\\u2010то дальше"))',
         "['кто‐то', 'дальше'] или ['кто-то', 'дальше']",
         lambda r: r in (["кто‐то", "дальше"], ["кто-то", "дальше"])),
    Case("C10", "перенос слова через строку: пере-\\nнос",
         'show(ft("пере-\\nнос дальше"))',
         "['перенос', 'дальше'] или ['пере', 'нос', 'дальше'], но не 'пере-'",
         lambda r: isinstance(r, list) and "пере-" not in r),
    Case("C14", "TAB между словами (тип токена)",
         'show(ft("кот\\tпёс", ru=False, types=True))',
         "TAB — пробел (WRDSPC = 1): [('кот', 16), ('\\t', 1), ('пёс', 16)]",
         lambda r: dict(r).get("\t") == 1),
    Case("C14", "\\r\\n между словами (тип токенов)",
         'show(ft("кот\\r\\nпёс", ru=False, types=True))',
         "\\r — не UNKNWN (64)",
         lambda r: dict(r).get("\r") != 64),
    # --- новое, 19.11 ---
    Case("N1", "NUL в тексте без фильтра: число токенов в text и types",
         'tk = FTTokenizer("кот\\0пёс дом\\n")\nb = tk.get_batch()\nshow((len(b["text"].split("\\0")) - 1, len(b["types"])))',
         "одинаковое число",
         lambda r: r[0] == r[1]),
    Case("N2", "латинская o внутри русского слова: кoт",
         'show(ft("кoт дом\\n"))',
         "['кoт', 'дом'] или ['дом'], но не обрывки 'к', 'т'",
         lambda r: r in (["кoт", "дом"], ["дом"])),
    Case("N3", "буквы вне U+0410–U+044F: Київ ўсё міръ",
         'show(ft("Київ ўсё міръ\\n"))',
         "слова целиком или выкинуты целиком, без обрывков 'Ки', 'в'",
         lambda r: isinstance(r, list) and all(w in ("Київ", "ўсё", "міръ") for w in r)),
    Case("N4", "число вплотную к букве: 2024г 5кг в2024",
         'show(ft("2024г 5кг в2024\\n"))',
         "решение по дизайну: сейчас 'г', 'кг', 'в' уходят в модель отдельными словами",
         lambda r: True, design=True),
    Case("N5", "апостроф: д'Артаньян О’Генри",
         'show(ft("д\'Артаньян О\\u2019Генри\\n"))',
         "решение по дизайну: сейчас рвётся на части",
         lambda r: True, design=True),
    Case("N6", "min_len=10 > max_len=2",
         'try:\n    tk = FTTokenizer("кот пёс\\n")\n    tk.set_token_min_len(10)\n    tk.set_token_max_len(2)\n    show(toks(tk))\nexcept Exception as e:\n    show(type(e).__name__)',
         "ошибка (ValueError), а не молча пустой результат",
         lambda r: r == "ValueError"),
    Case("C13", "тип склейки по первой части: 5-й",
         'show(ft("5-й", ru=False, types=True))',
         "решение по дизайну: сейчас NUMBER (8), с фильтром RUWORD слово выпадает",
         lambda r: True, design=True),
    Case("C13", "тип склейки по первой части: Ту-154",
         'show(ft("Ту-154 ", ru=False, types=True))',
         "решение по дизайну: сейчас RUWORD (16) с цифрами внутри",
         lambda r: True, design=True),
]


def run_once(case: Case) -> tuple[str, object]:
    """Один запуск в отдельном процессе -> ("ok", результат) | ("crash", текст) | ("pyerr", текст)."""
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", PRELUDE + "\n" + case.code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            raw = line[len("@@RESULT@@"):]
            try:
                return "ok", eval(raw)  # repr списков/строк/чисел из нашего же кода
            except Exception:
                return "ok", raw
    tail = (proc.stderr.strip().splitlines() or [""])[-1]
    if "Traceback" in proc.stderr:  # исключение Python в коде проверки — не падение C++
        return "pyerr", f"<ошибка проверки: {tail}>"
    code = f"{proc.returncode & 0xFFFFFFFF:#010x}" if proc.returncode < 0 or proc.returncode > 255 else proc.returncode
    return "crash", f"<процесс упал: код {code}{'; ' + tail if tail else ''}>"


def fmt(x: object, limit: int = 120) -> str:
    s = repr(x)
    return s if len(s) <= limit else s[:limit] + "…"


def status_of(c: Case) -> str:
    n = c.runs
    if c.pyerrors:
        return f"⚠️ ОШИБКА ПРОВЕРКИ {c.pyerrors}/{n}"
    if c.design:
        return "🔹 дизайн"
    if c.passed == n:
        return "✅ ок"
    fails = n - c.passed - c.crashes
    parts = []
    if c.crashes:
        parts.append("💥 ПАДЕНИЕ" + ("" if c.crashes == n else f" {c.crashes}/{n}"))
    if fails:
        parts.append("❌ ОШИБКА" + ("" if fails == n else f" {fails}/{n}"))
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="номера находок, например C4 C16")
    ap.add_argument("--md", help="записать отчёт в Markdown")
    args = ap.parse_args()

    # эмодзи в консоль Windows и в перенаправленный вывод (cp1251 их не умеет)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    cases = [c for c in CASES if not args.ids or c.cid in args.ids]
    width = max(len(c.title) for c in cases) if cases else 0
    print(f"Прогон: {len(cases)} случаев, {sum(c.runs for c in cases)} запусков\n")
    for c in cases:
        print(f"  {c.cid:<4} {c.title:<{width}}  ", end="", flush=True)
        for _ in range(c.runs):
            kind, r = run_once(c)
            if kind == "crash":
                c.crashes += 1
                c.results[str(r)] += 1
                mark = "💥"
            elif kind == "pyerr":
                c.pyerrors += 1
                c.results[str(r)] += 1
                mark = "⚠️"
            else:
                c.results[fmt(r)] += 1
                ok = c.ok(r)
                c.passed += ok
                mark = "🔹" if c.design else ("✅" if ok else "❌")
            print(mark, end="", flush=True)
        print()
    print()

    lines = []
    for c in cases:
        status = status_of(c)
        lines.append((c, status))

        print(f"[{status}] {c.cid}: {c.title}")
        print(f"    где:      {WHERE.get(c.cid, '—')}")
        print(f"    ждём:     {c.expect}")
        for res, n in c.results.most_common():
            cnt = f" ×{n}" if c.runs > 1 else ""
            print(f"    получили: {res}{cnt}")
        print()

    real = [c for c in cases if not c.design]
    bad = [c for c in real if c.passed < c.runs or c.pyerrors]
    crashed = [c for c in cases if c.crashes]
    broken = [c for c in cases if c.pyerrors]
    print(f"Итого: расхождений {len(bad)} из {len(real)}"
          f"  (✅ {len(real) - len(bad)}  ❌ {sum(1 for c in bad if c.crashes < c.runs and not c.pyerrors)}"
          f"  💥 {len(crashed)}  🔹 {len(cases) - len(real)} дизайн)")
    if crashed:
        print("\n💥 Процесс падает:")
        for c in crashed:
            print(f"   💥 {c.cid}: {c.title} — {c.crashes} из {c.runs}")
    if broken:
        print("\n⚠️ Ошибка в коде проверки (чинить скрипт, а не C++):")
        for c in broken:
            print(f"   ⚠️ {c.cid}: {c.title}")

    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("| № | Случай | Ждём | Получили | Статус |\n|---|---|---|---|---|\n")
            for c, status in lines:
                got = "<br>".join(
                    f"`{r}`" + (f" ×{n}" if c.runs > 1 else "")
                    for r, n in c.results.most_common()
                ).replace("|", "\\|")
                f.write(f"| {c.cid} | {c.title} | {c.expect.replace('|', '/')} | {got} | {status} |\n")
    return 1 if bad or broken else 0


if __name__ == "__main__":
    sys.exit(main())
