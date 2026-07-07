# Slovorez

English · [Русский](README.ru.md)

**Slovorez** is a neural library for morpheme segmentation of the Russian language: it splits a word into prefix, root, suffix and ending. Inference runs on CPU via ONNX Runtime — no GPU and no heavy dependencies — while the lexer core is written in C++. A lightweight model (~0.69M parameters) makes it suitable for production and edge use.

## Features

- Morpheme segmentation into classes: prefix, root, suffix, ending, postfix
- CPU inference via ONNX, no GPU required
- C++ tokenizer with pybind11 bindings
- Batch file processing with caching of already-segmented words

## Quick start

```python
from slovorez import Slovorez

model = Slovorez.from_pretrained("models/slovorez-test")
model.predict("Приставки и суффиксы выделяются автоматически.")
```

```python
[{'word': 'приставки',     'morphemes': [Morpheme('при', PREF, 0.61), Morpheme('став', ROOT, 0.78), Morpheme('к', SUFF, 0.71), Morpheme('и', END, 0.96)]},
 {'word': 'и',             'morphemes': [Morpheme('и', END, 0.54)]},
 {'word': 'суффиксы',      'morphemes': [Morpheme('суффикс', ROOT, 0.82), Morpheme('ы', END, 0.95)]},
 {'word': 'выделяются',    'morphemes': [Morpheme('вы', PREF, 0.93), Morpheme('дел', ROOT, 0.93), Morpheme('я', SUFF, 0.84), Morpheme('ют', SUFF, 0.79), Morpheme('ся', POSTFIX, 0.99)]},
 {'word': 'автоматически', 'morphemes': [Morpheme('автомат', ROOT, 0.73), Morpheme('ическ', SUFF, 0.80), Morpheme('и', SUFF, 0.76)]}]
```

Each morpheme is a `Morpheme(text, type, score)`.

Morpheme classes:

| class | meaning |
|---|---|
| PREF | prefix |
| ROOT | root |
| SUFF | suffix |
| END | ending (inflection) |
| POSTFIX | postfix |
| LINK | linking morpheme (interfix) |
| HYPH | hyphen |

You can pass text in any language — Slovorez extracts and segments only Russian words, preserving their order.

## Quality

Slovorez (Sq NoRoPE, ~0.69M parameters) on the Revised RuMorphsLemmas test set, under two splits:

- **Random split** — word forms split randomly; test roots may also occur in training.
- **Root split** — split by roots, so test roots are unseen in training (OOV) — the harder, generalization setting.

| Split | Boundary F1 | Root F1 | Accuracy | Word Acc |
|---|---|---|---|---|
| Random | 95.95 | 91.77 | 95.22 | 80.51 |
| Root (OOV) | 91.83 | 83.59 | 90.12 | 64.32 |

*Boundary F1 — F1 over all morpheme boundaries; Root F1 — F1 over root boundaries; Accuracy — character-level; Word Acc — share of fully correct words.*

## Installation

```bash
cd slovorez/
conda activate new_env

# CPU
pip install -e .[cpu]

# CUDA (GPU)
pip install -e .[gpu]
```

Demo:

```bash
python -m src.main
```

## How it works

The architecture is a Conv1D network with sequence labeling in the BIES scheme (morpheme boundaries as token labels). Tokenization and the lexer are implemented in C++; inference runs through ONNX Runtime on CPU.

## License

<!-- add license -->

## Citation

<!-- BibTeX once the paper is published -->
