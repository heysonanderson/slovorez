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

### Decoding

By default, predictions are decoded with greedy argmax. Pass `use_viterbi=True` for constrained BIES decoding, which enforces valid label transitions and improves boundary consistency:

```python
model.predict("...", use_viterbi=True)
```

## Quality

Slovorez (Sq NoRoPE, ~0.69M parameters) on the Revised RuMorphsLemmas test set, under two splits:

- **Random split** — word forms split randomly; test roots may also occur in training.
- **Root split** — split by roots, so test roots are unseen in training (OOV) — the harder, generalization setting.

| Split | Boundary F1 | Root F1 | Accuracy | Word Acc |
|---|---|---|---|---|
| Random | 95.95 | 91.77 | 95.22 | 80.51 |
| Root (OOV) | 91.83 | 83.59 | 90.12 | 64.32 |

*Boundary F1 — F1 over all morpheme boundaries; Root F1 — F1 over root boundaries; Accuracy — character-level; Word Acc — share of fully correct words.*

## Performance

Benchmarks on a corpus of Russian classic literature: ~3.6M tokens total, of which ~1.71M are Russian words (~150K unique forms); the rest are punctuation and non-Russian tokens, which the tokenizer skips. *Pure inference* runs the full pipeline (C++ tokenizer → dedup → ONNX inference → JSONL writer) over the ~150K unique forms; *real corpus* runs the whole corpus with the `SeenIndex` cache, which — by Zipf's law — turns repeats into cache hits (~91% hit rate here).

**Pure inference — ~150K unique word forms (words/s):**

| Configuration | 1 worker | 4 workers |
|---|---|---|
| CPU — Ryzen 7 5700X | ~13,200 | ~17,600 |
| GPU — RTX 5060 Ti | ~19,500 | ~23,000 |

**Real corpus — ~1.71M Russian words, sequential with cache (words/s):**

| Configuration | Decoder | Throughput |
|---|---|---|
| CPU — Ryzen 7 5700X | argmax | ~141,000 |
| CPU — Ryzen 7 5700X | Viterbi | ~67,000 |
| GPU — RTX 5060 Ti (batch 2048) | argmax | ~246,000 |

## Installation

```bash
cd slovorez/
conda activate <your_env>

# CPU (recommended)
pip install .[cpu]
```

Demo:

```bash
python -m src.main
```

### GPU (optional, for advanced users)

CPU is the default, recommended.

If you specifically want CUDA execution:

```bash
pip install .[gpu]
```

GPU support is provided **as-is** and is intended for users who manage their own CUDA/cuDNN stack. `onnxruntime-gpu` does **not** bundle CUDA or cuDNN — you are responsible for providing compatible runtime libraries and putting them on the library search path. Start here:

- ONNX Runtime CUDA execution provider: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ONNX Runtime install / version matrix: <https://onnxruntime.ai/docs/install/>

If your CUDA/cuDNN stack doesn't match ONNX Runtime's requirements, ORT falls back to CPU. The simplest way to avoid manual DLL setup is to install a PyTorch build compiled against the same major CUDA version as ONNX Runtime — ORT then reuses PyTorch's CUDA/cuDNN libraries automatically.

## How it works

The architecture is a Conv1D network with sequence labeling in the BIES scheme (morpheme boundaries as token labels). Tokenization and the lexer are implemented in C++; inference runs through ONNX Runtime on CPU.

## License

<!-- add license -->

## Citation

<!-- BibTeX once the paper is published -->