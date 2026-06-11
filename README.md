# Slovorez

English · [Русский](README.ru.md)

> Neural morpheme segmentation for Russian

<!-- badges after publishing:
![PyPI](https://img.shields.io/pypi/v/slovorez)
![License](https://img.shields.io/badge/license-MIT-blue)
-->

**Slovorez** is a neural library for morpheme segmentation of the Russian language: it splits a word into prefix, root, suffix and ending. Inference runs on CPU via ONNX Runtime — no GPU and no heavy dependencies — while the lexer core is written in C++. A lightweight model (~0.69M parameters) makes it suitable for production and edge use.

## Features

- Morpheme segmentation into classes: prefix, root, suffix, ending, postfix
- CPU inference via ONNX, no GPU required
- C++ tokenizer with pybind11 bindings
- Batch file processing with caching of already-segmented words

## Quick start

```python
from slovorez import Slovorez

model = Slovorez.from_pretrained("models/slovorez")
model.predict("Приставки и суффиксы выделяются автоматически.")
```

```python
[{'word': 'приставки',     'morphemes': [Morpheme('при', PREF, 0.61), Morpheme('став', ROOT, 0.78), Morpheme('к', SUFF, 0.71), Morpheme('и', END, 0.96)]},
 {'word': 'и',             'morphemes': [Morpheme('и', END, 0.54)]},
 {'word': 'суффиксы',      'morphemes': [Morpheme('суффикс', ROOT, 0.82), Morpheme('ы', END, 0.95)]},
 {'word': 'выделяются',    'morphemes': [Morpheme('вы', PREF, 0.93), Morpheme('дел', ROOT, 0.93), Morpheme('я', SUFF, 0.84), Morpheme('ют', SUFF, 0.79), Morpheme('ся', POSTFIX, 0.99)]},
 {'word': 'автоматически', 'morphemes': [Morpheme('автомат', ROOT, 0.73), Morpheme('ическ', SUFF, 0.80), Morpheme('и', SUFF, 0.76)]}]
```

Each morpheme is a `Morpheme(text, type, score)`. For example, `приставки` → **при**-**став**-**к**-**и** (prefix · root · suffix · ending).

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

## Installation

### Prerequisites

- CMake 3.15+
- A compiler with C++11 support
- Python 3.8+ with development headers

### Build from source

Install the build dependencies.

**Ubuntu / Debian / Mint / Pop!\_OS**

```bash
sudo apt install g++ cmake python3-dev
```

**Fedora / Red Hat**

```bash
sudo dnf install gcc-c++ cmake python3-devel
```

**Arch / Manjaro / EndeavourOS**

```bash
sudo pacman -S base-devel cmake python
```

**macOS** (requires Homebrew)

```bash
brew install cmake python@3.12
```

**Windows**

1. **C++ compiler** — Visual Studio with the *Desktop development with C++* workload, or MinGW-w64
2. **CMake** — from cmake.org
3. **Python** — from python.org. During installation, check **"Add Python to PATH"** and make sure **development headers** are included.

Build:

```bash
# Go to the project folder
cd slovorez/

# Create the build directory
mkdir build

# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

This produces the shared library file in the project root.

### Python package

```bash
cd slovorez/
conda activate new_env

# CPU
pip install .[cpu]

# CUDA (GPU)
pip install .[gpu]
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
