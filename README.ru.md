# Slovorez

[English](README.md) · **Русский**

**Slovorez** — нейросетевая библиотека для морфемной сегментации русского языка: разбивает слово на приставку, корень, суффикс и окончание (тот самый «разбор слова по составу»). Инференс идёт на CPU через ONNX Runtime — без GPU и тяжёлых зависимостей, — а ядро лексера написано на C++. Лёгкая модель (~0.69M параметров) делает библиотеку пригодной для продакшена и edge-сценариев.

## Возможности

- Морфемный разбор на классы: приставка, корень, суффикс, окончание, постфикс
- Инференс на CPU через ONNX, без GPU
- C++ токенизатор с привязками через pybind11
- Пакетная обработка файлов с кэшированием уже размеченных слов

## Быстрый старт

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

Каждая морфема — это `Morpheme(текст, класс, уверенность)`..

Классы морфем:

| класс | значение |
|---|---|
| PREF | приставка |
| ROOT | корень |
| SUFF | суффикс |
| END | окончание |
| POSTFIX | постфикс |
| LINK | соединительная морфема (интерфикс) |
| HYPH | дефис |

В текст можно подавать любой язык — Slovorez извлекает и размечает только русские слова, сохраняя их порядок.

## Качество

Slovorez (Sq NoRoPE, ~0.69M параметров) на тестовой выборке Revised RuMorphsLemmas, в двух режимах разбиения:

- **Random split** — словоформы разбиваются случайно; корни из теста могут встречаться и в обучении.
- **Root split** — разбиение по корням, корни теста не встречаются при обучении (OOV) — более сложный режим на обобщение.

| Сплит | F1 границ | Root F1 | Accuracy | Word Acc |
|---|---|---|---|---|
| Random | 95.95 | 91.77 | 95.22 | 80.51 |
| Root (OOV) | 91.83 | 83.59 | 90.12 | 64.32 |

*F1 границ — F1 по всем границам морфем; Root F1 — F1 по границам корней; Accuracy — посимвольная; Word Acc — доля полностью верно разобранных слов.*

## Установка

### Требования

- CMake 3.15+
- Компилятор с поддержкой C++11
- Python 3.8+ с заголовочными файлами для разработки

### Сборка из исходников

Установи зависимости для сборки.

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

**macOS** (требуется Homebrew)

```bash
brew install cmake python@3.12
```

**Windows**

1. **Компилятор C++** — Visual Studio с воркфлоу *Desktop development with C++* или MinGW-w64
2. **CMake** — с cmake.org
3. **Python** — с python.org. При установке отметь **«Add Python to PATH»** и убедись, что включены **development headers**.

Сборка:

```bash
# Перейти в папку проекта
cd slovorez/

# Создать каталог сборки
mkdir build

# Конфигурация и сборка
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

В корне проекта появится файл общей библиотеки.

### Установка Python-пакета

```bash
cd slovorez/
conda activate new_env

# CPU
pip install .[cpu]

# CUDA (GPU)
pip install .[gpu]
```

Демо:

```bash
python -m src.main
```

## Как это устроено

Архитектура — Conv1D с разметкой последовательностей в схеме BIES (границы морфем как метки токенов). Лексер и токенизация реализованы на C++, инференс — через ONNX Runtime на CPU.

## Лицензия

<!-- укажи лицензию -->

## Цитирование

<!-- BibTeX появится после публикации статьи -->
