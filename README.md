# Slovorez

English · [Русский](README.ru.md)

Slovorez does morpheme segmentation of Russian words. The model has 215K parameters and runs on CPU through ONNX Runtime. The tokenizer is written in C++.

The model was trained on lemmas. Support for inflected forms is planned for future versions, see "Limitations".

## Example

```python
from slovorez import Slovorez

model = Slovorez.from_pretrained()          # bundled model slovorez-flash
model.predict("перевыполнить, улыбаться")
```

```python
[{'word': 'перевыполнить',
  'morphemes': [['пере', 'PREF', 0.99], ['вы', 'PREF', 0.96], ['полн', 'ROOT', 0.99],
                ['и', 'SUFF', 1.0], ['ть', 'SUFF', 1.0]],
  'confidence': 0.95,
  'model': 'slovorez-flash'},
 {'word': 'улыбаться',
  'morphemes': [['улыб', 'ROOT', 0.86], ['а', 'SUFF', 0.99], ['ть', 'SUFF', 1.0],
                ['ся', 'POSTFIX', 1.0]],
  'confidence': 0.77,
  'model': 'slovorez-flash'}]
```

- `morphemes`: a list of `[text, type, score]`.
- `confidence`: confidence in the parse of the whole word, from 0 to 1.
- Words are lowercased and kept in text order, duplicates included. Only Cyrillic is parsed.

Morpheme types:

| type | meaning |
|---|---|
| PREF | prefix |
| ROOT | root |
| SUFF | suffix |
| END | ending (inflection) |
| POSTFIX | postfix |
| LINK | linking vowel (interfix) |
| HYPH | hyphen |

The Viterbi decoder only allows valid transitions between morphemes.

## Large files

```python
model.process_file("text.txt")                     # single thread
model.process_file("text.txt", threaded_mode=True) # inference in several threads
```

Results are appended to a jsonl file, one record per line, in the same format as `predict`. By default the file is `~/.cache/slovorez/predictions/<model>/<version>/predictions.jsonl`. `SLOVOREZ_HOME` changes the folder; `from_pretrained(predictions_path=...)` sets the file.

Words already in the file from earlier runs are skipped, so an interrupted run can be started again. Words are deduplicated before segmentation.

Words from the model's bundled dictionary (`base_dict.json`, ≈96K lemmas) are not segmented by `process_file` and are not written to the jsonl: the dictionary already has their parses.

## Installation

0.2.0 is on GitHub only for now:

```bash
pip install git+https://github.com/heysonanderson/slovorez
```

The tokenizer is compiled during installation, so you need a C++ compiler (MSVC Build Tools on Windows). Python 3.10+.

## Quality

slovorez-flash, Viterbi decoder. Metrics from `measure_quality` of Morozov et al. (2025), gold standard: `base_dict` 0.2.0.

| Test | Boundary F1 | Root F1 | Accuracy | Word Acc |
|---|---|---|---|---|
| Known roots (5,000 held-out words) | 95.69 | 87.15 | 92.53 | 75.50 |
| New roots (10 folds by root group) | 94.31 ± 0.43 | 81.95 ± 1.45 | 89.61 ± 0.82 | 69.42 ± 1.86 |

Boundary F1: over all morpheme boundaries. Root F1: over root boundaries. Accuracy: per character. Word Acc: share of words parsed fully correctly. In the second row the test roots were never seen in training; values are mean ± standard deviation over folds.

0.1.0 used a different model and a different test set.

## Speed

Benchmarks are being redone. The 0.1.0 numbers were removed: the method wasn't recorded and they don't reproduce.

## Limitations

Tested with the suite in `tests/` and on text corpora on one machine (Windows). Not tested under production load or on low-end devices.

- The model was trained on lemmas. Inflected forms come out worse, e.g. `выделяются` → `вы|дел|я|ютс|я`.
- Words longer than 64 characters are skipped without a warning.
- Tokenizer (C++): a chain of 9 or more hyphenated parts, or a word longer than 128 characters, loses all the text after it.
- A hyphenated word at the very end of a file, with no newline after it, can be lost. `predict` works around this.
- If `process_file` raises, create the object again with `from_pretrained`. Words written before the error are kept.
- The order of records in the jsonl depends on `PYTHONHASHSEED`.

## How it works

For each letter the model predicts a BIES tag (beginning, inside, end of a morpheme, or a single-letter morpheme) together with the morpheme type. Network: letter embeddings, three Conv1D layers (kernel 5), a head of 1×1 convolutions. The C++ tokenizer is bound through pybind11; inference runs through ONNX Runtime.

## License

MIT. Copyright (c) 2026 heysonanderson, pubari. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

```bibtex
@software{slovorez,
  author  = {heysonanderson and pubari},
  title   = {Slovorez: neural morpheme segmentation for Russian},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/heysonanderson/slovorez},
  license = {MIT}
}
```
