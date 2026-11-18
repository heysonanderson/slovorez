# bench

Speed benchmarks for slovorez. Run from the repository root.

## Results 2026-11-18

The summary of these tables is in the [README](../README.md#speed).

**Environment.** AMD Ryzen 7 5700X (8 cores / 16 threads), 32 GB RAM, Windows 10
(19045). Commit `6c435ef` with uncommitted changes.

| | 0.2.0 | 0.1.0 |
|---|---|---|
| model | `slovorez-flash` | `slovorez-test` |
| decoder | Viterbi (from config) | Viterbi (`use_viterbi=True`) |
| Python | 3.10.20 | 3.10.22 |
| numpy / onnxruntime | 2.2.6 / 1.23.2 | 2.2.6 / 1.23.2 |
| word length | 1..64 | 4..64 |

Both versions use `batch_size` 65536 and `model_batch` 256. Cells show the median
and (min–max) over 3 runs.

### 100K sentences

18 MB, 1,376,577 tokens, 131,954 unique, sha256 `795b7253c998…`.

Tokens per second:

| | 0.2.0 | 0.1.0 |
|---|---|---|
| `process_file`, 1 thread | **94K** (93K–94K) | 64K (64K–64K) |
| `process_file`, 2 workers | **157K** (155K–157K) | 43K (42K–43K) |
| `process_file`, 4 workers | **208K** (208K–209K) | 37K (36K–37K) |
| `process_file`, 8 workers | **229K** (227K–232K) | 37K (37K–38K) |
| tokenizer + dedup only | 632K (630K–636K) | 628K (624K–638K) |

| | 0.2.0 | 0.1.0 |
|---|---|---|
| MB/s, 1 thread | **1.2** | 0.8 |
| forms sent to the model | 117,204 | 115,138 |
| forms/s, 1 thread | **8K** | 5K |
| peak memory, MB | **228** | 238 |

### 1M sentences

181 MB, 13,794,075 tokens, 411,482 unique, sha256 `3a61c20a9323…`.

Tokens per second:

| | 0.2.0 | 0.1.0 |
|---|---|---|
| `process_file`, 1 thread | **210K** (208K–210K) | 152K (151K–154K) |
| `process_file`, 2 workers | **316K** (315K–317K) | 108K (108K–109K) |
| `process_file`, 4 workers | **378K** (378K–379K) | 93K (92K–94K) |
| `process_file`, 8 workers | **396K** (389K–398K) | 94K (94K–94K) |
| tokenizer + dedup only | 629K (623K–631K) | 628K (623K–634K) |

| | 0.2.0 | 0.1.0 |
|---|---|---|
| MB/s, 1 thread | **2.8** | 2.0 |
| forms sent to the model | 380,995 | 376,050 |
| forms/s, 1 thread | **6K** | 4K |
| forms/s, model + decoder only | **11K** (10K–11K) | 6K (6K–6K) |
| peak memory, MB | **244** | 283 |

### Latency and loading

| | 0.2.0 | 0.1.0 |
|---|---|---|
| `predict`, one sentence: median / p95, ms | **1.75 / 3.45** | 2.60 / 5.35 |
| `from_pretrained`, s (median) | **0.42** | 0.51 |

### Observations

- On a 10× larger corpus one thread is 2.2× faster: the model parses each
  unique form once, and the share of new forms falls as the corpus grows
  (8.5% of tokens on 100K, 2.8% on 1M).
- The ceiling of `process_file` is the tokenizer and dedup in the main thread,
  about 630K tokens/s. On 1M sentences 8 workers reach 63% of it.
- In 0.1.0, workers are slower than a single thread.
- Both versions share the tokenizer, so its speed is the same.

## Speed (`speed.py`)

```
python bench/get_corpus.py
python bench/speed.py --env 0.2.0=<python> --env 0.1.0+viterbi=<python>
```

`<python>` is the interpreter of an environment with the slovorez version to
measure. Without `--env` the current environment is measured. Every environment
needs `psutil`.

**Corpus.** Leipzig Corpora Collection, `rus_news_2023_1M` (CC BY 4.0).
`get_corpus.py` downloads the archive to `bench/data/`, checks its sha256, takes
the sentences without their numbers, drops the lines where the C++ tokenizer
stops (a chain of 9+ hyphenated parts or a word longer than 128 characters; one
line in this corpus), shuffles with a fixed seed (the archive is sorted
alphabetically) and writes `leipzig-1M.txt` and `leipzig-100K.txt` (the first
100K lines). The sha256 of the output is pinned in the script.

**What is measured** (on the 100K and 1M corpora):

- `process_file` in one thread and with 2, 4, 8 workers;
- tokenizer + dedup only (the main thread's work);
- model + decoder only, on the corpus's unique forms (1M only);
- `predict` latency for one sentence (1000 sentences, median and p95);
- `from_pretrained` time.

**Protocol.** Each measurement is a new process with an empty predictions cache.
One warm-up run is discarded, then 3 runs; if the spread (max − min) / median
exceeds 5%, up to 5 runs. Tables show the median and min–max.

**Metrics.** Tokens are Russian words from the tokenizer, length 1..64, the same
for every version (the bench checks that their counts match). So tokens/s
compare the time spent on the same text. The number of forms sent to the model
is always shown: the models have different `base_dict` dictionaries, and 0.1.0
skips words shorter than 4 letters. The bench stops if the tokenizer covers less
than 99% of the words matched by the regex `[а-яё]+(?:-[а-яё]+)*`.

**Comparison** is release against release (model + code): 0.2.0 with
`slovorez-flash`, and 0.1.0 with `slovorez-test` and `use_viterbi=True`.

Output: `bench/results/speed-<date>.json` (raw numbers) and `.md` (table).
`bench/data/` and `bench/results/` are not kept in git.

## Profile (`bench_line_profile.py`)

Line-by-line profile and memory of the sequential pipeline: tokenizer, dedup,
model, Viterbi, writing. Writes `results/line_profile.txt`. Only the sequential
mode is profiled, because line_profiler traces a single thread. Needs
`line_profiler` and `psutil`.
