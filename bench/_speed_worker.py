"""One speed measurement in a fresh process; prints one JSON line to stdout.

Started by bench/speed.py with the Python of the environment being measured,
so it must work with every slovorez version in the table (0.1.0 and 0.2.0).

    python bench/_speed_worker.py <scenario> <corpus> [--workers N] [--sentences N]

Scenarios:
    info       versions and default settings
    file       process_file; --workers 0 = sequential, N = threaded with N workers
    tokenizer  C++ tokenizer + dedup only (the main thread's work)
    model      model + decoder on the unique forms of the corpus (not timed: building the list)
    predict    predict() latency per sentence
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from importlib.metadata import version as pkg_version
from pathlib import Path


def peak_memory_mb() -> float:
    import psutil
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)      # Windows
    if peak is None:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = rss if sys.platform == "darwin" else rss * 1024
    return peak / 2**20


class V020:
    """slovorez 0.2.x: bundled slovorez-flash, Viterbi set by the model config."""

    decoder = "viterbi (from config)"

    def __init__(self, preds: Path):
        from slovorez import Slovorez
        self.m = Slovorez.from_pretrained(predictions_path=preds)
        self.minlen, self.maxlen = self.m._index.minlen, self.m._index.maxlen
        self.name = self.m._model_name

    def process_file(self, path: Path, workers: int) -> None:
        if workers:
            self.m.process_file(path, threaded_mode=True, max_workers=workers)
        else:
            self.m.process_file(path)

    def infer(self, words: list[str]) -> None:
        tok = self.m._tokenizer
        encoded, lengths = tok.encode_batch_padded(words)
        logits = self.m._model.predict(encoded)
        tok.decode_predictions_batch(words, logits, lengths, self.name)

    def predict(self, text: str) -> list:
        return self.m.predict(text)


class V010:
    """slovorez 0.1.x: bundled slovorez-test, use_viterbi=True everywhere."""

    decoder = "viterbi (use_viterbi=True)"

    def __init__(self, preds: Path):
        import slovorez
        # PY-1: 0.1.0 also loads predictions.jsonl from the model folder in
        # site-packages, whatever output_path says. Start from an empty cache.
        stale = Path(slovorez.__file__).parent / "models" / "slovorez-test" / "predictions.jsonl"
        if stale.is_file():
            stale.unlink()
        from slovorez import Slovorez
        self.m = Slovorez.from_pretrained("slovorez-test", output_path=preds, device="cpu")
        self.minlen, self.maxlen = self.m._index.min_len, self.m._index.max_len
        self.name = self.m._model_name

    def process_file(self, path: Path, workers: int) -> None:
        if workers:
            self.m.process_file(path, threaded_mode=True, max_workers=workers, use_viterbi=True)
        else:
            self.m.process_file(path, use_viterbi=True)

    def infer(self, words: list[str]) -> None:
        tok = self.m._tokenizer
        logits = self.m._model.predict(tok.encode_batch(words))
        tok.decode_predictions_detail(words, logits, self.name, use_viterbi=True)

    def predict(self, text: str) -> list:
        return self.m.predict(text, use_viterbi=True)


def detect():
    """Adapter by API, not by metadata: an old editable install keeps stale metadata."""
    import inspect
    import slovorez
    from slovorez import Slovorez
    is_020 = "predictions_path" in inspect.signature(Slovorez.from_pretrained).parameters
    ver = pkg_version("slovorez")
    pyproject = Path(slovorez.__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():                       # editable install: the repo is the source of truth
        import re
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
        if m:
            ver = f"{m.group(1)} (editable)"
    return ver, (V020 if is_020 else V010)


def load(preds: Path):
    ver, adapter = detect()
    t0 = time.perf_counter()
    a = adapter(preds)
    return ver, a, time.perf_counter() - t0


def defaults() -> dict:
    import slovorez.slovorez as s
    return {"batch_size": s._DEFAULT_BATCH_SIZE, "model_batch": s._DEFAULT_MODEL_BATCH}


def tokenize(path: Path, minlen: int, maxlen: int, batch_size: int):
    """Same calls as the main thread of process_file: tokenizer batches + set() per batch."""
    from slovorez.core.tokenizer import FFTokenizer
    from slovorezCXX import TokenType
    tok = FFTokenizer(path)
    tok.set_batch_size(batch_size)
    tok.set_filter(TokenType.RUWORD)
    tok.set_token_min_len(minlen)
    tok.set_token_max_len(maxlen)
    tokens, unique = 0, set()
    batch = tok.get_batch_tokens()
    while batch:
        tokens += len(batch)
        unique |= set(batch)
        batch = tok.get_batch_tokens()
    return tokens, unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["info", "file", "tokenizer", "model", "predict"])
    ap.add_argument("corpus", type=Path, nargs="?")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--sentences", type=int, default=1000)
    ap.add_argument("--minlen", type=int, help="tokenizer only: override the model's minlen")
    ap.add_argument("--maxlen", type=int, help="tokenizer only: override the model's maxlen")
    args = ap.parse_intermixed_args()

    out: dict = {"scenario": args.scenario}
    with tempfile.TemporaryDirectory() as tmp:
        preds = Path(tmp) / "predictions.jsonl"
        ver, a, load_s = load(preds)
        out.update(slovorez=ver, model=a.name, from_pretrained_s=load_s)
        d = defaults()

        if args.scenario == "info":
            import numpy
            import onnxruntime
            out.update(
                python=platform.python_version(), numpy=numpy.__version__,
                onnxruntime=onnxruntime.__version__, decoder=a.decoder,
                minlen=a.minlen, maxlen=a.maxlen, **d,
            )

        elif args.scenario == "file":
            t0 = time.perf_counter()
            a.process_file(args.corpus, args.workers)
            out["seconds"] = time.perf_counter() - t0
            with open(preds, encoding="utf-8") as f:
                out["forms_to_model"] = sum(1 for _ in f)
            out["workers"] = args.workers

        elif args.scenario == "tokenizer":
            lo = a.minlen if args.minlen is None else args.minlen
            hi = a.maxlen if args.maxlen is None else args.maxlen
            t0 = time.perf_counter()
            tokens, unique = tokenize(args.corpus, lo, hi, d["batch_size"])
            out.update(seconds=time.perf_counter() - t0, tokens=tokens, unique=len(unique),
                       minlen=lo, maxlen=hi)

        elif args.scenario == "model":
            _, unique = tokenize(args.corpus, a.minlen, a.maxlen, d["batch_size"])
            words = sorted(unique, key=lambda w: (len(w), w))   # little padding, as in process_file
            step = d["model_batch"]
            t0 = time.perf_counter()
            for i in range(0, len(words), step):
                a.infer(words[i:i + step])
            out.update(seconds=time.perf_counter() - t0, forms=len(words))

        elif args.scenario == "predict":
            with open(args.corpus, encoding="utf-8") as f:
                lines = [line.strip() for line in f]
            every = max(1, len(lines) // args.sentences)
            sample = [s for s in lines[::every] if s][: args.sentences]
            for s in sample[:50]:                      # warm-up
                a.predict(s)
            times, words = [], 0
            for s in sample:
                t0 = time.perf_counter()
                words += len(a.predict(s))
                times.append(time.perf_counter() - t0)
            times.sort()
            out.update(
                sentences=len(sample), words=words,
                median_ms=statistics.median(times) * 1e3,
                p95_ms=times[int(0.95 * (len(times) - 1))] * 1e3,
            )

        out["peak_mb"] = peak_memory_mb()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
