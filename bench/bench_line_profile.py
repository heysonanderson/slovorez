# -*- coding: utf-8 -*-
"""
Line profiling of the Slovorez pipeline (sequential mode).

Why sequential:
    line_profiler traces only the thread it was enabled in. In threaded
    mode the model and the decoder run in worker threads and would show
    0 s. In sequential mode every stage runs in the main thread, so the
    whole pipeline is visible: C++ tokenizer, dedup, encode, ONNX,
    decode (Viterbi), writer.

Setup:
    The model is built directly (src/main.py is not used). Predictions go
    to a temporary file that is deleted afterwards, so the SeenIndex
    starts with the base dictionary only -- the same workload as a first
    real run on the text.

Profiled functions:
    Slovorez._process_file_sequential, Slovorez._infer_and_persist,
    BaseTokenizer.get_batch_tokens, SeenIndex.filter_unseen,
    SeenIndex.mark_seen, MorphCodec.encode_batch_padded,
    MorphCodec.decode_predictions_batch, _viterbi_batch,
    ModelResource.predict, LogWriter.write

Usage:
    python bench/bench_line_profile.py
    python bench/bench_line_profile.py --text text.txt
    python bench/bench_line_profile.py --decoder argmax --runs 2

Results:
    Report printed and written to bench/results/line_profile.txt
    (or --out).
"""
from __future__ import annotations

import argparse
import io
import os
import tempfile
import threading
import time
from pathlib import Path

import psutil
from line_profiler import LineProfiler

from slovorez.slovorez import Slovorez
from slovorez.core import codec as codec_mod
from slovorez.core.cache import LogWriter, SeenIndex
from slovorez.core.codec import MorphCodec
from slovorez.core.engine import ModelResource
from slovorez.core.tokenizer import BaseTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = ROOT / "large.txt"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "line_profile.txt"

PROFILED = [
    Slovorez._process_file_sequential,
    Slovorez._infer_and_persist,
    BaseTokenizer.get_batch_tokens,
    SeenIndex.filter_unseen,
    SeenIndex.mark_seen,
    MorphCodec.encode_batch_padded,
    MorphCodec.decode_predictions_batch,
    codec_mod._viterbi_batch,
    ModelResource.predict,
    LogWriter.write,
]


def sample_memory(stop: threading.Event, peak: list) -> None:
    process = psutil.Process()
    while not stop.is_set():
        peak[0] = max(peak[0], process.memory_info().rss)
        time.sleep(0.05)


def one_run(text: Path, decoder: str, model_batch: int,
            profiler: LineProfiler) -> tuple[float, int]:
    fd, preds = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(preds)
    try:
        model = Slovorez.from_pretrained(predictions_path=preds)
        model._tokenizer.method = decoder   # anything but "viterbi" decodes by argmax
        model.predict("прогрев модели")      # session creation stays outside the profile
        profiler.enable()
        t0 = time.perf_counter()
        model._process_file_sequential(text, model_batch=model_batch)
        elapsed = time.perf_counter() - t0
        profiler.disable()
        provider = model._model._session.get_providers()[0]
        with open(preds, encoding="utf-8") as f:
            unique = sum(1 for _ in f)
    finally:
        if os.path.exists(preds):
            os.remove(preds)
    print(f"  {elapsed:.2f} s, {unique:,} unique forms, provider {provider}")
    return elapsed, unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Line profile of the sequential pipeline.")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--decoder", default="viterbi", choices=["viterbi", "argmax"])
    parser.add_argument("--model-batch", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    profiler = LineProfiler()
    for fn in PROFILED:
        profiler.add_function(fn)

    print(f"Profiling sequential pipeline on {args.text} "
          f"({args.text.stat().st_size:,} bytes), "
          f"decoder={args.decoder}, model_batch={args.model_batch}")

    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak = [rss_before]
    stop = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(stop, peak), daemon=True)
    sampler.start()

    walls = []
    for run in range(1, args.runs + 1):
        print(f"run {run}:")
        walls.append(one_run(args.text, args.decoder, args.model_batch, profiler))

    stop.set()
    sampler.join()
    rss_after = process.memory_info().rss

    lines = [
        f"Text: {args.text} ({args.text.stat().st_size:,} bytes)",
        f"decoder: {args.decoder}  model_batch: {args.model_batch}  runs: {args.runs}",
        "Wall time per run (profiled; line_profiler adds overhead): "
        + ", ".join(f"{w:.2f} s" for w, _ in walls),
        f"Unique forms per run: {walls[0][1]:,}",
        f"RSS: before {rss_before / 1024**2:.1f} MiB, peak {peak[0] / 1024**2:.1f} MiB, "
        f"after {rss_after / 1024**2:.1f} MiB",
        "",
    ]
    buffer = io.StringIO()
    profiler.print_stats(stream=buffer, stripzeros=True)
    report = "\n".join(lines) + buffer.getvalue()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report saved to: {args.out}")


if __name__ == "__main__":
    main()
