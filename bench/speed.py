"""Speed bench: release against release, each in its own environment.

    python bench/get_corpus.py                      # once
    python bench/speed.py --env 0.2.0=<python> --env 0.1.0=<python>

Every measurement runs in a fresh process (bench/_speed_worker.py) with an
empty predictions cache: one warm-up, then --runs measurements; if the spread
(max - min) / median exceeds --spread, more runs up to --max-runs. The table
shows the median and min-max.

Tokens are Russian words from the C++ tokenizer, length 1..64, the same for
every version (the bench checks it), so tokens/s compare the same text. The
tokenizer must cover >= 99% of the regex [а-яё]+(?:-[а-яё]+)* count, or the
bench stops (C1 would silently cut the corpus).

Raw numbers: bench/results/speed-<date>.json, table: bench/results/speed-<date>.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
WORKER = HERE / "_speed_worker.py"
CORPORA = {"100K": "leipzig-100K.txt", "1M": "leipzig-1M.txt"}
WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*")

sys.path.insert(0, str(HERE))
from get_corpus import OUTPUT_SHA256, sha256  # noqa: E402


# ---------------------------------------------------------------- running

def run_worker(python: str, args: list[str]) -> dict:
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run([python, str(WORKER), *args], capture_output=True, text=True,
                          encoding="utf-8", env=env, cwd=HERE)
    if proc.returncode != 0:
        sys.exit(f"worker failed: {python} {' '.join(args)}\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def series(python: str, args: list[str], opt) -> list[dict]:
    """One warm-up (not kept), then runs until the spread is small enough."""
    run_worker(python, args)
    runs = [run_worker(python, args) for _ in range(opt.runs)]
    while len(runs) < opt.max_runs and spread([r["seconds"] for r in runs]) > opt.spread:
        runs.append(run_worker(python, args))
    shown = " ".join(Path(a).name if os.sep in a or "/" in a else a for a in args)
    print(f"    {shown:<50} " + " ".join(f"{r['seconds']:.2f}" for r in runs) + " s", flush=True)
    return runs


def spread(xs: list[float]) -> float:
    return (max(xs) - min(xs)) / statistics.median(xs)


# ---------------------------------------------------------------- machine

def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True).stdout.strip()
            if out:
                return out
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def machine() -> dict:
    import psutil
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {
        "cpu": cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "date": dt.date.today().isoformat(),
    }


# ---------------------------------------------------------------- corpus

def corpus_info(name: str, path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"{path} not found: run python bench/get_corpus.py")
    got = sha256(path)
    if got != OUTPUT_SHA256[path.name]:
        sys.exit(f"{path}: sha256 {got} != {OUTPUT_SHA256[path.name]}")
    text = path.read_text(encoding="utf-8")
    return {"name": name, "file": path.name, "sha256": got, "bytes": path.stat().st_size,
            "lines": text.count("\n"), "regex_words": len(WORD_RE.findall(text.lower()))}


# ---------------------------------------------------------------- report

def med(runs: list[dict], key: str = "seconds") -> float:
    return statistics.median(r[key] for r in runs)


def fmt_rate(x: float) -> str:
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"


def cell(runs: list[dict], amount: float) -> str:
    """Rate = amount / seconds: median (min-max) over runs."""
    rates = sorted(amount / r["seconds"] for r in runs)
    return f"{fmt_rate(statistics.median(rates))} ({fmt_rate(rates[0])}–{fmt_rate(rates[-1])})"


def markdown(res: dict) -> str:
    m, envs = res["machine"], res["envs"]
    labels = list(envs)
    head = "| | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n"
    row = lambda name, vals: f"| {name} | " + " | ".join(vals) + " |\n"
    out = [f"CPU: {m['cpu']} ({m['cores_physical']} cores / {m['cores_logical']} threads), "
           f"RAM {m['ram_gb']} GB, {m['os']}. Commit {m['commit']}{' (dirty)' if m['dirty'] else ''}, {m['date']}.\n"]
    for label, e in envs.items():
        i = e["info"]
        out.append(f"- {label}: slovorez {i['slovorez']}, model {i['model']}, decoder {i['decoder']}, "
                   f"Python {i['python']}, numpy {i['numpy']}, onnxruntime {i['onnxruntime']}, "
                   f"batch_size {i['batch_size']}, model_batch {i['model_batch']}, "
                   f"word length {i['minlen']}..{i['maxlen']}\n")
    out.append(f"\nCorpus: Leipzig rus_news_2023_1M (CC BY 4.0), shuffled, `bench/get_corpus.py`. "
               f"Median (min–max) of {res['options']['runs']}+ runs after a warm-up, each in a new process "
               f"with an empty cache.\n")
    for size, c in res["corpora"].items():
        tokens = c["tokens"]
        out.append(f"\n**{size} sentences**: {c['bytes'] / 2**20:.0f} MB, {tokens:,} tokens, "
                   f"{c['unique']:,} unique, sha256 `{c['sha256'][:12]}…`\n\n")
        out.append("Tokens/s:\n\n" + head)
        out.append(row("`process_file`, 1 thread", [cell(envs[l]["runs"][size]["file-0"], tokens) for l in labels]))
        for w in res["options"]["workers"]:
            out.append(row(f"`process_file`, {w} workers",
                           [cell(envs[l]["runs"][size][f"file-{w}"], tokens) for l in labels]))
        out.append(row("tokenizer + dedup only", [cell(envs[l]["runs"][size]["tokenizer"], tokens) for l in labels]))
        out.append("\n" + head)
        out.append(row("MB/s, 1 thread", [f"{c['bytes'] / 2**20 / med(envs[l]['runs'][size]['file-0']):.1f}"
                                          for l in labels]))
        out.append(row("forms sent to the model", [f"{envs[l]['runs'][size]['file-0'][0]['forms_to_model']:,}"
                                                  for l in labels]))
        out.append(row("forms/s, 1 thread", [cell(envs[l]["runs"][size]["file-0"],
                                                  envs[l]["runs"][size]["file-0"][0]["forms_to_model"])
                                             for l in labels]))
        if "model" in envs[labels[0]]["runs"][size]:
            out.append(row("forms/s, model + decoder only", [
                cell(envs[l]["runs"][size]["model"], envs[l]["runs"][size]["model"][0]["forms"]) for l in labels]))
        out.append(row("peak memory, MB", [
            f"{max(r['peak_mb'] for runs in envs[l]['runs'][size].values() for r in runs):.0f}" for l in labels]))
    out.append("\nOther:\n\n" + head)
    out.append(row("`predict`, one sentence: median / p95, ms", [
        f"{envs[l]['predict']['median_ms']:.2f} / {envs[l]['predict']['p95_ms']:.2f}" for l in labels]))
    out.append(row("`from_pretrained`, s (median)", [f"{envs[l]['from_pretrained_s']:.2f}" for l in labels]))
    return "".join(out)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", action="append", metavar="LABEL=PYTHON",
                    help="a version to measure (repeat); default: this Python")
    ap.add_argument("--sizes", default="100K,1M")
    ap.add_argument("--workers", default="2,4,8")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-runs", type=int, default=5)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--sentences", type=int, default=1000)
    ap.add_argument("--model-size", default="1M", help="corpus for the model-only scenario")
    opt = ap.parse_args()
    opt.workers = [int(w) for w in opt.workers.split(",")]
    envs = dict(e.split("=", 1) for e in (opt.env or [f"current={sys.executable}"]))
    sizes = opt.sizes.split(",")

    res: dict = {"machine": machine(), "options": vars(opt) | {"env": envs}, "corpora": {}, "envs": {}}
    for size in sizes:
        res["corpora"][size] = corpus_info(size, DATA / CORPORA[size])
    print(json.dumps(res["machine"], ensure_ascii=False))

    for label, python in envs.items():
        print(f"== {label}: {python}", flush=True)
        e = res["envs"][label] = {"python": python, "info": run_worker(python, ["info"]), "runs": {}}
        for size in sizes:
            path = str(DATA / CORPORA[size])
            r = e["runs"][size] = {}
            # same length limits for every version: tokens/s compare the same text
            r["tokenizer"] = series(python, ["tokenizer", path, "--minlen", "1", "--maxlen", "64"], opt)
            r["file-0"] = series(python, ["file", path, "--workers", "0"], opt)
            for w in opt.workers:
                r[f"file-{w}"] = series(python, ["file", path, "--workers", str(w)], opt)
            if size == opt.model_size:
                r["model"] = series(python, ["model", path], opt)
        big = DATA / CORPORA[sizes[-1]]
        e["predict"] = run_worker(python, ["predict", str(big), "--sentences", str(opt.sentences)])
        e["from_pretrained_s"] = statistics.median(
            x["from_pretrained_s"] for runs in e["runs"].values() for rs in runs.values() for x in rs)

    # checks: the same tokens for every version, and the tokenizer covers the text
    for size in sizes:
        c = res["corpora"][size]
        counts = {l: (e["runs"][size]["tokenizer"][0]["tokens"], e["runs"][size]["tokenizer"][0]["unique"])
                  for l, e in res["envs"].items()}
        if len(set(counts.values())) != 1:
            sys.exit(f"{size}: token counts differ between versions: {counts}")
        c["tokens"], c["unique"] = next(iter(counts.values()))
        c["coverage"] = c["tokens"] / c["regex_words"]
        print(f"{size}: tokenizer {c['tokens']:,} / regex {c['regex_words']:,} = {c['coverage']:.4f}")
        if c["coverage"] < 0.99:
            sys.exit(f"{size}: tokenizer coverage {c['coverage']:.4f} < 0.99")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / f"speed-{res['machine']['date']}"
    stem.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = markdown(res)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"written: {stem}.json, {stem}.md")


if __name__ == "__main__":
    main()
