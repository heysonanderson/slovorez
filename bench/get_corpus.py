"""Download and prepare the speed-bench corpus.

Leipzig Corpora Collection, rus_news_2023_1M (CC BY 4.0):
https://wortschatz.uni-leipzig.de/en/download/Russian

Steps: download the archive (skipped if it is already in bench/data with the
right sha256), take *-sentences.txt, strip the "<id>\t" prefix, drop lines
that break the C++ tokenizer (C1: a chain of 9+ hyphenated parts or a word
longer than 128 characters stops it for the rest of the file), shuffle with a
fixed seed (the file is sorted alphabetically, so a prefix would be biased),
write the full corpus and its first 100K lines. The output sha256 is checked.

    python bench/get_corpus.py
"""
from __future__ import annotations

import hashlib
import random
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://downloads.wortschatz-leipzig.de/corpora/rus_news_2023_1M.tar.gz"
ARCHIVE_SHA256 = "9e09e5298f4a2a2dffed14d00478bbe61f02e53647986150879520c877e8f76d"
MEMBER = "rus_news_2023_1M/rus_news_2023_1M-sentences.txt"
SEED = 0
SMALL_LINES = 100_000

# A mismatch means the preparation is not reproducible.
OUTPUT_SHA256: dict[str, str | None] = {
    "leipzig-1M.txt": "3a61c20a9323437efa5c3be5c6c672d3c6aa444b290628ddd78b5d3083be589e",
    "leipzig-100K.txt": "795b7253c998ba95a61a774ffd7048e59aa631a2df216c57121d22550002a2e4",
}

DATA = Path(__file__).resolve().parent / "data"

# C1 (plans: 05-cxx-review). Over-broad on purpose: dropping a few extra lines
# is harmless, keeping one that stops the tokenizer is not.
_KILLER = re.compile(r"\w+(?:-\w+){8,}|\w{129,}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(archive: Path) -> None:
    if archive.is_file() and sha256(archive) == ARCHIVE_SHA256:
        print(f"archive ok: {archive}")
        return
    print(f"downloading {URL} ...")
    tmp = archive.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    got = sha256(tmp)
    if got != ARCHIVE_SHA256:
        tmp.unlink()
        sys.exit(f"archive sha256 {got} != {ARCHIVE_SHA256}")
    tmp.replace(archive)


def read_sentences(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile(MEMBER)
        if member is None:
            sys.exit(f"{MEMBER} not found in {archive}")
        raw = member.read().decode("utf-8")
    lines = []
    for row in raw.splitlines():
        _, _, text = row.partition("\t")
        text = text.strip()
        if text:
            lines.append(text)
    return lines


def main() -> None:
    DATA.mkdir(exist_ok=True)
    archive = DATA / "rus_news_2023_1M.tar.gz"
    fetch(archive)

    lines = read_sentences(archive)
    kept = [s for s in lines if not _KILLER.search(s)]
    dropped = len(lines) - len(kept)
    random.Random(SEED).shuffle(kept)
    print(f"sentences: {len(lines):,}, dropped (C1): {dropped:,}, kept: {len(kept):,}")

    outputs = {"leipzig-1M.txt": kept, "leipzig-100K.txt": kept[:SMALL_LINES]}
    bad = False
    for name, part in outputs.items():
        path = DATA / name
        path.write_text("\n".join(part) + "\n", encoding="utf-8", newline="\n")
        got = sha256(path)
        want = OUTPUT_SHA256[name]
        status = "ok" if got == want else ("NOT PINNED" if want is None else "MISMATCH")
        bad |= status == "MISMATCH"
        print(f"{name}: {len(part):,} lines, {path.stat().st_size:,} bytes, sha256 {got} [{status}]")
    if bad:
        sys.exit("output sha256 mismatch")


if __name__ == "__main__":
    main()
