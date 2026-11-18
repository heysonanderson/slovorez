"""Everything slovorez writes lives under $SLOVOREZ_HOME.

$SLOVOREZ_HOME  (default: $XDG_CACHE_HOME/slovorez or ~/.cache/slovorez)
└── predictions/<name>/<version>/predictions.jsonl

Model directories are read-only: predictions are never written next to
model files. Only subdirectories live in the root. Nothing here creates
directories; writers do that right before writing.
"""

import os
from pathlib import Path

_PREDS_DIR = "predictions"
_PREDS_FILE = "predictions.jsonl"

def _expand(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p)))

def slovorez_home() -> Path:
    if env := os.environ.get("SLOVOREZ_HOME"):
        return _expand(env).resolve()
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return _expand(xdg) / "slovorez"
    return Path.home() / ".cache" / "slovorez"

def check_part(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be a string, got {type(value).__name__}: {value!r}"
        )
    if not value or value in {".", ".."} or "/" in value or "\\" in value or ":" in value:
        raise ValueError(
            f"Invalid {field}: {value!r}. "
            f"Must be non-empty and must not contain '/', '\\', ':' or be '.'/'..'."
        )

def model_predictions_path(name: str, version: str) -> Path:
    check_part(name, "name")
    check_part(version, "version")
    return slovorez_home() / _PREDS_DIR / name / version / _PREDS_FILE