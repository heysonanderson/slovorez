from pathlib import Path
from typing import Union

# Directory of the installed package itself:
#   in the repo:        <repo>/src/slovorez/
#   after pip install:  <env>/site-packages/slovorez/
PACKAGE_ROOT = Path(__file__).resolve().parent

# Repo root when running from a source checkout (src-layout: src/slovorez -> repo).
# After installation this points somewhere inside site-packages and is meaningless,
# so it is only ever used as a low-priority fallback for development workflows.
_REPO_ROOT = PACKAGE_ROOT.parent.parent

MODEL_CONFIG_NAME = "config.json"
DEFAULT_MODEL_NAME = "slovorez-test"


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path: absolute paths as-is, otherwise relative to cwd,
    falling back to the repo root for development checkouts."""
    p = Path(path)

    if p.is_absolute():
        return p

    cwd_path = p.resolve()
    if cwd_path.exists():
        return cwd_path

    repo_path = (_REPO_ROOT / p).resolve()
    if repo_path.exists():
        return repo_path

    return cwd_path


def resolve_model_dir(model_name_or_path: Union[str, Path] = DEFAULT_MODEL_NAME) -> Path:
    """Resolve a model directory from a name or path.

    Resolution order for a bare name like ``"slovorez-test"``:
      1. Absolute path -- used as-is.
      2. Relative to cwd (``./slovorez-test``).
      3. Relative to cwd/models/ (``./models/slovorez-test``).
      4. Models bundled inside the installed package
         (``site-packages/slovorez/models/slovorez-test``).
      5. Relative to the repo root and repo-root/models/
         (development checkout only).

    A directory qualifies only if it contains ``config.json``.

    Returns:
        The resolved model directory.

    Raises:
        FileNotFoundError: if no candidate directory contains a model.

    Example::

        model_dir = resolve_model_dir()                 # bundled default model
        model_dir = resolve_model_dir("my-finetune")    # ./my-finetune, ./models/my-finetune, ...
        model_dir = resolve_model_dir("/abs/path/to/model")
    """
    p = Path(model_name_or_path)

    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [
            Path.cwd() / p,
            Path.cwd() / "models" / p,
            PACKAGE_ROOT / "models" / p,
            _REPO_ROOT / p,
            _REPO_ROOT / "models" / p,
        ]

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:  # cwd == repo root makes duplicates
            continue
        checked.append(candidate)
        if candidate.is_dir() and (candidate / MODEL_CONFIG_NAME).is_file():
            return candidate

    raise FileNotFoundError(
        f"Model directory not found: '{model_name_or_path}'. "
        f"A model directory must contain '{MODEL_CONFIG_NAME}'. Searched in:\n"
        + "\n".join(f"  - {c}" for c in checked)
    )


def file_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_file()


def dir_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_dir()