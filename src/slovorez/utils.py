from pathlib import Path
from typing import Union

LIBRARY_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LIBRARY_ROOT.parent
MODEL_CONFIG_NAME = "config.json"


def resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)

    if p.is_absolute():
        return p

    cwd_path = p.resolve()
    if cwd_path.exists():
        return cwd_path

    internal_path = (PROJECT_ROOT / p).resolve()
    if internal_path.exists():
        return internal_path

    return cwd_path


def resolve_model_dir(model_name_or_path: Union[str, Path]) -> Path:
    """Resolve a model directory from a name or path.

    Resolution order for a bare name like "slovorez-test":
      1. Absolute path -- used as-is.
      2. Relative to cwd (e.g. ``./slovorez-test``).
      3. Relative to cwd/models/ (e.g. ``./models/slovorez-test``).
      4. Relative to PROJECT_ROOT.
      5. Relative to PROJECT_ROOT/models/.

    Returns the resolved directory Path. Raises FileNotFoundError if the
    directory does not exist or config.json is missing inside it.

    Example::

        model_dir = resolve_model_dir("slovorez-test")
        # Tries cwd/slovorez-test, cwd/models/slovorez-test,
        # PROJECT_ROOT/slovorez-test, PROJECT_ROOT/models/slovorez-test
    """
    p = Path(model_name_or_path)

    candidates = [p] if p.is_absolute() else [
        Path.cwd() / p,
        Path.cwd() / "models" / p,
        PROJECT_ROOT / p,
        PROJECT_ROOT / "models" / p,
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir() and (candidate / MODEL_CONFIG_NAME).is_file():
            return candidate

    raise FileNotFoundError(
        f"Model directory not found: '{model_name_or_path}'. "
        f"Searched in:\n"
        + "\n".join(f"  - {c.resolve()}" for c in candidates)
    )


def file_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_file()


def dir_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_dir()
