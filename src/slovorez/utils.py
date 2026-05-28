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

    Accepts three forms:
      - absolute path to a directory
      - relative path / name (e.g. "slovorez-v1") -> resolved via resolve_path,
        same two-step lookup (cwd first, then PROJECT_ROOT)

    Returns the resolved directory Path. Raises FileNotFoundError if the
    directory does not exist or config.json is missing inside it.

    Example::

        model_dir = resolve_model_dir("slovorez-test")
        # -> PROJECT_ROOT / "models" / "slovorez-test"  (if that exists)
    """
    candidate = resolve_path(model_name_or_path)

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: '{model_name_or_path}'. "
            f"Resolved to: {candidate}"
        )

    config_file = candidate / MODEL_CONFIG_NAME
    if not config_file.is_file():
        raise FileNotFoundError(
            f"'{MODEL_CONFIG_NAME}' not found inside model directory: {candidate}"
        )

    return candidate


def file_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_file()


def dir_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_dir()
