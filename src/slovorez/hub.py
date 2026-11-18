from pathlib import Path
from typing import Union

_BUNDLED_MODELS_DIR = Path(__file__).resolve().parent / "models"
CONFIG_NAME = "config.json"

def bundled_models() -> list[str]:
    return sorted(
        p.name for p in _BUNDLED_MODELS_DIR.iterdir()
        if (p / CONFIG_NAME).is_file()
    )

def resolve_model_dir(name_or_path: Union[str, Path]) -> Path:
    """Resolve a model directory from a name or path.

    Resolution order:
      1. An existing path (absolute or relative to cwd) -- must contain config.json,
         otherwise FileNotFoundError. A local directory wins over a bundled name.
      2. A bundled model name (see ``bundled_models()``).

    A directory qualifies only if it contains ``config.json``.

    Example:
        resolve_model_dir("slovorez-flash")      # bundled
        resolve_model_dir("./my-finetune")       # local directory
    """
    p = Path(name_or_path)

    if p.exists():
        if not (p / CONFIG_NAME).is_file():
            raise FileNotFoundError(f"'{p}' exists but has no {CONFIG_NAME}")
        return p.resolve()

    name = str(name_or_path)
    if "/" not in name and "\\" not in name:
        candidate = _BUNDLED_MODELS_DIR / name
        if (candidate / CONFIG_NAME).is_file():
            return candidate

    raise FileNotFoundError(
        f"Model directory not found: '{name_or_path}'. "
        f"A model directory must contain '{CONFIG_NAME}'. "
        f"Bundled models: {', '.join(bundled_models())}"
    )