from pathlib import Path
from typing import Union

LIBRARY_ROOT = Path(__file__).resolve().parent.parent 
PROJECT_ROOT = LIBRARY_ROOT.parent 

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

def file_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_file()

def dir_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_dir()