from pathlib import Path
from typing import Union

def _get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            return parent
    return current.parent

PROJECT_ROOT = _get_project_root()

def get_project_path(*path_parts: Union[str, Path], create_dir=False) -> Path:
    full_path = PROJECT_ROOT.joinpath(*path_parts)
    if create_dir:
        full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path

def _resolve_path(path: Union[str, Path, None], *path_parts: Union[str, Path]) -> Path:
    if path is None:
        return get_project_path(*path_parts)
    
    full_path = Path(path).joinpath(*path_parts)
    
    if not full_path.is_absolute():
        return get_project_path(full_path)
        
    return full_path

def file_exists(path: Union[str, Path, None] = None, *path_parts: Union[str, Path]) -> bool:
    return _resolve_path(path, *path_parts).is_file()

def dir_exists(path: Union[str, Path, None] = None, *path_parts: Union[str, Path]) -> bool:
    return _resolve_path(path, *path_parts).is_dir()