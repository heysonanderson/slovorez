import json
from pathlib import Path
from typing import Union, Any
from slovorez.utils import resolve_path

def ensure_dir(path: Union[str, Path]):
    abs_path = resolve_path(path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)

def to_json(json_object: Any, path: Union[str, Path]):
    abs_path = resolve_path(path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(abs_path, 'w', encoding='utf-8') as f:
        json.dump(json_object, f, indent=2, ensure_ascii=False)

def load_json(path: Union[str, Path]) -> dict:
    abs_path = resolve_path(path)
    with open(abs_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def append_to_jsonl(resolved_path: Union[str, Path], data: dict):
    with open(resolved_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def stream_jsonl(path: Union[str, Path]):
    abs_path = resolve_path(path)
    with open(abs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)