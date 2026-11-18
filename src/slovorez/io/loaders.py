import json
from pathlib import Path
from typing import Union, Any

def to_json(json_object: Any, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_object, f, indent=2, ensure_ascii=False)

def load_json(path: Union[str, Path]) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def append_to_jsonl(path: Union[str, Path], data: dict):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def stream_jsonl(path: Union[str, Path]):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)