import slovorezCXX
from pathlib import Path
from typing import Union

class BaseTokenizer:
    def get_batch_tokens(self, lower: bool = True) -> list[str] | None:
        batch = self.get_batch()
        if not batch:
            return None
        text = batch["text"]
        if lower:
            text = text.lower()
        return text.split('\0')[:-1]

class FFTokenizer(slovorezCXX.FFTokenizer, BaseTokenizer):
    def __init__(self, file_path: Union[str, Path]):

        abs_path = Path(file_path).resolve()
        if not abs_path.is_file():
            raise FileNotFoundError(f"Not a file: {abs_path}")

        super().__init__(str(abs_path))
        
        if not self.is_fopen():
            raise PermissionError(f"Cannot open file: {abs_path}")
        
        self.file_path = abs_path

class FTTokenizer(slovorezCXX.FTTokenizer, BaseTokenizer):
    def __init__(self, text: str):

        if not isinstance(text, str):
            raise TypeError(f"expected str, not {type(text).__name__}")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        super().__init__(text)
        
    def is_fopen(self):
        return True
