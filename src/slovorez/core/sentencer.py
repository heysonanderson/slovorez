import slovorezCXX
from pathlib import Path
from typing import Union
from slovorez.utils import resolve_path

class BaseSentencer:
    pass

class FFSentencer(slovorezCXX.FFSentencer, BaseSentencer):
    def __init__(self, file_path: Union[str, Path], validated: bool=False):
               
        if not validated:
            abs_path = resolve_path(file_path)
            if not abs_path.exists():
                raise FileNotFoundError(f"File not found: {abs_path}")
        else:
            abs_path = file_path

        super().__init__(str(abs_path))
        
        if not self.is_fopen():
            raise PermissionError(f"Cannot open file: {abs_path}")
        
        self.file_path = abs_path

class FTSentencer(slovorezCXX.FTSentencer, BaseSentencer):
    def __init__(self, text: str, validated: bool=False):
        if not validated:    
            if not isinstance(text, str):
                raise TypeError(f"expected str, not {type(text).__name__}")

            if not text.strip():
                raise ValueError("Text cannot be empty")
        
        super().__init__(text)
        
    def is_fopen(self):
        return True

class Sentencer:
    """
    Factory: creating a class based on the source data
    """
    def __new__(cls, source: Union[str, Path], batch_size: int = 1024):
        abs_path = resolve_path(source)

        if abs_path.is_file():
            instance = FFSentencer(abs_path, validated=True)
        else:
            instance = FTSentencer(str(source), validated=True)
        
        instance.set_batch_size(batch_size)
        return instance