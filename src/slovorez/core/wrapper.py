import slovorezCXX
from pathlib import Path
from typing import Union
from slovorez.utils import file_exists
from dataclasses import dataclass

class BaseSentencer:
    pass

class FFSentencer(slovorezCXX.FFSentencer, BaseSentencer):
    def __init__(self, file_path: Union[str, Path], validated: bool=False):
        if not validated:
            if not file_exists(file_path):
                raise FileNotFoundError(f"{file_path}")

            if isinstance(file_path, Path):
                file_path = str(file_path)
        
        super().__init__(file_path)
        
        if not self.is_fopen():
            raise PermissionError(f"Cannot open file: {file_path}")
        
        self.file_path = file_path  

class FTSentencer(slovorezCXX.FTSentencer, BaseSentencer):
    def __init__(self, text: str, validated: bool=False):
        if not validated:    
            if not isinstance(text, str):
                raise TypeError(f"expected str, not {type(text).__name__}")

            if not text.strip():
                raise ValueError("Text cannot be empty")
        
        super().__init__(text)
        self.text = text
        
    def is_fopen(self):
        return True

class Sentencer:
    """
    Factory:
        creating a class based on the source data
    """
    def __new__(cls, source: Union[str, Path], batch_size: int = 1024):
        is_file = isinstance(source, Path) or (isinstance(source, str) and file_exists(source))
        
        if is_file:
            source = str(source)
            instance = FFSentencer(source, validated=True)
        else:
            instance = FTSentencer(source, validated=True)
        
        instance.set_batch_size(batch_size)
        
        return instance

    def __init__(self, *args, **kwargs):
        pass

# for name, value in slovorezCXX.TokenType.__members__.items():
#     print(f"{name}: {value}")