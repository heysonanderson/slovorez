from typing import List, Iterator, overload, Union
from enum import Enum
from pathlib import Path

class TokenType(Enum):
    NOTTKN: int
    WRDSPC: int
    NWLINE: int
    ENWORD: int
    NUMBER: int
    RUWORD: int
    PNCTTN: int
    UNKNWN: int

class UTF8Char:
    """UTF8Char Struct
    Attributes:
        data: char raw bytes
        size: number of bytes in the char
        char: encoded char
    """
    data: bytes
    size: int
    char: str

class Token:
    """Token Struct
    Attributes:
        data: list of UTF8Char
        size: number of UTF8Chars
        type: defined in slovorezCXX enum TokenType. Represents token type
        str: encoded string of token
    """
    data: List[UTF8Char]
    size: int
    type: TokenType
    str: str

class TokenVector:
    """Token Container
    """
    def __len__(self) -> int: ...
    
    @overload
    def __getitem__(self, i: int) -> Token: ...
    
    @overload
    def __getitem__(self, s: slice) -> List[Token]: ...
    
    def __iter__(self) -> Iterator[Token]: ...

class FFSentencer:
    """From File (FF). Expects string with absolute path to the file containing text.
    
    Args:
        file_path: str or pathlib.Path, absolute path to the file.
        validated: bool, if `True` skips validation
    
    Example:
        >>> s = FFSentencer("data.txt")
        
        >>> s.set_batch_size(2048)
        
        >>> batch = s.get_batch()
    """
    def __init__(self, file_path: Union[str, Path]): ...
    def get_batch(self) -> TokenVector: ...
    def set_batch_size(self, size: int) -> None: ...
    def is_fopen(self) -> bool: ...

class FTSentencer:
    """From Text (FT). Expects the string.
      
    Args:
        text: str, input text to process
        validated: bool, if `True` skips validation
    
    Example:
        >>> s = FTSentencer("Расплескалась синева. Раскрошилась краснота.")
        
        >>> s.set_batch_size(2048)
        
        >>> batch = s.get_batch()
    """
    def __init__(self, text: str): ...
    def get_batch(self) -> TokenVector: ...
    def set_batch_size(self, size: int) -> None: ...