from slovorezCXX import TokenType
import numpy as np

def find_uncached(tokens: list[str], ltokens: list[str], types: np.ndarray, cache_set: set, uncached: set = None, needed_toktype: TokenType = None, min_toklen: int = 1, max_toklen: int = 64) -> set[str]:
    
    if not uncached:
        uncached = set()
    
    if not needed_toktype:
        needed_toktype = TokenType.RUWORD
    
    if not min_toklen:
        min_toklen = 1
    
    if not max_toklen:
        max_toklen = 64
        
    mask = (types == needed_toktype.value)

    matching_indices = np.where(mask)[0]
    matching_tokens = [ltokens[i] for i in matching_indices]

    valid_tokens = {t for t in matching_tokens if min_toklen <= len(t) <= max_toklen}
    return valid_tokens - cache_set
