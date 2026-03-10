from slovorezCXX import TokenVector, TokenType


def find_uncached(stream: TokenVector, cache_set: set, uncached: set = None, needed_toktype: TokenType = None, min_toklen: int = 1, max_toklen: int = 64) -> set[str]:
    
    if not uncached:
        uncached = set()
    
    if not needed_toktype:
        needed_toktype = TokenType.RUWORD
    
    if not min_toklen:
        min_toklen = 1
    
    if not max_toklen:
        max_toklen = 64
        
    ml = 0
    for t in stream:
        size = t.size
        if t.type != needed_toktype or size > max_toklen or size < min_toklen:
            continue
        lookup_key = t.str.lower()
        if lookup_key not in cache_set:
            uncached.add(lookup_key)
            ml+=1
    return uncached, ml