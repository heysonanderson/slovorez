def find_uncached(ltokens: list[str], cache_set: set, uncached: set = None, min_toklen: int = 1, max_toklen: int = 64) -> set[str]:
    
    if not uncached:
        uncached = set()
    
    if not min_toklen:
        min_toklen = 1
    
    if not max_toklen:
        max_toklen = 64
        
    valid_tokens = {t for t in ltokens if min_toklen <= len(t) <= max_toklen}
    return valid_tokens - cache_set
