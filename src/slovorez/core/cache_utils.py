def find_uncached(ltokens: list[str], cache_set: set, min_toklen: int = 1, max_toklen: int = 64) -> set[str]:
    
    if not min_toklen:
        min_toklen = 1
    
    if not max_toklen:
        max_toklen = 64
        
    return {t for t in ltokens if min_toklen <= len(t) <= max_toklen and t not in cache_set}
