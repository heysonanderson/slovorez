from slovorez.core.models import Token, TokenStream, WordAnalysis

def to_token_stream(raw_text: str, name: str = "main")-> TokenStream:

    if not raw_text or not raw_text.strip():
        return TokenStream(
            name=name,
            tokens=[]
        ) 

    tokens_list = []

    parts = raw_text.split()

    for part in parts:
        tokens_list.append(
            Token(
                text=part,
                lookup_key=part.lower()
            )
        )

    return TokenStream(
        name=name,
        tokens=tokens_list
    )


# def find_uncached(stream: TokenStream, cache: dict):

#     missing_keys={
#         t.lookup_key for t in stream.tokens
#         if t.lookup_key not in cache
#     }

#     if missing_keys: 
#         batch_to_process = list(missing_keys)

#     print(f"{len(batch_to_process) / len(stream.tokens)}")

#     return batch_to_process

def find_uncached(stream: TokenStream, cache: dict):
    missing_keys = set()
    missing_count = 0
    
    for t in stream.tokens:
        if t.lookup_key not in cache:
            missing_keys.add(t.lookup_key)
            missing_count += 1
    
    batch_to_process = list(missing_keys)
    
    return batch_to_process, missing_count