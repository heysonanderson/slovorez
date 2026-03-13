import slovorezCXX

TEXT_FILE = "text.txt"

sentencer = slovorezCXX.FFSentencer(TEXT_FILE)

bcount = 1
for batch in sentencer.stream:
    types = batch["types"]
    tokens = batch["text"].split('\0')[:-1]
    ltext = batch["text"].lower()
    ltokens = ltext.split('\0')[:-1]

    for i in range(len(tokens)):
        print(f"{slovorezCXX.TokenType(types[i])} - {tokens[i]} ({ltokens[i]})")
    
    print(bcount)
    bcount = bcount + 1
