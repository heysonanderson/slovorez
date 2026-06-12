#ifndef SLOVOREZ_LEXER_H
#define SLOVOREZ_LEXER_H

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "utf8_decoder.h"
#include "unicode_trie.h"
#include "token.h"

typedef struct {
    Token rtoken;
    Token ctxtoken;
    UTF8Char utf8c;
} LexerContext;

void slovorez_lexer_init(LexerContext* lctx);
bool slovorez_lexer_token_get(LexerContext* lctx, unsigned char c);
void slovorez_lexer_token_finalize(LexerContext* lctx);

#endif // SLOVOREZ_LEXER_H
