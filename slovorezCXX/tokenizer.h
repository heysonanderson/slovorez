#ifndef SLOVOREZ_TOKENIZER_H
#define SLOVOREZ_TOKENIZER_H

#include "utf8_decoder.h"
#include "token.h"
#include "lexer.h"

constexpr size_t MAX_TOKEN_SEQUENCE_SIZE = 16;

typedef struct {
    Token tokens[MAX_TOKEN_SEQUENCE_SIZE];
    TokenType type;
    size_t size;
} TokenContext;

typedef struct {
    Token rtoken;
    TokenContext tokenctx;
    LexerContext lctx;
} TokenizerContext;

void slovorez_tokenizer_init(TokenizerContext* tctx);
bool slovorez_tokenizer_token_get(TokenizerContext* tctx, unsigned char c);
bool slovorez_tokenizer_end(TokenizerContext* tctx);

#endif // SLOVOREZ_TOKENIZER_H
