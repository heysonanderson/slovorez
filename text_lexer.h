#ifndef SLOVOREZ_LEXER_H
#define SLOVOREZ_LEXER_H

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include "utf8_decoder.h"

constexpr unsigned char ASCII_LETTER_A      = 0x41;
constexpr unsigned char ASCII_LETTER_Z      = 0x5A;
constexpr unsigned char ASCII_LETTER_a      = 0x61;
constexpr unsigned char ASCII_LETTER_z      = 0x7A;
constexpr unsigned char ASCII_NUMBER_0      = 0x30;
constexpr unsigned char ASCII_NUMBER_9      = 0x39;

enum class TokenType {
    NOTTKN,         ///< Not a token
    ENWORD,         ///< English word
    NUMBER,         ///< Number
    PNCTTN,         ///< Punctuation
    RUWORD,         ///< Russian word
    MLWORD          ///< Multi-language word
};
constexpr const char* TokenTypeStr[] = {
    "NOTTKN",
    "ENWORD",
    "NUMBER",
    "PNCTTN",
    "RUWORD",
    "MLWORD"
};

typedef struct {
    char data[128];
    TokenType type;
    size_t size;
} Token;

typedef struct {
    std::vector<Token> tokens;
    Token ctxtoken;
    UTF8Char utf8c;
} LexerContext;

void slovorez_lexer_init(LexerContext* lctx, size_t token_num = 1024);
bool slovorez_lexer_token_get(LexerContext* lctx, unsigned char c);

#endif // SLOVOREZ_LEXER_H
