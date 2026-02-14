#ifndef SLOVOREZ_LEXER_H
#define SLOVOREZ_LEXER_H

#include <cstdio>
#include <cstring>
#include <cstddef>
#include <vector>

// NOTE: Cyrillic symbols are usually 2 bytes

constexpr unsigned char ASCII_END           = 0x80;
constexpr unsigned char FIRST_3BIT_MASK     = 0xE0;
constexpr unsigned char UTF8_2BYTE_SGNT     = 0xC0;

enum class TokenType {
    ENWORD,
    NUMBER,
    PNCTTN,
    RUWORD
};
constexpr const char* SyntaxPartTypeStr[] = {
    "ENWORD",
    "NUMBER",
    "PNCTTN",
    "RUWORD"
};


typedef struct {
    char data[128];
    TokenType type;
    size_t size;
} Token;

typedef struct {
    std::vector<Token> tokens;
    unsigned char buffer[4];
    size_t bytes_expected;
    size_t bytes_gathered;
} LexerContext;

void _slovorez_lexer_reset_ctx(LexerContext* lctx);
void slovorez_lexer_init(LexerContext* lctx, size_t token_num = 1024);
void slovorez_lexer_char(LexerContext* lctx, unsigned char c);

#endif // SLOVOREZ_LEXER_H
