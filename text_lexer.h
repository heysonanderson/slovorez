#ifndef SLOVOREZ_LEXER_H
#define SLOVOREZ_LEXER_H

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <vector>

// NOTE: Cyrillic symbols are usually 2 bytes

constexpr unsigned char ASCII_END           = 0x80;
constexpr unsigned char FIRST_3BIT_MASK     = 0xE0;
constexpr unsigned char UTF8_2BYTE_SGNT     = 0xC0;

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
    Token last_token;
    unsigned char buffer[4];
    size_t bytes_expected;
    size_t bytes_gathered;
} LexerContext;

void _slovorez_lexer_reset_ctx(LexerContext* lctx);
bool _slovorez_lexer_is_enword(unsigned char c);
bool _slovorez_lexer_is_number(unsigned char c);
void _slovorez_lexer_new_token(LexerContext* lctx);
void _slovorez_lexer_insert_1byte(Token* token, unsigned char c);
void _slovorez_lexer_insert_2bytes(Token* token, unsigned char* buffer);
void _slovorez_lexer_push_token(LexerContext* lctx);
void slovorez_lexer_init(LexerContext* lctx, size_t token_num = 1024);
void slovorez_lexer_char(LexerContext* lctx, unsigned char c);

#endif // SLOVOREZ_LEXER_H
