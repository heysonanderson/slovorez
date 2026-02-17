#ifndef SLOVOREZ_LEXER_H
#define SLOVOREZ_LEXER_H

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "utf8_decoder.h"

constexpr uint32_t ASCII_LETTER_A      = 0x41;
constexpr uint32_t ASCII_LETTER_Z      = 0x5A;
constexpr uint32_t ASCII_LETTER_a      = 0x61;
constexpr uint32_t ASCII_LETTER_z      = 0x7A;
constexpr uint32_t ASCII_NUMBER_0      = 0x30;
constexpr uint32_t ASCII_NUMBER_9      = 0x39;

constexpr uint32_t UTF8_RU_LETTERS_RANGE_START = 0xD090;
constexpr uint32_t UTF8_RU_LETTERS_RANGE_END   = 0xD18F;
constexpr uint32_t UTF8_RU_UPPERCASE_LETTER_IO = 0xD081;
constexpr uint32_t UTF8_RU_LOWERCASE_LETTER_IO = 0xD191;

enum class TokenType {
    NOTTKN,         ///< Not a token
    WRDSPC,         ///< Word spacing
    NWLINE,         ///< New line
    ENWORD,         ///< English word
    NUMBER,         ///< Number
    RUWORD,         ///< Russian word
    PNCTTN,         ///< Punctuation
    UNKNWN          ///< Unknown character
};
constexpr const char* TokenTypeStr[] = {
    "NOTTKN",
    "WRDSPC",
    "NWLINE",
    "ENWORD",
    "NUMBER",
    "RUWORD",
    "PNCTTN",
    "UNKNWN"
};

typedef struct {
    UTF8Char data[128];
    TokenType type;
    size_t size;
} Token;

typedef struct {
    std::vector<Token> tokens;
    Token ctxtoken;
    UTF8Char utf8c;
} LexerContext;

bool _slovorez_lexer_is_enletter(unsigned char c);
bool _slovorez_lexer_is_number(unsigned char c);
bool _slovorez_lexer_is_ruletter(const UTF8Char& utf8c);
void _slovorez_lexer_token_insert_utf8_char(Token* token, UTF8Char* utf8c);
void _slovorez_lexer_new_token(LexerContext* lctx);
void _slovorez_lexer_token_finalize(LexerContext* lctx);
void slovorez_lexer_init(LexerContext* lctx, size_t token_num = 1024);
bool slovorez_lexer_token_get(LexerContext* lctx, unsigned char c);

#endif // SLOVOREZ_LEXER_H
