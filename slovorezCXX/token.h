#ifndef SLOVOREZ_TOKEN_H
#define SLOVOREZ_TOKEN_H

#include "utf8_decoder.h"

enum class TokenType : uint64_t {
    NOTTKN = 0,         ///< Not token
    WRDSPC = 1,         ///< Word spacing
    NWLINE = 2,         ///< New line
    ENWORD = 4,         ///< English word
    NUMBER = 8,         ///< Number
    RUWORD = 16,        ///< Russian word
    PNCTTN = 32,        ///< Punctuation
    UNKNWN = 64         ///< Unknown character
};

constexpr uint8_t NOTTKN_IDX = 0;
constexpr uint8_t WRDSPC_IDX = 1;
constexpr uint8_t NWLINE_IDX = 2;
constexpr uint8_t ENWORD_IDX = 3;
constexpr uint8_t NUMBER_IDX = 4;
constexpr uint8_t RUWORD_IDX = 5;
constexpr uint8_t PNCTTN_IDX = 6;
constexpr uint8_t UNKNWN_IDX = 7;

constexpr TokenType slovorez_ttidx_to_tt[] = {
    TokenType::NOTTKN,
    TokenType::WRDSPC,
    TokenType::NWLINE,
    TokenType::ENWORD,
    TokenType::NUMBER,
    TokenType::RUWORD,
    TokenType::PNCTTN,
    TokenType::UNKNWN
};

typedef struct {
    UTF8Char data[128];
    TokenType type;
    size_t size;
} Token;

inline bool slovorez_token_filter_match(TokenType tt, uint64_t filter_mask)
{
    return static_cast<uint64_t>(tt) & filter_mask;
}

#endif // SLOVOREZ_TOKEN_H
