#ifndef SLOVOREZ_LEXER_H
#define SLOVOREZ_LEXER_H

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "utf8_decoder.h"

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

void slovorez_lexer_init(LexerContext* lctx, size_t token_num = 1024);
bool slovorez_lexer_token_get(LexerContext* lctx, unsigned char c);

static TokenType slovorez_get_utf8_tt(const UTF8Char& utf8c)
{
    uint32_t bval = utf8c.get_bval();
    switch (bval)
    {
        case 0x0A:          // New line
        {
            return TokenType::NWLINE;
        }
        case 0x20:          // Space
        case 0xC2A0:        // No-break space
        {
            return TokenType::WRDSPC;
        }
        case 0x30:          // 0
        case 0x31:          // 1
        case 0x32:          // 2
        case 0x33:          // 3
        case 0x34:          // 4
        case 0x35:          // 5
        case 0x36:          // 6
        case 0x37:          // 7
        case 0x38:          // 8
        case 0x39:          // 9
        {
            return TokenType::NUMBER;
        }
        case 0x41:          // A
        case 0x42:          // B
        case 0x43:          // C
        case 0x44:          // D
        case 0x45:          // E
        case 0x46:          // F
        case 0x47:          // G
        case 0x48:          // H
        case 0x49:          // I
        case 0x4A:          // J
        case 0x4B:          // K
        case 0x4C:          // L
        case 0x4D:          // M
        case 0x4E:          // N
        case 0x4F:          // O
        case 0x50:          // P
        case 0x51:          // Q
        case 0x52:          // R
        case 0x53:          // S
        case 0x54:          // T
        case 0x55:          // U
        case 0x56:          // V
        case 0x57:          // W
        case 0x58:          // X
        case 0x59:          // Y
        case 0x5A:          // Z
        case 0x61:          // a
        case 0x62:          // b
        case 0x63:          // c
        case 0x64:          // d
        case 0x65:          // e
        case 0x66:          // f
        case 0x67:          // g
        case 0x68:          // h
        case 0x69:          // i
        case 0x6A:          // j
        case 0x6B:          // k
        case 0x6C:          // l
        case 0x6D:          // m
        case 0x6E:          // n
        case 0x6F:          // o
        case 0x70:          // p
        case 0x71:          // q
        case 0x72:          // r
        case 0x73:          // s
        case 0x74:          // t
        case 0x75:          // u
        case 0x76:          // v
        case 0x77:          // w
        case 0x78:          // x
        case 0x79:          // y
        case 0x7A:          // z
        {
            return TokenType::ENWORD;
        }
        case 0xD081:        // Ё
        case 0xD090:        // А
        case 0xD091:        // Б
        case 0xD092:        // В
        case 0xD093:        // Г
        case 0xD094:        // Д
        case 0xD095:        // Е
        case 0xD096:        // Ж
        case 0xD097:        // З
        case 0xD098:        // И
        case 0xD099:        // Й
        case 0xD09A:        // К
        case 0xD09B:        // Л
        case 0xD09C:        // М
        case 0xD09D:        // Н
        case 0xD09E:        // О
        case 0xD09F:        // П
        case 0xD0A0:        // Р
        case 0xD0A1:        // С
        case 0xD0A2:        // Т
        case 0xD0A3:        // У
        case 0xD0A4:        // Ф
        case 0xD0A5:        // Х
        case 0xD0A6:        // Ц
        case 0xD0A7:        // Ч
        case 0xD0A8:        // Ш
        case 0xD0A9:        // Щ
        case 0xD0AA:        // Ъ
        case 0xD0AB:        // Ы
        case 0xD0AC:        // Ь
        case 0xD0AD:        // Э
        case 0xD0AE:        // Ю
        case 0xD0AF:        // Я
        case 0xD0B0:        // а
        case 0xD0B1:        // б
        case 0xD0B2:        // в
        case 0xD0B3:        // г
        case 0xD0B4:        // д
        case 0xD0B5:        // е
        case 0xD0B6:        // ж
        case 0xD0B7:        // з
        case 0xD0B8:        // и
        case 0xD0B9:        // й
        case 0xD0BA:        // к
        case 0xD0BB:        // л
        case 0xD0BC:        // м
        case 0xD0BD:        // н
        case 0xD0BE:        // о
        case 0xD0BF:        // п
        case 0xD180:        // р
        case 0xD181:        // с
        case 0xD182:        // т
        case 0xD183:        // у
        case 0xD184:        // ф
        case 0xD185:        // х
        case 0xD186:        // ц
        case 0xD187:        // ч
        case 0xD188:        // ш
        case 0xD189:        // щ
        case 0xD18A:        // ъ
        case 0xD18B:        // ы
        case 0xD18C:        // ь
        case 0xD18D:        // э
        case 0xD18E:        // ю
        case 0xD18F:        // я
        case 0xD191:        // ё
        {
            return TokenType::RUWORD;
        }
        case 0x21:          // !
        case 0x22:          // "
        case 0x23:          // #
        case 0x24:          // $
        case 0x25:          // %
        case 0x26:          // &
        case 0x27:          // '
        case 0x28:          // (
        case 0x29:          // )
        case 0x2A:          // *
        case 0x2B:          // +
        case 0x2C:          // ,
        case 0x2D:          // -
        case 0x2E:          // .
        case 0x2F:          // /
        case 0x3A:          // :
        case 0x3B:          // ;
        case 0x3C:          // <
        case 0x3D:          // =
        case 0x3E:          // >
        case 0x3F:          // ?
        case 0x40:          // @
        case 0x5B:          // [
        case 0x5C:          // Backslash
        case 0x5D:          // ]
        case 0x5E:          // ^
        case 0x5F:          // _
        case 0x60:          // `
        case 0x7B:          // {
        case 0x7C:          // |
        case 0x7D:          // }
        case 0x7E:          // ~
        case 0xC2B7:        // ·
        case 0xC2AB:        // «
        case 0xC2BB:        // »
        case 0xE28496:      // №
        case 0xE28090:      // ‐
        case 0xE28091:      // ‑
        case 0xE28092:      // ‒
        case 0xE28093:      // –
        case 0xE28094:      // —
        case 0xE28095:      // ―
        case 0xE28096:      // ‖
        case 0xE28097:      // ‗
        case 0xE28098:      // '
        case 0xE28099:      // '
        case 0xE2809A:      // ‚
        case 0xE2809B:      // ‛
        case 0xE2809C:      // "
        case 0xE2809D:      // "
        case 0xE2809E:      // „
        case 0xE2809F:      // ‟
        case 0xE280A4:      // ․
        case 0xE280A5:      // ‥
        case 0xE280A6:      // …
        case 0xE280A7:      // ‧
        {
            return TokenType::PNCTTN;
        }
        default:
        {
            return TokenType::UNKNWN;
        }
    }
}

#endif // SLOVOREZ_LEXER_H
