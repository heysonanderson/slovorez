#ifndef SLOVOREZ_UTF8_DECODER_H
#define SLOVOREZ_UTF8_DECODER_H

#include <cstring>
#include <cstdint>

typedef struct UTF8Char {
    unsigned char bytes[4];
    uint32_t codepoint;
    size_t curr_size;
    size_t size;
} UTF8Char;

constexpr unsigned char UTF8_CBYTE_MASK = 0x80;
constexpr unsigned char UTF8_CBYTE_SGNT = 0x80; // 10yyzzzz
constexpr unsigned char UTF8_1BYTE_MASK = 0x80;
constexpr unsigned char UTF8_1BYTE_SGNT = 0x00; // 0yyyzzzz
constexpr unsigned char UTF8_2BYTE_MASK = 0xE0;
constexpr unsigned char UTF8_2BYTE_SGNT = 0xC0; // 110xxxyy
constexpr unsigned char UTF8_3BYTE_MASK = 0xF0;
constexpr unsigned char UTF8_3BYTE_SGNT = 0xE0; // 1110wwww
constexpr unsigned char UTF8_4BYTE_MASK = 0xF8;
constexpr unsigned char UTF8_4BYTE_SGNT = 0xF0; // 11110uvv

constexpr unsigned char UTF8_CBYTE_CODEPOINT_MASK = 0x3F;
constexpr unsigned char UTF8_2BYTE_CODEPOINT_MASK = 0x1F;
constexpr unsigned char UTF8_3BYTE_CODEPOINT_MASK = 0x0F;
constexpr unsigned char UTF8_4BYTE_CODEPOINT_MASK = 0x07;

static inline int64_t _slovorez_utf8_decoder_char_size(unsigned char c)
{
    if ((c & UTF8_1BYTE_MASK) == UTF8_1BYTE_SGNT) return 1;
    if ((c & UTF8_2BYTE_MASK) == UTF8_2BYTE_SGNT) return 2;
    if ((c & UTF8_3BYTE_MASK) == UTF8_3BYTE_SGNT) return 3;
    if ((c & UTF8_4BYTE_MASK) == UTF8_4BYTE_SGNT) return 4;
    if ((c & UTF8_CBYTE_MASK) == UTF8_CBYTE_SGNT) return 0;
    return -1;
}

static inline void _slovorez_utf8_decoder_char_reset(UTF8Char* utf8c)
{
    memset(utf8c, 0, sizeof(UTF8Char));
}

static inline void _slovorez_utf8_decoder_codepoint_get(UTF8Char* utf8c)
{
    switch (utf8c->size)
    {
        case 1:
        {
            utf8c->codepoint = utf8c->bytes[0];
            break;
        }
        case 2:
        {
            utf8c->codepoint = ((utf8c->bytes[0] & UTF8_2BYTE_CODEPOINT_MASK) << 6)
                              | (utf8c->bytes[1] & UTF8_CBYTE_CODEPOINT_MASK);
            break;
        }
        case 3:
        {
            utf8c->codepoint = ((utf8c->bytes[0] & UTF8_3BYTE_CODEPOINT_MASK) << 12)
                             | ((utf8c->bytes[1] & UTF8_CBYTE_CODEPOINT_MASK) << 6)
                             |  (utf8c->bytes[2] & UTF8_CBYTE_CODEPOINT_MASK);
            break;
        }
        case 4:
        {
            utf8c->codepoint = ((utf8c->bytes[0] & UTF8_4BYTE_CODEPOINT_MASK) << 18)
                             | ((utf8c->bytes[1] & UTF8_CBYTE_CODEPOINT_MASK) << 12)
                             | ((utf8c->bytes[2] & UTF8_CBYTE_CODEPOINT_MASK) << 6)
                             |  (utf8c->bytes[3] & UTF8_CBYTE_CODEPOINT_MASK);
            break;
        }
    }
}

inline bool slovorez_utf8_decoder_char_get(UTF8Char* utf8c, unsigned char c)
{
    const int64_t csize = _slovorez_utf8_decoder_char_size(c);
    if (csize == -1) [[unlikely]]
    {
        _slovorez_utf8_decoder_char_reset(utf8c);
        return false;
    }

    if (csize != 0)
    {
        _slovorez_utf8_decoder_char_reset(utf8c);
        utf8c->size = csize;
    }
    utf8c->bytes[utf8c->curr_size++] = c;

    if (utf8c->size == utf8c->curr_size)
    {
        _slovorez_utf8_decoder_codepoint_get(utf8c);
        return true;
    }
    return false;
}

#endif // SLOVOREZ_UTF8_DECODER_H
