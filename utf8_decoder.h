#ifndef SLOVOREZ_UTF8_DECODER_H
#define SLOVOREZ_UTF8_DECODER_H

#include <cstring>
#include <cstdint>

typedef struct UTF8Char {
    char data[4];
    size_t size;

    inline uint32_t get_binary_num() const
    {
        uint32_t value = 0;
        for (int i = 0; i < this->size; ++i)
        {
            value = (value << 8) | (unsigned char)this->data[i];
        }
        return value;
    }
    inline bool in_range(uint32_t low, uint32_t high) const
    {
        uint32_t value = this->get_binary_num();
        return value >= low && value <= high;
    }

    inline bool operator==(uint32_t other) const
    {
        uint32_t value = this->get_binary_num();
        return value == other;
    }
} UTF8Char;

constexpr unsigned char UTF8_1BYTE_MASK = 0x80;
constexpr unsigned char UTF8_1BYTE_SGNT = 0x00; // 0yyyzzzz
constexpr unsigned char UTF8_2BYTE_MASK = 0xE0;
constexpr unsigned char UTF8_2BYTE_SGNT = 0xC0; // 110xxxyy
constexpr unsigned char UTF8_3BYTE_MASK = 0xF0;
constexpr unsigned char UTF8_3BYTE_SGNT = 0xE0; // 1110wwww
constexpr unsigned char UTF8_4BYTE_MASK = 0xF8;
constexpr unsigned char UTF8_4BYTE_SGNT = 0xF0; // 11110uvv

inline size_t slovorez_utf8_decoder_char_size(unsigned char c)
{
    if ((c & UTF8_1BYTE_MASK) == UTF8_1BYTE_SGNT) return 1;
    if ((c & UTF8_2BYTE_MASK) == UTF8_2BYTE_SGNT) return 2;
    if ((c & UTF8_3BYTE_MASK) == UTF8_3BYTE_SGNT) return 3;
    if ((c & UTF8_4BYTE_MASK) == UTF8_4BYTE_SGNT) return 4;
    return 0;
}

inline void slovorez_utf8_decoder_char_reset(UTF8Char* utf8c)
{
    memset(utf8c, 0, sizeof(UTF8Char));
}

inline bool slovorez_utf8_decoder_char_get(UTF8Char* utf8c, unsigned char c)
{
    utf8c->data[utf8c->size++] = c;
    return (utf8c->size == slovorez_utf8_decoder_char_size(utf8c->data[0]));
}

#endif // SLOVOREZ_UTF8_DECODER_H
