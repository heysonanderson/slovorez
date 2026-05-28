#ifndef SLOVOREZ_UNICODE_TRIE_H
#define SLOVOREZ_UNICODE_TRIE_H

#include <cmath>

#include "utf8_decoder.h"
#include "token.h"

constexpr size_t UNICODE_CAPACITY       = 0x110000;
constexpr size_t TRIE_BLOCK_SIZE_POW2   = 6;
constexpr size_t TRIE_BLOCK_SIZE        = std::pow(2, TRIE_BLOCK_SIZE_POW2);
constexpr size_t TRIE_BLOCK_COUNT       = 256;
constexpr size_t TRIE_TOTAL_BLOCKS      = UNICODE_CAPACITY >> TRIE_BLOCK_SIZE_POW2;

struct UnicodeTrie {
    uint16_t pointers[TRIE_TOTAL_BLOCKS];
    uint8_t blocks[TRIE_BLOCK_COUNT][TRIE_BLOCK_SIZE];
    size_t blocks_count = 0;

    TokenType lookup(uint32_t codepoint) const
    {
        uint8_t ttidx = blocks[pointers[codepoint >> TRIE_BLOCK_SIZE_POW2]][codepoint & (TRIE_BLOCK_SIZE - 1)];
        return slovorez_ttidx_to_tt[ttidx];
    }

    static const UnicodeTrie& get();

private:
    explicit UnicodeTrie();
};

#endif // SLOVOREZ_UNICODE_TRIE_H
