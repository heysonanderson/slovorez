#include "text_lexer.h"

void slovorez_lexer_init(LexerContext* lctx, size_t token_num)
{
    lctx->tokens.reserve(token_num);
    memset(&lctx->ctxtoken, 0, sizeof(Token));
    slovorez_utf8_decoder_char_reset(&lctx->utf8c);
}

bool slovorez_lexer_token_get(LexerContext* lctx, unsigned char c)
{
    if (!slovorez_utf8_decoder_char_get(&lctx->utf8c, c))
    {
        return false;
    }

    fprintf(stdout , "'%.*s' ", lctx->utf8c.size, lctx->utf8c.data);

    // LEXING

    slovorez_utf8_decoder_char_reset(&lctx->utf8c);
    return true;
}
