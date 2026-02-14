#include "text_lexer.h"

void _slovorez_lexer_reset_ctx(LexerContext* lctx)
{
    memset(lctx->buffer, 0, 4);
    lctx->bytes_expected = 1;
    lctx->bytes_gathered = 0;
}

void slovorez_lexer_init(LexerContext* lctx, size_t token_num)
{
    lctx->tokens.reserve(token_num);
    _slovorez_lexer_reset_ctx(lctx);
}

void slovorez_lexer_char(LexerContext* lctx, unsigned char c)
{
    if (c < ASCII_END)
    {
        lctx->buffer[0] = c;
        lctx->bytes_gathered = 1;
    }
    else if ((c & FIRST_3BIT_MASK) == UTF8_2BYTE_SGNT)
    {
        lctx->buffer[0] = c;
        lctx->bytes_expected = 2;
        lctx->bytes_gathered = 1;
    }
    else if (c >= ASCII_END)
    {
        if (lctx->bytes_expected != lctx->bytes_gathered)
        {
            lctx->buffer[lctx->bytes_gathered++] = c;
        }
    }
    if (lctx->bytes_expected != lctx->bytes_gathered)
    {
        return;
    }

    fprintf(stdout, "'%s'", lctx->buffer); // DEBUG OUTPUT - REMOVE LATER

    // Lexing

    _slovorez_lexer_reset_ctx(lctx);
}
