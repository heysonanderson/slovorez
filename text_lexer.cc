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
    lctx->last_token = (Token*)malloc(sizeof(Token));
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

    if (lctx->bytes_expected == 1)
    {
        switch (lctx->buffer[0])
        {
            case ' ':
            {
                if ((*lctx->last_token).size != 0)
                {
                    lctx->tokens.push_back(std::move(*lctx->last_token));
                    memset(lctx->last_token, 0, sizeof(Token));
                }
                break;
            }
        }
    }
    else if (lctx->bytes_expected == 2)
    {
        memcpy((*lctx->last_token).data + (*lctx->last_token).size, &lctx->buffer, lctx->bytes_gathered);
        (*lctx->last_token).size += lctx->bytes_gathered;
        (*lctx->last_token).type = TokenType::RUWORD;
    }
    _slovorez_lexer_reset_ctx(lctx);
}

void slovorez_lexer_finish(LexerContext* lctx)
{
    lctx->tokens.push_back(std::move(*lctx->last_token));
    free(lctx->last_token);
}
