#include "text_lexer.h"

void _slovorez_lexer_reset_ctx(LexerContext* lctx)
{
    memset(lctx->buffer, 0, 4);
    lctx->bytes_expected = 1;
    lctx->bytes_gathered = 0;
}

bool _slovorez_lexer_is_enword(unsigned char c)
{
    return (c >= ASCII_LETTER_A && c <= ASCII_LETTER_Z) || (c >= ASCII_LETTER_a && c <= ASCII_LETTER_z);
}

bool _slovorez_lexer_is_number(unsigned char c)
{
    return c >= ASCII_NUMBER_0 && c <= ASCII_NUMBER_9;
}

void _slovorez_lexer_new_token(LexerContext* lctx)
{
    if (lctx->bytes_expected == 1)
    {
        if (_slovorez_lexer_is_enword(lctx->buffer[0]))
        {
            lctx->last_token.type = TokenType::ENWORD;
        }
        else if (_slovorez_lexer_is_number(lctx->buffer[0]))
        {
            lctx->last_token.type = TokenType::NUMBER;
        }
        else
        {
            lctx->last_token.type = TokenType::PNCTTN;
        }
        _slovorez_lexer_insert_1byte(&lctx->last_token, lctx->buffer[0]);
    }
    else if (lctx->bytes_expected == 2)
    {
        lctx->last_token.type = TokenType::RUWORD;
        _slovorez_lexer_insert_2bytes(&lctx->last_token, lctx->buffer);
    }
}

void _slovorez_lexer_insert_1byte(Token* token, unsigned char c)
{
    memcpy(token->data + token->size, &c, 1);
    token->size += 1;
}

void _slovorez_lexer_insert_2bytes(Token* token, unsigned char* buffer)
{
    memcpy(token->data + token->size, buffer, 2);
    token->size += 2;
}

void _slovorez_lexer_push_token(LexerContext* lctx)
{
    lctx->tokens.push_back(std::move(lctx->last_token));
    memset(&lctx->last_token, 0, sizeof(Token));
}

void slovorez_lexer_init(LexerContext* lctx, size_t token_num)
{
    lctx->tokens.reserve(token_num);
    memset(&lctx->last_token, 0, sizeof(Token));
    _slovorez_lexer_reset_ctx(lctx);
}

void slovorez_lexer_char(LexerContext* lctx, unsigned char c)
{
    if (c == ' ' || c == '\n' || c == '\0')
    {
        _slovorez_lexer_push_token(lctx);
        return;
    }

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
    switch (lctx->last_token.type)
    {
        case TokenType::RUWORD:
        {
            if (lctx->bytes_gathered == 2)
            {
                _slovorez_lexer_insert_2bytes(&lctx->last_token, lctx->buffer);
            }
            else
            {
                _slovorez_lexer_push_token(lctx);
                _slovorez_lexer_new_token(lctx);
            }
            break;
        }
        case TokenType::ENWORD:
        {
            if (lctx->bytes_gathered == 1)
            {
                // WEAK HYPHEN CHECK: 'multi--------color' is possible
                if (_slovorez_lexer_is_enword(lctx->buffer[0]) || lctx->buffer[0] == '-')
                {
                    _slovorez_lexer_insert_1byte(&lctx->last_token, lctx->buffer[0]);
                }
                else
                {
                    _slovorez_lexer_push_token(lctx);
                    _slovorez_lexer_new_token(lctx);
                }
            }
            else
            {
                lctx->last_token.type = TokenType::MLWORD;
                _slovorez_lexer_insert_2bytes(&lctx->last_token, lctx->buffer);
            }
            break;
        }
        case TokenType::NUMBER:
        {
            if(lctx->bytes_gathered == 1 && _slovorez_lexer_is_number(lctx->buffer[0]))
            {
                _slovorez_lexer_insert_1byte(&lctx->last_token, lctx->buffer[0]);
            }
            else
            {
                _slovorez_lexer_push_token(lctx);
                _slovorez_lexer_new_token(lctx);
            }
            break;
        }
        case TokenType::PNCTTN:
        {
            if (lctx->bytes_gathered == 1 && !_slovorez_lexer_is_enword(lctx->buffer[0]) && !_slovorez_lexer_is_number(lctx->buffer[0]))
            {
                _slovorez_lexer_insert_1byte(&lctx->last_token, lctx->buffer[0]);
            }
            else
            {
                _slovorez_lexer_push_token(lctx);
                _slovorez_lexer_new_token(lctx);
            }
            break;
        }
        case TokenType::MLWORD:
        {
            if (lctx->bytes_gathered == 1)
            {
                // WEAK HYPHEN CHECK AS WELL
                // WHAT ABOUT NUMBERS?
                if (_slovorez_lexer_is_enword(lctx->buffer[0]) || lctx->buffer[0] == '-')
                {
                    _slovorez_lexer_insert_1byte(&lctx->last_token, lctx->buffer[0]);
                }
                else
                {
                    _slovorez_lexer_push_token(lctx);
                    _slovorez_lexer_new_token(lctx);
                }
            }
            else if (lctx->bytes_gathered == 2)
            {
                _slovorez_lexer_insert_2bytes(&lctx->last_token, lctx->buffer);
            }
            break;
        }
        case TokenType::NOTTKN:
        {
            _slovorez_lexer_new_token(lctx);
            break;
        }
    }
    _slovorez_lexer_reset_ctx(lctx);
}
