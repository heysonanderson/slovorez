#include "text_lexer.h"

bool _slovorez_lexer_is_enletter(unsigned char c)
{
    return (c >= ASCII_LETTER_A && c <= ASCII_LETTER_Z) || (c >= ASCII_LETTER_a && c <= ASCII_LETTER_z);
}

bool _slovorez_lexer_is_number(unsigned char c)
{
    return c >= ASCII_NUMBER_0 && c <= ASCII_NUMBER_9;
}

bool _slovorez_lexer_is_ruletter(const UTF8Char& utf8c)
{
    return utf8c.in_range(UTF8_RU_LETTERS_RANGE_START, UTF8_RU_LETTERS_RANGE_END) || utf8c == UTF8_RU_UPPERCASE_LETTER_IO || utf8c == UTF8_RU_LOWERCASE_LETTER_IO;
}

bool _slovorez_lexer_is_extended_punctuation(const UTF8Char& utf8c)
{
    return utf8c == UTF8_NUMERO_SIGN || utf8c == UTF8_LEFT_ANGLE_QUOTATION || utf8c == UTF8_RIGHT_ANGLE_QUOTATION || utf8c == UTF8_MIDDLE_DOT || utf8c.in_range(UTF8_3BYTE_PNCTTN_RANGE_START, UTF8_3BYTE_PNCTTN_RANGE_END);
}

void _slovorez_lexer_token_insert_utf8_char(Token* token, UTF8Char* utf8c)
{
    memcpy(&token->data[token->size++], utf8c, sizeof(UTF8Char));
}

void _slovorez_lexer_token_finalize(LexerContext* lctx)
{
    lctx->tokens.push_back(std::move(lctx->ctxtoken));
    memset(&lctx->ctxtoken, 0, sizeof(Token));
}

void _slovorez_lexer_new_token(LexerContext* lctx)
{
    switch (lctx->utf8c.size)
    {
        case 1:
        {
            if (_slovorez_lexer_is_number(lctx->utf8c.data[0]))
            {
                lctx->ctxtoken.type = TokenType::NUMBER;
            }
            else if (_slovorez_lexer_is_enletter(lctx->utf8c.data[0]))
            {
                lctx->ctxtoken.type = TokenType::ENWORD;
            }
            else
            {
                switch(lctx->utf8c.data[0])
                {
                    case ' ':
                    {
                        lctx->ctxtoken.type = TokenType::WRDSPC; 
                        break;
                    }
                    case '\n':
                    {
                        lctx->ctxtoken.type = TokenType::NWLINE; 
                        break;
                    }
                    default:
                    {
                        lctx->ctxtoken.type = TokenType::PNCTTN;
                    }
                }
            }
            _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
            break;
        }
        case 2:
        {
            if (_slovorez_lexer_is_extended_punctuation(lctx->utf8c))
            {
                lctx->ctxtoken.type = TokenType::PNCTTN;
            }
            else if (lctx->utf8c == UTF8_NO_BREAK_SPACE)
            {
                lctx->ctxtoken.type = TokenType::WRDSPC;
            }
            else
            {
                lctx->ctxtoken.type = TokenType::RUWORD;
            }
            _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
            break;
        }
        case 3:
        {
            if (_slovorez_lexer_is_extended_punctuation(lctx->utf8c))
            {
                lctx->ctxtoken.type = TokenType::PNCTTN;
            }
            else
            {
                lctx->ctxtoken.type = TokenType::UNKNWN;
            }
            _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
            break;
        }
        case 4:
        {
            lctx->ctxtoken.type = TokenType::UNKNWN;
            _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
            break;
        }
    }
}

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

    bool token_ready = false;
    switch (lctx->ctxtoken.type)
    {
        case TokenType::NOTTKN:
        {
            _slovorez_lexer_new_token(lctx);
            break;
        }
        case TokenType::RUWORD:
        {
            switch (lctx->utf8c.size)
            {
                case 2:
                {
                    _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
                    break;
                }
                default:
                {
                    _slovorez_lexer_token_finalize(lctx);
                    token_ready = true;
                    _slovorez_lexer_new_token(lctx);
                    break;
                }
            }
            break;
        }
        case TokenType::PNCTTN:
        case TokenType::WRDSPC:
        case TokenType::NWLINE:
        {
            _slovorez_lexer_token_finalize(lctx);
            token_ready = true;
            _slovorez_lexer_new_token(lctx);
            break;
        }
        case TokenType::NUMBER:
        {
            if (lctx->utf8c.size == 1 && _slovorez_lexer_is_number(lctx->utf8c.data[0]))
            {
                _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
            }
            else
            {
                _slovorez_lexer_token_finalize(lctx);
                token_ready = true;
                _slovorez_lexer_new_token(lctx);
            }
            break;
        }
        case TokenType::ENWORD:
        {
            if (lctx->utf8c.size == 1 && _slovorez_lexer_is_enletter(lctx->utf8c.data[0]))
            {
                _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
            }
            else
            {
                _slovorez_lexer_token_finalize(lctx);
                token_ready = true;
                _slovorez_lexer_new_token(lctx);
            }
            break;
        }
        case TokenType::UNKNWN:
        {
            _slovorez_lexer_token_finalize(lctx);
            token_ready = true;
            _slovorez_lexer_new_token(lctx);
            break;
        }
    }
    slovorez_utf8_decoder_char_reset(&lctx->utf8c);
    return token_ready & !lctx->tokens.empty();
}
