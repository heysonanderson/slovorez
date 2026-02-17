#include "text_lexer.h"

static inline void _slovorez_lexer_token_insert_utf8_char(Token* token, UTF8Char* utf8c)
{
    memcpy(&token->data[token->size++], utf8c, sizeof(UTF8Char));
}

static inline void _slovorez_lexer_new_token(LexerContext* lctx)
{
    lctx->ctxtoken.type = slovorez_get_utf8_tt(lctx->utf8c);
    _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
}

static inline void _slovorez_lexer_token_finalize(LexerContext* lctx)
{
    lctx->tokens.push_back(std::move(lctx->ctxtoken));
    memset(&lctx->ctxtoken, 0, sizeof(Token));
}

static bool _slovorez_lexer_token_try_finalize(LexerContext* lctx)
{
    switch (lctx->ctxtoken.type)
    {
        case TokenType::NOTTKN:
        {
            _slovorez_lexer_new_token(lctx);
            return false;
        }
        case TokenType::ENWORD:
        case TokenType::NUMBER:
        case TokenType::RUWORD:
        {
            TokenType utf8_tt = slovorez_get_utf8_tt(lctx->utf8c);
            if (lctx->ctxtoken.type == utf8_tt)
            {
                _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
                return false;
            }
            _slovorez_lexer_token_finalize(lctx);
            _slovorez_lexer_new_token(lctx);
            return true;
        }
        case TokenType::PNCTTN:
        case TokenType::WRDSPC:
        case TokenType::NWLINE:
        case TokenType::UNKNWN:
        {
            _slovorez_lexer_token_finalize(lctx);
            _slovorez_lexer_new_token(lctx);
            return true;
        }
    }
    return false;
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
    bool token_ready = _slovorez_lexer_token_try_finalize(lctx) && !lctx->tokens.empty();
    slovorez_utf8_decoder_char_reset(&lctx->utf8c);
    return token_ready;
}
