#include "lexer.h"

static inline void _slovorez_lexer_token_insert_utf8_char(Token* token, UTF8Char* utf8c)
{
    memcpy(&token->data[token->size++], utf8c, sizeof(UTF8Char));
}

static inline void _slovorez_lexer_new_token(LexerContext* lctx)
{
    lctx->ctxtoken.type = UnicodeTrie::get().lookup(lctx->utf8c.codepoint);
    _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
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
            const TokenType utf8ct = UnicodeTrie::get().lookup(lctx->utf8c.codepoint);
            if (lctx->ctxtoken.type == utf8ct)
            {
                _slovorez_lexer_token_insert_utf8_char(&lctx->ctxtoken, &lctx->utf8c);
                return false;
            }
            slovorez_lexer_token_finalize(lctx);
            _slovorez_lexer_new_token(lctx);
            return true;
        }
        case TokenType::PNCTTN:
        case TokenType::WRDSPC:
        case TokenType::NWLINE:
        case TokenType::UNKNWN:
        {
            slovorez_lexer_token_finalize(lctx);
            _slovorez_lexer_new_token(lctx);
            return true;
        }
    }
    return false;
}

void slovorez_lexer_init(LexerContext* lctx)
{
    memset(&lctx->rtoken, 0, sizeof(Token));
    memset(&lctx->ctxtoken, 0, sizeof(Token));
}

bool slovorez_lexer_token_get(LexerContext* lctx, unsigned char c)
{
    if (!slovorez_utf8_decoder_char_get(&lctx->utf8c, c))
    {
        return false;
    }
    return _slovorez_lexer_token_try_finalize(lctx);
}

void slovorez_lexer_token_finalize(LexerContext* lctx)
{
    lctx->rtoken = lctx->ctxtoken;
    memset(&lctx->ctxtoken, 0, sizeof(Token));
}
