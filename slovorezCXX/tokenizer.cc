#include "tokenizer.h"

static inline void _slovorez_tokenizer_token_merge(TokenizerContext* tctx)
{
    tctx->tokenctx.tokens[tctx->tokenctx.size++] = tctx->lctx.rtoken;
}

static inline void _slovorez_tokenizer_new_token(TokenizerContext* tctx)
{
    tctx->tokenctx.type = tctx->lctx.rtoken.type;
    _slovorez_tokenizer_token_merge(tctx);
}

static inline void _slovorez_tokenizer_token_finalize(TokenizerContext* tctx)
{
    memset(&tctx->rtoken, 0, sizeof(Token));
    tctx->rtoken.type = tctx->tokenctx.type;
    for (int i = 0; i < tctx->tokenctx.size; ++i)
    {
        memcpy(tctx->rtoken.data + tctx->rtoken.size, tctx->tokenctx.tokens[i].data, tctx->tokenctx.tokens[i].size * sizeof(UTF8Char));
        tctx->rtoken.size += tctx->tokenctx.tokens[i].size;
    }
    memset(&tctx->tokenctx, 0, sizeof(TokenContext));
}

static bool _slovorez_tokenizer_token_try_finalize(TokenizerContext* tctx)
{
    switch (tctx->tokenctx.type)
    {
        case TokenType::NOTTKN:
        {
            _slovorez_tokenizer_new_token(tctx);
            return false;
        }
        case TokenType::WRDSPC:
        case TokenType::NWLINE:
        case TokenType::PNCTTN:
        case TokenType::UNKNWN:
        {
            _slovorez_tokenizer_token_finalize(tctx);
            _slovorez_tokenizer_new_token(tctx);
            return true;
        }
        case TokenType::ENWORD:
        case TokenType::NUMBER:
        case TokenType::RUWORD:
        {
            switch (tctx->tokenctx.tokens[tctx->tokenctx.size - 1].type)
            {
                case TokenType::ENWORD:
                case TokenType::NUMBER:
                case TokenType::RUWORD:
                {
                    if (tctx->lctx.rtoken.type == TokenType::PNCTTN && tctx->lctx.rtoken.data[0].bytes[0] == '-')
                    {
                        _slovorez_tokenizer_token_merge(tctx);
                        return false;
                    }
                    _slovorez_tokenizer_token_finalize(tctx);
                    _slovorez_tokenizer_new_token(tctx);
                    return true;
                }
                case TokenType::PNCTTN:
                {
                    switch (tctx->lctx.rtoken.type)
                    {
                        case TokenType::ENWORD:
                        case TokenType::NUMBER:
                        case TokenType::RUWORD:
                        {
                            _slovorez_tokenizer_token_merge(tctx);
                            return false;
                        }
                        default:
                        {
                            _slovorez_tokenizer_token_finalize(tctx);
                            _slovorez_tokenizer_new_token(tctx);
                            return true;
                        }
                    }
                }
                default:
                {
                    _slovorez_tokenizer_token_finalize(tctx);
                    _slovorez_tokenizer_new_token(tctx);
                    return true;
                }
            }
        }
    }
    return false;
}

void slovorez_tokenizer_init(TokenizerContext* tctx)
{
    memset(tctx, 0, sizeof(TokenizerContext));
    slovorez_lexer_init(&tctx->lctx);
}

bool slovorez_tokenizer_token_get(TokenizerContext* tctx, unsigned char c)
{
    if (!slovorez_lexer_token_get(&tctx->lctx, c))
    {
        return false;
    }
    return _slovorez_tokenizer_token_try_finalize(tctx);
}

bool slovorez_tokenizer_end(TokenizerContext* tctx)
{
    slovorez_lexer_token_finalize(&tctx->lctx);
    return _slovorez_tokenizer_token_try_finalize(tctx);
}
