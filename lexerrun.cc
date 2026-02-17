#include <cstdio>
#include <clocale>
#include "text_lexer.h"

void print_tokens(const std::vector<Token>& tokens)
{
    for (int i = 0; i < tokens.size(); ++i)
    {
        fprintf(
            stdout,
            "%s SIZE=%-3d %s [ ",
            TokenTypeStr[static_cast<int>(tokens[i].type)],
            tokens[i].size,
            tokens[i].data
        );
        for (int j = 0; j < tokens[i].size; ++j)
        {
            fprintf(stdout, "'%c", tokens[i].data[j]);
            fprintf(stdout, "' ");
        }
        fprintf(stdout, "]\n");
    }
}

int main(void)
{
    setlocale(LC_ALL, "");

    FILE *f = nullptr;
    f = fopen("../text.txt", "r"); // Temp. hardcoded file - change later
    if (f == nullptr)
    {
        fprintf(stderr, "Could not open text.txt file\n");
        return 1;
    }

    LexerContext lctx;
    slovorez_lexer_init(&lctx);
    int c;
    while ((c = fgetc(f)) != EOF)
    {
        slovorez_lexer_token_get(&lctx, (unsigned char)c);
    }
    fclose(f);
    return 0;
}
