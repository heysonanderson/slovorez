#include <cstdio>
#include <clocale>
#include "text_lexer.h"

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
        slovorez_lexer_char(&lctx, (unsigned char)c);
    }
    // OUTPUT
    for (int i = 0; i < lctx.tokens.size(); ++i)
    {
        fprintf(
            stdout,
            "%s SIZE=%-3d %s [ ",
            TokenTypeStr[static_cast<int>(lctx.tokens[i].type)],
            lctx.tokens[i].size,
            lctx.tokens[i].data
        );
        for (int j = 0; j < lctx.tokens[i].size; ++j)
        {
            fprintf(stdout, "'%c", lctx.tokens[i].data[j]);
            if ((lctx.tokens[i].data[j] & FIRST_3BIT_MASK) == UTF8_2BYTE_SGNT)
            {
                fprintf(stdout, "%c", lctx.tokens[i].data[++j]);
            }
            fprintf(stdout, "' ");
        }
        fprintf(stdout, "]\n");
    }
    fclose(f);
    return 0;
}
