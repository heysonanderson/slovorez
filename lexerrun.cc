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

    fclose(f);
    return 0;
}
