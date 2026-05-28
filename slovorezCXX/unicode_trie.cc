#include "unicode_trie.h"

static void _slovorez_unicode_trie_flat_fill(uint8_t* flat)
{
    /* ==========================================
            Unicode Basic Latin Block (ASCII)
     ========================================== */

    /* -- NWLINE / WRDSPC --------------------- */

    flat[0x000A] = NWLINE_IDX; // U+000A        Control character: Line Feed (lf)
    flat[0x000B] = WRDSPC_IDX; // U+000B        Control character: Line Tabulation
    flat[0x0020] = WRDSPC_IDX; // U+0020	    Space

    /* -- PCNTTN U+0021 - U+002F -------------- *
     *
     * U+0021   !   Exclamation Mark
     * U+0022   "   Quotation Mark
     * U+0023   #   Number Sign
     * U+0024   $   Dollar Sign
     * U+0025   %   Percent Sign
     * U+0026   &   Ampersand
     * U+0027   '   Apostrophe
     * U+0028   (   Left Parenthesis
     * U+0029   )   Right Parenthesis
     * U+002A   *   Asterisk
     * U+002B   +   Plus Sign
     * U+002C   ,   Comma
     * U+002D   -   Hyphen-minus
     * U+002E   .   Full Stop
     * U+002F   /   Solidus
     */
    for (uint32_t codepoint = 0x0021; codepoint <= 0x002F; codepoint++)
    {
        flat[codepoint] = PNCTTN_IDX;
    }

    /* -- NUMBER U+0030 - U+0039 -------------- *
     *
     * U+0030   0   Digit Zero
     * U+0031   1   Digit One
     * U+0032   2   Digit Two
     * U+0033   3   Digit Three
     * U+0034   4   Digit Four
     * U+0035   5   Digit Five
     * U+0036   6   Digit Six
     * U+0037   7   Digit Seven
     * U+0038   8   Digit Eight
     * U+0039   9   Digit Nine
     */
    for (uint32_t codepoint = 0x0030; codepoint <= 0x0039; codepoint++)
    {
        flat[codepoint] = NUMBER_IDX;
    }

    /* -- PNCTTN U+003A - U+0040 -------------- *
     *
     * U+003A   :   Colon
     * U+003B   ;   Semicolon
     * U+003C   <   Less-than Sign
     * U+003D   =   Equals Sign
     * U+003E   >   Greater-than Sign
     * U+003F   ?   Question Mark
     * U+0040   @   Commercial At
     */
    for (uint32_t codepoint = 0x003A; codepoint <= 0x0040; codepoint++)
    {
        flat[codepoint] = PNCTTN_IDX;
    }

    /* -- ENWORD U+0041 - U+005A -------------- *
     *
     * U+0041   A   Latin Capital Letter A
     * U+0042   B   Latin Capital Letter B
     * U+0043   C   Latin Capital Letter C
     * U+0044   D   Latin Capital Letter D
     * U+0045   E   Latin Capital Letter E
     * U+0046   F   Latin Capital Letter F
     * U+0047   G   Latin Capital Letter G
     * U+0048   H   Latin Capital Letter H
     * U+0049   I   Latin Capital Letter I
     * U+004A   J   Latin Capital Letter J
     * U+004B   K   Latin Capital Letter K
     * U+004C   L   Latin Capital Letter L
     * U+004D   M   Latin Capital Letter M
     * U+004E   N   Latin Capital Letter N
     * U+004F   O   Latin Capital Letter O
     * U+0050   P   Latin Capital Letter P
     * U+0051   Q   Latin Capital Letter Q
     * U+0052   R   Latin Capital Letter R
     * U+0053   S   Latin Capital Letter S
     * U+0054   T   Latin Capital Letter T
     * U+0055   U   Latin Capital Letter U
     * U+0056   V   Latin Capital Letter V
     * U+0057   W   Latin Capital Letter W
     * U+0058   X   Latin Capital Letter X
     * U+0059   Y   Latin Capital Letter Y
     * U+005A   Z   Latin Capital Letter Z
     */
    for (uint32_t codepoint = 0x0041; codepoint <= 0x005A; codepoint++)
    {
        flat[codepoint] = ENWORD_IDX;
    }

    /* -- PNCTTN U+005B - U+0060 -------------- *
     *
     * U+005B   [   Left Square Bracket
     * U+005C   \   Reverse Solidus
     * U+005D   ]   Right Square Bracket
     * U+005E   ^   Circumflex Accent
     * U+005F   _   Low Line
     * U+0060   `   Grave Accent
     */
    for (uint32_t codepoint = 0x005B; codepoint <= 0x0060; codepoint++)
    {
        flat[codepoint] = PNCTTN_IDX;
    }

    /* -- ENWORD U+0061 - U+007A -------------- *
     *
     * U+0061   a   Latin Small Letter A
     * U+0062   b   Latin Small Letter B
     * U+0063   c   Latin Small Letter C
     * U+0064   d   Latin Small Letter D
     * U+0065   e   Latin Small Letter E
     * U+0066   f   Latin Small Letter F
     * U+0067   g   Latin Small Letter G
     * U+0068   h   Latin Small Letter H
     * U+0069   i   Latin Small Letter I
     * U+006A   j   Latin Small Letter J
     * U+006B   k   Latin Small Letter K
     * U+006C   l   Latin Small Letter L
     * U+006D   m   Latin Small Letter M
     * U+006E   n   Latin Small Letter N
     * U+006F   o   Latin Small Letter O
     * U+0070   p   Latin Small Letter P
     * U+0071   q   Latin Small Letter Q
     * U+0072   r   Latin Small Letter R
     * U+0073   s   Latin Small Letter S
     * U+0074   t   Latin Small Letter T
     * U+0075   u   Latin Small Letter U
     * U+0076   v   Latin Small Letter V
     * U+0077   w   Latin Small Letter W
     * U+0078   x   Latin Small Letter X
     * U+0079   y   Latin Small Letter Y
     * U+007A   z   Latin Small Letter Z
     */
    for (uint32_t codepoint = 0x0061; codepoint <= 0x007A; codepoint++)
    {
        flat[codepoint] = ENWORD_IDX;
    }

    /* -- PNCTTN U+007B - U+007E -------------- *
     *
     * U+007B   {   Left Curly Bracket
     * U+007C   |   Vertical Line
     * U+007D   }   Right Curly Bracket
     * U+007E   ~   Tilde
     */
    for (uint32_t codepoint = 0x007B; codepoint <= 0x007E; codepoint++)
    {
        flat[codepoint] = PNCTTN_IDX;
    }

    /* ==========================================
            Unicode Latin-1 Supplement Block
     ========================================== */

    /* -- WRDSPC / PNCTTN --------------------- */

    flat[0x00A0] = WRDSPC_IDX; // U+00A0        No-break Space
    flat[0x00AB] = PNCTTN_IDX; // U+00AB    «   Left-pointing Double Angle Quotation Mark
    flat[0x00B7] = PNCTTN_IDX; // U+00B7    ·   Middle Dot
    flat[0x00BB] = PNCTTN_IDX; // U+00BB    »   Right-pointing Double Angle Quotation Mark

    /* ==========================================
                Unicode Cyrillic Block
     ========================================== */

    /* -- RUWORD ------------------------------ */

    flat[0x0401] = RUWORD_IDX; // U+0401  Ё   Cyrillic Capital Letter Io

    /* -- RUWORD U+0410 - U+044F -------------- *
     *
     * U+0410   А   Cyrillic Capital Letter A
     * U+0411   Б   Cyrillic Capital Letter Be
     * U+0412   В   Cyrillic Capital Letter Ve
     * U+0413   Г   Cyrillic Capital Letter Ghe
     * U+0414   Д   Cyrillic Capital Letter De
     * U+0415   Е   Cyrillic Capital Letter Ie
     * U+0416   Ж   Cyrillic Capital Letter Zhe
     * U+0417   З   Cyrillic Capital Letter Ze
     * U+0418   И   Cyrillic Capital Letter I
     * U+0419   Й   Cyrillic Capital Letter Short I
     * U+041A   К   Cyrillic Capital Letter Ka
     * U+041B   Л   Cyrillic Capital Letter El
     * U+041C   М   Cyrillic Capital Letter Em
     * U+041D   Н   Cyrillic Capital Letter En
     * U+041E   О   Cyrillic Capital Letter O
     * U+041F   П   Cyrillic Capital Letter Pe
     * U+0420   Р   Cyrillic Capital Letter Er
     * U+0421   С   Cyrillic Capital Letter Es
     * U+0422   Т   Cyrillic Capital Letter Te
     * U+0423   У   Cyrillic Capital Letter U
     * U+0424   Ф   Cyrillic Capital Letter Ef
     * U+0425   Х   Cyrillic Capital Letter Ha
     * U+0426   Ц   Cyrillic Capital Letter Tse
     * U+0427   Ч   Cyrillic Capital Letter Che
     * U+0428   Ш   Cyrillic Capital Letter Sha
     * U+0429   Щ   Cyrillic Capital Letter Shcha
     * U+042A   Ъ   Cyrillic Capital Letter Hard Sign
     * U+042B   Ы   Cyrillic Capital Letter Yeru
     * U+042C   Ь   Cyrillic Capital Letter Soft Sign
     * U+042D   Э   Cyrillic Capital Letter E
     * U+042E   Ю   Cyrillic Capital Letter Yu
     * U+042F   Я   Cyrillic Capital Letter Ya
     * U+0430   а   Cyrillic Small Letter A
     * U+0431   б   Cyrillic Small Letter Be
     * U+0432   в   Cyrillic Small Letter Ve
     * U+0433   г   Cyrillic Small Letter Ghe
     * U+0434   д   Cyrillic Small Letter De
     * U+0435   е   Cyrillic Small Letter Ie
     * U+0436   ж   Cyrillic Small Letter Zhe
     * U+0437   з   Cyrillic Small Letter Ze
     * U+0438   и   Cyrillic Small Letter I
     * U+0439   й   Cyrillic Small Letter Short I
     * U+043A   к   Cyrillic Small Letter Ka
     * U+043B   л   Cyrillic Small Letter El
     * U+043C   м   Cyrillic Small Letter Em
     * U+043D   н   Cyrillic Small Letter En
     * U+043E   о   Cyrillic Small Letter O
     * U+043F   п   Cyrillic Small Letter Pe
     * U+0440   р   Cyrillic Small Letter Er
     * U+0441   с   Cyrillic Small Letter Es
     * U+0442   т   Cyrillic Small Letter Te
     * U+0443   у   Cyrillic Small Letter U
     * U+0444   ф   Cyrillic Small Letter Ef
     * U+0445   х   Cyrillic Small Letter Ha
     * U+0446   ц   Cyrillic Small Letter Tse
     * U+0447   ч   Cyrillic Small Letter Che
     * U+0448   ш   Cyrillic Small Letter Sha
     * U+0449   щ   Cyrillic Small Letter Shcha
     * U+044A   ъ   Cyrillic Small Letter Hard Sign
     * U+044B   ы   Cyrillic Small Letter Yeru
     * U+044C   ь   Cyrillic Small Letter Soft Sign
     * U+044D   э   Cyrillic Small Letter E
     * U+044E   ю   Cyrillic Small Letter Yu
     * U+044F   я   Cyrillic Small Letter Ya
     */
    for (uint32_t codepoint = 0x0410; codepoint <= 0x044F; codepoint++)
    {
        flat[codepoint] = RUWORD_IDX;
    }

    /* -- RUWORD ------------------------------ */

    flat[0x0451] = RUWORD_IDX; // U+0451  ё   Cyrillic Small Letter Io

    /* ==========================================
            Unicode General Punctuation Block
     ========================================== */

    /* -- PNCTTN U+2010 - U+201F -------------- *
     *
     * U+2010   ‐   Hyphen
     * U+2011   ‑   Non-breaking Hyphen
     * U+2012   ‒   Figure Dash
     * U+2013   –   En Dash
     * U+2014   —   Em Dash
     * U+2015   ―   Horizontal Bar
     * U+2016   ‖   Double Vertical Line
     * U+2017   ‗   Double Low Line
     * U+2018   ‘   Left Single Quotation Mark
     * U+2019   ’   Right Single Quotation Mark
     * U+201A   ‚   Single Low-9 Quotation Mark
     * U+201B   ‛   Single High-reversed-9 Quotation Mark
     * U+201C   “   Left Double Quotation Mark
     * U+201D   ”   Right Double Quotation Mark
     * U+201E   „   Double Low-9 Quotation Mark
     * U+201F   ‟   Double High-reversed-9 Quotation Mark
     */
    for (uint32_t codepoint = 0x2010; codepoint <= 0x201F; codepoint++)
    {
        flat[codepoint] = PNCTTN_IDX;
    }

    /* -- PNCTTN U+2024 - U+2027 -------------- *
     *
     * U+2024   ․   One Dot Leader
     * U+2025   ‥   Two Dot Leader
     * U+2026   …   Horizontal Ellipsis
     * U+2027   ‧   Hyphenation Point
     */
    for (uint32_t codepoint = 0x2024; codepoint <= 0x2027; codepoint++)
    {
        flat[codepoint] = PNCTTN_IDX;
    }

    /* -- PNCTTN ------------------------------ */

    flat[0x2116] = PNCTTN_IDX; // U+2116  №   Numero Sign
}

static void _slovorez_unicode_trie_make(UnicodeTrie* trie, uint8_t* flat)
{
    for (int block = 0; block < TRIE_TOTAL_BLOCKS; ++block)
    {
        uint8_t* chunk = flat + block * TRIE_BLOCK_SIZE;

        int found = -1;
        for (int u = 0; u < trie->blocks_count; ++u)
        {
            if (memcmp(chunk, trie->blocks[u], TRIE_BLOCK_SIZE) == 0)
            {
                found = u;
                break;
            }
        }

        if (found != -1)
        {
            trie->pointers[block] = found;
        }
        else
        {
            memcpy(trie->blocks[trie->blocks_count], chunk, TRIE_BLOCK_SIZE);
            trie->pointers[block] = trie->blocks_count;
            trie->blocks_count += 1;
        }
    }
}

UnicodeTrie::UnicodeTrie()
{
    uint8_t flat[UNICODE_CAPACITY] = {};
    memset(flat, UNKNWN_IDX, sizeof(flat));
    _slovorez_unicode_trie_flat_fill(flat);
    _slovorez_unicode_trie_make(this, flat);
}

const UnicodeTrie& UnicodeTrie::get()
{
    static UnicodeTrie trie;
    return trie;
}
