#pragma once

enum ECharset {
    CODES_UNSUPPORTED = -2, // valid but unsupported encoding
    CODES_UNKNOWN = -1,     // invalid or unspecified encoding
    CODES_WIN,              // [ 0] WINDOWS_1251     Windows
    CODES_KOI8,             // [ 1] KOI8_U           Koi8-u
    CODES_ALT,              // [ 2] IBM_866          MS DOS, alternative
    CODES_MAC,              // [ 3] MAC_CYRILLIC     Macintosh
    CODES_MAIN,             // [ 4] ISO_LATIN_CYRILLIC Main
    CODES_ASCII,            // [ 5] WINDOWS_1252     Latin 1
    CODES_RESERVED_3,       // reserved code: use it for new encodings before adding them to the end of the list
    CODES_WIN_EAST,         // [ 7] WINDOWS_1250     WIN PL
    CODES_ISO_EAST,         // [ 8] ISO_8859_2       ISO PL
    // our superset of subset of windows-1251
    CODES_YANDEX,   // [ 9] YANDEX
    CODES_UTF_16BE, // [10] UTF_16BE
    CODES_UTF_16LE, // [11] UTF_16LE
    // missing standard codepages
    CODES_IBM855,       // [12] IBM_855
    CODES_UTF8,         // [13] UTF8
    CODES_UNKNOWNPLANE, // [14] Unrecognized characters are mapped into the PUA: U+F000..U+F0FF

    CODES_KAZWIN,       // [15] WINDOWS_1251_K   Kazakh version of Windows-1251
    CODES_TATWIN,       // [16] WINDOWS_1251_T   Tatarian version of Windows-1251
    CODES_ARMSCII,      // [17] Armenian ASCII
    CODES_GEO_ITA,      // [18] Academy of Sciences Georgian
    CODES_GEO_PS,       // [19] Georgian Parliament
    CODES_ISO_8859_3,   // [20] Latin-3: Turkish, Maltese and Esperanto
    CODES_ISO_8859_4,   // [21] Latin-4: Estonian, Latvian, Lithuanian, Greenlandic, Sami
    CODES_ISO_8859_6,   // [22] Latin/Arabic: Arabic
    CODES_ISO_8859_7,   // [23] Latin/Greek: Greek
    CODES_ISO_8859_8,   // [24] Latin/Hebrew: Hebrew
    CODES_ISO_8859_9,   // [25] Latin-5 or Turkish: Turkish
    CODES_ISO_8859_13,  // [26] Latin-7 or Baltic Rim: Baltic languages
    CODES_ISO_8859_15,  // [27] Latin-9: Western European languages
    CODES_ISO_8859_16,  // [28] Latin-10: South-Eastern European languages
    CODES_WINDOWS_1253, // [29] for Greek
    CODES_WINDOWS_1254, // [30] for Turkish
    CODES_WINDOWS_1255, // [31] for Hebrew
    CODES_WINDOWS_1256, // [32] for Arabic
    CODES_WINDOWS_1257, // [33] for Estonian, Latvian and Lithuanian

    // these codes are all the other 8bit codes known by libiconv
    // they follow in alphanumeric order
    CODES_CP1046,
    CODES_CP1124,
    CODES_CP1125,
    CODES_CP1129,
    CODES_CP1131,
    CODES_CP1133,
    CODES_CP1161, // [40]
    CODES_CP1162,
    CODES_CP1163,
    CODES_CP1258,
    CODES_CP437,
    CODES_CP737,
    CODES_CP775,
    CODES_CP850,
    CODES_CP852,
    CODES_CP853,
    CODES_CP856, // [50]
    CODES_CP857,
    CODES_CP858,
    CODES_CP860,
    CODES_CP861,
    CODES_CP862,
    CODES_CP863,
    CODES_CP864,
    CODES_CP865,
    CODES_CP869,
    CODES_CP874, // [60]
    CODES_CP922,
    CODES_HP_ROMAN8,
    CODES_ISO646_CN,
    CODES_ISO646_JP,
    CODES_ISO8859_10,
    CODES_ISO8859_11,
    CODES_ISO8859_14,
    CODES_JISX0201,
    CODES_KOI8_T,
    CODES_MAC_ARABIC, // [70]
    CODES_MAC_CENTRALEUROPE,
    CODES_MAC_CROATIAN,
    CODES_MAC_GREEK,
    CODES_MAC_HEBREW,
    CODES_MAC_ICELAND,
    CODES_MAC_ROMANIA,
    CODES_MAC_ROMAN,
    CODES_MAC_THAI,
    CODES_MAC_TURKISH,
    CODES_RESERVED_2, // [80] reserved code: use it for new encodings before adding them to the end of the list
    CODES_MULELAO,
    CODES_NEXTSTEP,
    CODES_PT154,
    CODES_RISCOS_LATIN1,
    CODES_RK1048,
    CODES_TCVN,
    CODES_TDS565,
    CODES_TIS620,
    CODES_VISCII,

    // libiconv multibyte codepages
    CODES_BIG5, // [90]
    CODES_BIG5_HKSCS,
    CODES_BIG5_HKSCS_1999,
    CODES_BIG5_HKSCS_2001,
    CODES_CP932,
    CODES_CP936,
    CODES_CP949,
    CODES_CP950,
    CODES_EUC_CN,
    CODES_EUC_JP,
    CODES_EUC_KR, // [100]
    CODES_EUC_TW,
    CODES_GB18030,
    CODES_GBK,
    CODES_HZ,
    CODES_ISO_2022_CN,
    CODES_ISO_2022_CN_EXT,
    CODES_ISO_2022_JP,
    CODES_ISO_2022_JP_1,
    CODES_ISO_2022_JP_2,
    CODES_ISO_2022_KR, // [110]
    CODES_JOHAB,
    CODES_SHIFT_JIS,

    CODES_MAX
};
