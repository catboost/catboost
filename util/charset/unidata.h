#pragma once

#include "unicode_table.h"

#include <util/system/defaults.h> // wchar32, ui64, ULL()

enum WC_TYPE {        // TODO move no NUnicode
    Lu_UPPER = 1,     // 'Ъ'
    Ll_LOWER = 2,     // 'ъ'
    Lt_TITLE = 3,     // 'Ъ'
    Lm_EXTENDER = 4,  // '-'
    Lm_LETTER = 5,    // 'ъ'
    Lo_OTHER = 6,     // '?'
    Lo_IDEOGRAPH = 7, // '?'
    Lo_KATAKANA = 8,  // '?'
    Lo_HIRAGANA = 9,  // '?'
    Lo_LEADING = 10,  // '?'
    Lo_VOWEL = 11,    // '?'
    Lo_TRAILING = 12, // '?'

    Mn_NONSPACING = 13, // '`'
    Me_ENCLOSING = 14,  // '`'
    Mc_SPACING = 15,    // '`'

    Nd_DIGIT = 16,     // '9'           // convert to digit
    Nl_LETTER = 17,    // 'X'           // X,V,C,L,I ...
    Nl_IDEOGRAPH = 18, // '?'
    No_OTHER = 19,     // '9'

    Zs_SPACE = 20,     // ' ' [\40\240] SPACE ... NO-BREAK SPACE (00A0)
    Zs_ZWSPACE = 21,   // ' '           // nothing ?
    Zl_LINE = 22,      // '\n'
    Zp_PARAGRAPH = 23, // '\n'

    Cc_ASCII = 24,     // '\x1A'        // can not happen
    Cc_SPACE = 25,     // '\x1A'        // can not happen
    Cc_SEPARATOR = 26, // '\x1A'        // can not happen

    Cf_FORMAT = 27, // '\x1A'        // nothing ?
    Cf_JOIN = 28,   // '\x1A'        // nothing ?
    Cf_BIDI = 29,   // '\x1A'        // nothing ?
    Cf_ZWNBSP = 30, // '\x1A'        // nothing ?

    Cn_UNASSIGNED = 0, // '?'
    Co_PRIVATE = 0,    // '?'
    Cs_LOW = 31,       // '?'
    Cs_HIGH = 32,      // '?'

    Pd_DASH = 33,      // '-'
    Pd_HYPHEN = 34,    // '-' [-]       HYPHEN-MINUS
    Ps_START = 35,     // '(' [([{]     LEFT PARENTHESIS ... LEFT CURLY BRACKET
    Ps_QUOTE = 36,     // '"'
    Pe_END = 37,       // ')' [)]}]     RIGHT PARENTHESIS ... RIGHT CURLY BRACKET
    Pe_QUOTE = 38,     // '"'
    Pi_QUOTE = 39,     // '"'
    Pf_QUOTE = 40,     // '"'
    Pc_CONNECTOR = 41, // '_' [_]       LOW LINE
    Po_OTHER = 42,     // '*' [#%&*/@\] NUMBER SIGN ... REVERSE SOLIDUS
    Po_QUOTE = 43,     // '"' ["]       QUOTATION MARK
    Po_TERMINAL = 44,  // '.' [!,.:;?]  EXCLAMATION MARK ... QUESTION MARK
    Po_EXTENDER = 45,  // '-' [№]       MIDDLE DOT (00B7)
    Po_HYPHEN = 46,    // '-'

    Sm_MATH = 47,     // '=' [+<=>|~]  PLUS SIGN ... TILDE
    Sm_MINUS = 48,    // '-'
    Sc_CURRENCY = 49, // '$' [$]       DOLLAR SIGN
    Sk_MODIFIER = 50, // '`' [^`]      CIRCUMFLEX ACCENT ... GRAVE ACCENT
    So_OTHER = 51,    // '°' [°]       DEGREE SIGN (00B0)

    Ps_SINGLE_QUOTE = 52, // '\'' [']   OPENING SINGLE QUOTE
    Pe_SINGLE_QUOTE = 53, // '\'' [']   CLOSING SINGLE QUOTE
    Pi_SINGLE_QUOTE = 54, // '\'' [']   INITIAL SINGLE QUOTE
    Pf_SINGLE_QUOTE = 55, // '\'' [']   FINAL SINGLE QUOTE
    Po_SINGLE_QUOTE = 56, // '\'' [']   APOSTROPHE and PRIME

    CCL_NUM = 57,
    CCL_MASK = 0x3F,

    IS_ASCII_XDIGIT = 1 << 6,
    IS_DIGIT = 1 << 7,
    IS_NONBREAK = 1 << 8,

    IS_PRIVATE = 1 << 9,

    IS_COMPAT = 1 << 10,
    IS_CANON = 1 << 11,

    NFD_QC = 1 << 12,
    NFC_QC = 1 << 13,
    NFKD_QC = 1 << 14,
    NFKC_QC = 1 << 15,

    BIDI_OFFSET = 16,
    SVAL_OFFSET = 22,
};

const size_t DEFCHAR_BUF = 58; // CCL_NUM + 1

#define SHIFT(i) (ULL(1) << (i))

namespace NUnicode {
    using TCombining = ui8;

    namespace NPrivate {
        struct TProperty {
            ui32 Info;
            i32 Lower;
            i32 Upper;
            i32 Title;
            TCombining Combining;
        };

        extern const size_t DEFAULT_KEY;

        using TUnidataTable = NUnicodeTable::TTable<NUnicodeTable::TSubtable<NUnicodeTable::UNICODE_TABLE_SHIFT, NUnicodeTable::TValues<TProperty>>>;
        const TUnidataTable& UnidataTable();

        inline const TProperty& CharProperty(wchar32 ch) {
            return UnidataTable().Get(ch, DEFAULT_KEY);
        }

        inline ui32 CharInfo(wchar32 ch) {
            return CharProperty(ch).Info;
        }

        inline bool IsBidi(wchar32 ch, ui32 type) {
            return ((NUnicode::NPrivate::CharInfo(ch) >> BIDI_OFFSET) & 15) == type;
        }
    } // namespace NPrivate

    inline size_t UnicodeInstancesLimit() {
        return NPrivate::UnidataTable().Size();
    }

    inline TCombining DecompositionCombining(wchar32 ch) {
        return NPrivate::CharProperty(ch).Combining;
    }

    inline WC_TYPE CharType(wchar32 ch) {
        return (WC_TYPE)(NUnicode::NPrivate::CharInfo(ch) & CCL_MASK);
    }
    inline bool CharHasType(wchar32 ch, ui64 type_bits) {
        return (SHIFT(NUnicode::CharType(ch)) & type_bits) != 0;
    }
} // namespace NUnicode

// all usefull properties

inline bool IsComposed(wchar32 ch) {
    return NUnicode::NPrivate::CharInfo(ch) & (IS_COMPAT | IS_CANON);
}
inline bool IsCanonComposed(wchar32 ch) {
    return NUnicode::NPrivate::CharInfo(ch) & IS_CANON;
}
inline bool IsCompatComposed(wchar32 ch) {
    return NUnicode::NPrivate::CharInfo(ch) & IS_COMPAT;
}

inline bool IsWhitespace(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cc_SPACE) | SHIFT(Zs_SPACE) | SHIFT(Zs_ZWSPACE) | SHIFT(Zl_LINE) | SHIFT(Zp_PARAGRAPH));
}
inline bool IsAsciiCntrl(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cc_ASCII) | SHIFT(Cc_SPACE) | SHIFT(Cc_SEPARATOR));
}
inline bool IsBidiCntrl(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cf_BIDI));
}
inline bool IsJoinCntrl(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cf_JOIN));
}
inline bool IsFormatCntrl(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cf_FORMAT));
}
inline bool IsIgnorableCntrl(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cf_FORMAT) | SHIFT(Cf_JOIN) | SHIFT(Cf_BIDI) | SHIFT(Cf_ZWNBSP));
}
inline bool IsCntrl(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Cf_FORMAT) | SHIFT(Cf_JOIN) | SHIFT(Cf_BIDI) | SHIFT(Cf_ZWNBSP) |
                                     SHIFT(Cc_ASCII) | SHIFT(Cc_SPACE) | SHIFT(Cc_SEPARATOR));
}
inline bool IsZerowidth(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cf_FORMAT) | SHIFT(Cf_JOIN) | SHIFT(Cf_BIDI) | SHIFT(Cf_ZWNBSP) | SHIFT(Zs_ZWSPACE));
}
inline bool IsLineSep(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Zl_LINE));
}
inline bool IsParaSep(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Zp_PARAGRAPH));
}
inline bool IsDash(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Pd_DASH) | SHIFT(Pd_HYPHEN) | SHIFT(Sm_MINUS));
}
inline bool IsHyphen(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Pd_HYPHEN) | SHIFT(Po_HYPHEN));
}
inline bool IsQuotation(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Po_QUOTE) | SHIFT(Ps_QUOTE) | SHIFT(Pe_QUOTE) | SHIFT(Pi_QUOTE) |
                                     SHIFT(Pf_QUOTE) | SHIFT(Po_SINGLE_QUOTE) | SHIFT(Ps_SINGLE_QUOTE) |
                                     SHIFT(Pe_SINGLE_QUOTE) | SHIFT(Pi_SINGLE_QUOTE) | SHIFT(Pf_SINGLE_QUOTE));
}

inline bool IsSingleQuotation(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Po_SINGLE_QUOTE) | SHIFT(Ps_SINGLE_QUOTE) | SHIFT(Pe_SINGLE_QUOTE) |
                                     SHIFT(Pi_SINGLE_QUOTE) | SHIFT(Pf_SINGLE_QUOTE));
}

inline bool IsTerminal(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Po_TERMINAL));
}
inline bool IsPairedPunct(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Ps_START) | SHIFT(Pe_END) | SHIFT(Ps_QUOTE) | SHIFT(Pe_QUOTE) |
                                     SHIFT(Pi_QUOTE) | SHIFT(Pf_QUOTE) | SHIFT(Ps_SINGLE_QUOTE) |
                                     SHIFT(Pe_SINGLE_QUOTE) | SHIFT(Pi_SINGLE_QUOTE) | SHIFT(Pf_SINGLE_QUOTE));
}
inline bool IsLeftPunct(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Ps_START) | SHIFT(Ps_QUOTE) | SHIFT(Ps_SINGLE_QUOTE));
}
inline bool IsRightPunct(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Pe_END) | SHIFT(Pe_QUOTE) | SHIFT(Pe_SINGLE_QUOTE));
}
inline bool IsCombining(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Mc_SPACING) | SHIFT(Mn_NONSPACING) | SHIFT(Me_ENCLOSING));
}
inline bool IsNonspacing(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Mn_NONSPACING) | SHIFT(Me_ENCLOSING));
}
inline bool IsAlphabetic(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Lu_UPPER) | SHIFT(Ll_LOWER) | SHIFT(Lt_TITLE) | SHIFT(Lm_EXTENDER) | SHIFT(Lm_LETTER) | SHIFT(Lo_OTHER) | SHIFT(Nl_LETTER));
}
inline bool IsIdeographic(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lo_IDEOGRAPH) | SHIFT(Nl_IDEOGRAPH));
}
inline bool IsKatakana(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lo_KATAKANA));
}
inline bool IsHiragana(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lo_HIRAGANA));
}
inline bool IsHangulLeading(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lo_LEADING));
}
inline bool IsHangulVowel(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lo_VOWEL));
}
inline bool IsHangulTrailing(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lo_TRAILING));
}
inline bool IsHexdigit(wchar32 ch) {
    return NUnicode::NPrivate::CharInfo(ch) & IS_ASCII_XDIGIT;
}
inline bool IsDecdigit(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Nd_DIGIT));
}
inline bool IsNumeric(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Nd_DIGIT) | SHIFT(Nl_LETTER) | SHIFT(Nl_IDEOGRAPH) | SHIFT(No_OTHER));
}
inline bool IsCurrency(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Sc_CURRENCY));
}
inline bool IsMath(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Sm_MATH));
}
inline bool IsSymbol(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Sm_MATH) | SHIFT(Sm_MINUS) | SHIFT(Sc_CURRENCY) | SHIFT(Sk_MODIFIER) | SHIFT(So_OTHER));
}
inline bool IsLowSurrogate(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cs_LOW));
}
inline bool IsHighSurrogate(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cs_HIGH));
}
inline bool IsNonbreak(wchar32 ch) {
    return NUnicode::NPrivate::CharInfo(ch) & IS_NONBREAK;
}
inline bool IsPrivate(wchar32 ch) {
    return (NUnicode::NPrivate::CharInfo(ch) & IS_PRIVATE) && !NUnicode::CharHasType(ch, SHIFT(Cs_HIGH));
}
inline bool IsUnassigned(wchar32 ch) {
    return (NUnicode::CharType(ch) == 0) && !(NUnicode::NPrivate::CharInfo(ch) & IS_PRIVATE);
}
inline bool IsPrivateHighSurrogate(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Cs_HIGH)) && (NUnicode::NPrivate::CharInfo(ch) & IS_PRIVATE);
}

// transformations

inline wchar32 ToLower(wchar32 ch) {
    return static_cast<wchar32>(ch + NUnicode::NPrivate::CharProperty(ch).Lower);
}
inline wchar32 ToUpper(wchar32 ch) {
    return static_cast<wchar32>(ch + NUnicode::NPrivate::CharProperty(ch).Upper);
}
inline wchar32 ToTitle(wchar32 ch) {
    return static_cast<wchar32>(ch + NUnicode::NPrivate::CharProperty(ch).Title);
}

inline int ToDigit(wchar32 ch) {
    ui32 i = NUnicode::NPrivate::CharInfo(ch);
    return (i & IS_DIGIT) ? static_cast<int>(i >> SVAL_OFFSET) : -1;
}

// BIDI properties

inline bool IsBidiLeft(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 1);
}
inline bool IsBidiRight(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 2);
}
inline bool IsBidiEuronum(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 3);
}
inline bool IsBidiEurosep(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 4);
}
inline bool IsBidiEuroterm(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 5);
}
inline bool IsBidiArabnum(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 6);
}
inline bool IsBidiCommsep(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 7);
}
inline bool IsBidiBlocksep(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 8);
}
inline bool IsBidiSegmsep(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 9);
}
inline bool IsBidiSpace(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 10);
}
inline bool IsBidiNeutral(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 11);
}
inline bool IsBidiNotappl(wchar32 ch) {
    return NUnicode::NPrivate::IsBidi(ch, 0);
}

inline bool IsSpace(wchar32 ch) {
    return IsWhitespace(ch);
}
inline bool IsLower(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Ll_LOWER));
}
inline bool IsUpper(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lu_UPPER));
}
inline bool IsTitle(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Lt_TITLE));
}
inline bool IsAlpha(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Lu_UPPER) | SHIFT(Ll_LOWER) | SHIFT(Lt_TITLE) | SHIFT(Lm_LETTER) | SHIFT(Lm_EXTENDER) |
                                     SHIFT(Lo_OTHER) | SHIFT(Lo_IDEOGRAPH) | SHIFT(Lo_KATAKANA) | SHIFT(Lo_HIRAGANA) |
                                     SHIFT(Lo_LEADING) | SHIFT(Lo_VOWEL) | SHIFT(Lo_TRAILING));
}
inline bool IsAlnum(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Lu_UPPER) | SHIFT(Ll_LOWER) | SHIFT(Lt_TITLE) | SHIFT(Lm_LETTER) | SHIFT(Lm_EXTENDER) |
                                     SHIFT(Lo_OTHER) | SHIFT(Lo_IDEOGRAPH) | SHIFT(Lo_KATAKANA) | SHIFT(Lo_HIRAGANA) |
                                     SHIFT(Lo_LEADING) | SHIFT(Lo_VOWEL) | SHIFT(Lo_TRAILING) |
                                     SHIFT(Nd_DIGIT) | SHIFT(Nl_LETTER) | SHIFT(Nl_IDEOGRAPH) | SHIFT(No_OTHER));
}
inline bool IsPunct(wchar32 ch) {
    return NUnicode::CharHasType(ch,
                                 SHIFT(Pd_DASH) |
                                     SHIFT(Pd_HYPHEN) | SHIFT(Ps_START) | SHIFT(Ps_QUOTE) | SHIFT(Pe_END) | SHIFT(Pe_QUOTE) | SHIFT(Pc_CONNECTOR) |
                                     SHIFT(Po_OTHER) | SHIFT(Po_QUOTE) | SHIFT(Po_TERMINAL) | SHIFT(Po_EXTENDER) | SHIFT(Po_HYPHEN) |
                                     SHIFT(Pi_QUOTE) | SHIFT(Pf_QUOTE));
}
inline bool IsXdigit(wchar32 ch) {
    return IsHexdigit(ch);
}
inline bool IsDigit(wchar32 ch) {
    return IsDecdigit(ch);
}

inline bool IsCommonDigit(wchar32 ch) {
    // IsDigit returns true for some exotic symbols like "VAI DIGIT TWO" (U+A622)
    // and cannot be used safely with FromString() convertors
    const wchar32 ZERO = '0';
    const wchar32 NINE = '9';
    return ch >= ZERO && ch <= NINE;
}

inline bool IsGraph(wchar32 ch) {
    return IsAlnum(ch) || IsPunct(ch) || IsSymbol(ch);
}
inline bool IsBlank(wchar32 ch) {
    return NUnicode::CharHasType(ch, SHIFT(Zs_SPACE) | SHIFT(Zs_ZWSPACE)) || ch == '\t';
}
inline bool IsPrint(wchar32 ch) {
    return IsAlnum(ch) || IsPunct(ch) || IsSymbol(ch) || IsBlank(ch);
}

inline bool IsRomanDigit(wchar32 ch) {
    if (NUnicode::CharHasType(ch, SHIFT(Nl_LETTER)) && 0x2160 <= ch && ch <= 0x2188) {
        return true;
    }
    if (ch < 127) {
        switch (static_cast<char>(::ToLower(ch))) {
            case 'i':
            case 'v':
            case 'x':
            case 'l':
            case 'c':
            case 'd':
            case 'm':
                return true;
        }
    }
    return false;
}

#undef SHIFT
