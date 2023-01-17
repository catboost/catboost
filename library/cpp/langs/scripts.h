#pragma once

#include <util/generic/strbuf.h>

// Writing systems, a.k.a. scripts
//
enum EScript {
    SCRIPT_UNKNOWN = 0,
    SCRIPT_LATIN,
    SCRIPT_CYRILLIC,

    SCRIPT_GREEK,
    SCRIPT_ARABIC,
    SCRIPT_HEBREW,
    SCRIPT_ARMENIAN,
    SCRIPT_GEORGIAN,

    SCRIPT_HAN,
    SCRIPT_KATAKANA,
    SCRIPT_HIRAGANA,
    SCRIPT_HANGUL,

    SCRIPT_DEVANAGARI,
    SCRIPT_BENGALI,
    SCRIPT_GUJARATI,
    SCRIPT_GURMUKHI,
    SCRIPT_KANNADA,
    SCRIPT_MALAYALAM,
    SCRIPT_ORIYA,
    SCRIPT_TAMIL,
    SCRIPT_TELUGU,
    SCRIPT_THAANA,
    SCRIPT_SINHALA,

    SCRIPT_MYANMAR,
    SCRIPT_THAI,
    SCRIPT_LAO,
    SCRIPT_KHMER,
    SCRIPT_TIBETAN,
    SCRIPT_MONGOLIAN,

    SCRIPT_ETHIOPIC,
    SCRIPT_RUNIC,
    SCRIPT_COPTIC,
    SCRIPT_SYRIAC,

    SCRIPT_OTHER,
    SCRIPT_MAX
};

// According to ISO 15924 codes. See https://en.wikipedia.org/wiki/ISO_15924
//
EScript ScriptByName(const TStringBuf& name);
EScript ScriptByNameOrDie(const TStringBuf& name);
const char* IsoNameByScript(EScript script);
const char* FullNameByScript(EScript script);
