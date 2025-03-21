#pragma once

#include "scripts.h"

#include <util/generic/strbuf.h>
#include <util/system/defaults.h>

#if defined(_win_)
// LANG_LAO is #define in WinNT.h
#undef LANG_LAO
#endif

// Language names are given according to ISO 639-2/B
// Some languages are not present in ISO 639-2/B. Then ISO 639-3 is used.
// http://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
enum ELanguage: unsigned {
    LANG_UNK = 0,           // Unknown
    LANG_RUS = 1,           // Russian
    LANG_ENG = 2,           // English
    LANG_POL = 3,           // Polish
    LANG_HUN = 4,           // Hungarian
    LANG_UKR = 5,           // Ukrainian
    LANG_GER = 6,           // German
    LANG_FRE = 7,           // French
    LANG_TAT = 8,           // Tatar
    LANG_BEL = 9,           // Belarusian
    LANG_KAZ = 10,          // Kazakh
    LANG_ALB = 11,          // Albanian
    LANG_SPA = 12,          // Spanish
    LANG_ITA = 13,          // Italian
    LANG_ARM = 14,          // Armenian
    LANG_DAN = 15,          // Danish
    LANG_POR = 16,          // Portuguese
    LANG_ICE = 17,          // Icelandic
    LANG_SLO = 18,          // Slovak
    LANG_SLV = 19,          // Slovene
    LANG_DUT = 20,          // Dutch (Netherlandish language)
    LANG_BUL = 21,          // Bulgarian
    LANG_CAT = 22,          // Catalan
    LANG_HRV = 23,          // Croatian
    LANG_CZE = 24,          // Czech
    LANG_GRE = 25,          // Greek
    LANG_HEB = 26,          // Hebrew
    LANG_NOR = 27,          // Norwegian
    LANG_MAC = 28,          // Macedonian
    LANG_SWE = 29,          // Swedish
    LANG_KOR = 30,          // Korean
    LANG_LAT = 31,          // Latin
    LANG_BASIC_RUS = 32,    // Simplified version of Russian (used at lemmer only)
    LANG_BOS = 33,          // Bosnian
    LANG_MLT = 34,          // Maltese
    LANG_EMPTY = 35,        // Indicate that document is empty
    LANG_UNK_LAT = 36,      // Any unrecognized latin language
    LANG_UNK_CYR = 37,      // Any unrecognized cyrillic language
    LANG_UNK_ALPHA = 38,    // Any unrecognized alphabetic language not fit into previous categories
    LANG_FIN = 39,          // Finnish
    LANG_EST = 40,          // Estonian
    LANG_LAV = 41,          // Latvian
    LANG_LIT = 42,          // Lithuanian
    LANG_BAK = 43,          // Bashkir
    LANG_TUR = 44,          // Turkish
    LANG_RUM = 45,          // Romanian (also Moldavian)
    LANG_MON = 46,          // Mongolian
    LANG_UZB = 47,          // Uzbek
    LANG_KIR = 48,          // Kirghiz
    LANG_TGK = 49,          // Tajik
    LANG_TUK = 50,          // Turkmen
    LANG_SRP = 51,          // Serbian
    LANG_AZE = 52,          // Azerbaijani
    LANG_BASIC_ENG = 53,    // Simplified version of English (used at lemmer only)
    LANG_GEO = 54,          // Georgian
    LANG_ARA = 55,          // Arabic
    LANG_PER = 56,          // Persian
    LANG_CHU = 57,          // Church Slavonic
    LANG_CHI = 58,          // Chinese
    LANG_JPN = 59,          // Japanese
    LANG_IND = 60,          // Indonesian
    LANG_MAY = 61,          // Malay
    LANG_THA = 62,          // Thai
    LANG_VIE = 63,          // Vietnamese
    LANG_GLE = 64,          // Irish (Gaelic)
    LANG_TGL = 65,          // Tagalog (Filipino)
    LANG_HIN = 66,          // Hindi
    LANG_AFR = 67,          // Afrikaans
    LANG_URD = 68,          // Urdu
    LANG_MYA = 69,          // Burmese
    LANG_KHM = 70,          // Khmer
    LANG_LAO = 71,          // Lao
    LANG_TAM = 72,          // Tamil
    LANG_BEN = 73,          // Bengali
    LANG_GUJ = 74,          // Gujarati
    LANG_KAN = 75,          // Kannada
    LANG_PAN = 76,          // Punjabi
    LANG_SIN = 77,          // Sinhalese
    LANG_SWA = 78,          // Swahili
    LANG_BAQ = 79,          // Basque
    LANG_WEL = 80,          // Welsh
    LANG_GLG = 81,          // Galician
    LANG_HAT = 82,          // Haitian Creole
    LANG_MLG = 83,          // Malagasy
    LANG_CHV = 84,          // Chuvash
    LANG_UDM = 85,          // Udmurt
    LANG_KPV = 86,          // Komi-Zyrian
    LANG_MHR = 87,          // Meadow Mari (Eastern Mari)
    LANG_SJN = 88,          // Sindarin
    LANG_MRJ = 89,          // Hill Mari (Western Mari)
    LANG_KOI = 90,          // Komi-Permyak
    LANG_LTZ = 91,          // Luxembourgish
    LANG_GLA = 92,          // Scottish Gaelic
    LANG_CEB = 93,          // Cebuano
    LANG_PUS = 94,          // Pashto
    LANG_KMR = 95,          // Kurmanji
    LANG_AMH = 96,          // Amharic
    LANG_ZUL = 97,          // Zulu
    LANG_IBO = 98,          // Igbo
    LANG_YOR = 99,          // Yoruba
    LANG_COS = 100,         // Corsican
    LANG_XHO = 101,         // Xhosa
    LANG_JAV = 102,         // Javanese
    LANG_NEP = 103,         // Nepali
    LANG_SND = 104,         // Sindhi
    LANG_SOM = 105,         // Somali
    LANG_EPO = 106,         // Esperanto
    LANG_TEL = 107,         // Telugu
    LANG_MAR = 108,         // Marathi
    LANG_HAU = 109,         // Hausa
    LANG_YID = 110,         // Yiddish
    LANG_MAL = 111,         // Malayalam
    LANG_MAO = 112,         // Maori
    LANG_SUN = 113,         // Sundanese
    LANG_PAP = 114,         // Papiamento
    LANG_UZB_CYR = 115,     // Cyrillic Uzbek
    LANG_TRANSCR_IPA = 116, // International Phonetic Alphabet Transcription
    LANG_EMJ = 117,         // Emoji
    LANG_UYG = 118,         // Uyghur
    LANG_BRE = 119,         // Breton
    LANG_SAH = 120,         // Yakut
    LANG_KAZ_LAT = 121,     // Latin Kazakh
    LANG_KJH = 122,         // Khakas
    LANG_OSS = 123,         // Ossetian
    LANG_TYV = 124,         // Tuvan
    LANG_CHE = 125,         // Chechen
    LANG_MNS = 126,         // Mansi
    LANG_ARZ = 127,         // Egyptian Arabic
    LANG_KRC = 128,         // Karachay-Balkar
    LANG_KBD = 129,         // Kabardino-Cherkess
    LANG_NOG = 130,         // Nogai
    LANG_ABQ = 131,         // Abaza
    LANG_MYV = 132,         // Erzya
    LANG_MDF = 133,         // Moksha
    LANG_MAX
};

/**
 * Converts string to corresponding enum. Will try to extract the primary language code from
 * constructions like "en-cockney" or "zh_Hant". In case of failure will return `LANG_UNK`.
 *
 * @param name              Language name
 * @return                  Language enum
 */
ELanguage LanguageByName(const TStringBuf& name);

/**
 * Same as `LanguageByName`, but in case of failure will return `LANG_MAX`.
 *
 * @see LanguageByName
 */
ELanguage LanguageByNameStrict(const TStringBuf& name);

/**
 * Converts language enum to corresponding ISO 639-2/B alpha-3 code. For languages missing in ISO
 * standard convertions are:
 *   - LANG_UNK:          "unk"
 *   - LANG_BASIC_RUS:    "basic-rus"
 *   - LANG_EMPTY:        "empty"
 *   - LANG_UNK_LAT:      "unklat"
 *   - LANG_UNK_CYR:      "unkcyr"
 *   - LANG_UNK_ALPHA:    "unkalpha"
 *   - LANG_BASIC_ENG:    "basic-eng"
 *   - LANG_TRANSCR_IPA   "transcr-ipa"
 * If language is missing in `ELanguage` or if it is a `LANG_MAX` then return value will be
 * `nullptr`.
 *
 * @param language         Language enum
 * @return                 Language ISO 639-2/B alpha-3 code
 */
const char* NameByLanguage(ELanguage language);

/**
 * Converts language enum to corresponding ISO 639-1 alpha-2 code. For languages missing in ISO
 * standard convertions are:
 *   - LANG_UNK:          "mis"
 *   - LANG_BASIC_RUS:    "bas-ru"
 *   - LANG_EMPTY:        ""
 *   - LANG_UNK_LAT:      ""
 *   - LANG_UNK_CYR:      ""
 *   - LANG_UNK_ALPHA:    ""
 *   - LANG_BASIC_ENG:    "bas-en"
 *   - LANG_TRANSCR_IPA   "tr-ipa"
 * If language is missing in `ELanguage` or if it is a `LANG_MAX` then return value will be
 * `nullptr`.
 *
 * @param language         Language enum
 * @return                 Language ISO 639-1 alpha-2 code
 */
const char* IsoNameByLanguage(ELanguage language);

/**
 * Converts language enum to corresponding human-readable language name. E.g. "Russian" for
 * `LANG_RUS` or "Basic Russian" for `LANG_BASIC_RUS`. If language is missing in `ELanguage` or if
 * it is a `LANG_MAX` then return value will be `nullptr`.
 *
 * @param language         Language enum
 */
const char* FullNameByLanguage(ELanguage language);

/**
 * Same as `LanguageByNameStrict` but in case of failure will throw `yexception`.
 *
 * @see LanguageByNameStrict
 */
ELanguage LanguageByNameOrDie(const TStringBuf& name);

constexpr bool UnknownLanguage(const ELanguage language) noexcept {
    return language == LANG_UNK || language == LANG_UNK_LAT || language == LANG_UNK_CYR || language == LANG_UNK_ALPHA || language == LANG_EMPTY;
}

EScript ScriptByLanguage(ELanguage language);
EScript ScriptByGlyph(wchar32 glyph);

namespace NCharsetInternal {
    void InitScriptData(ui8 data[], size_t len);
}

inline bool LatinScript(ELanguage language) {
    return ScriptByLanguage(language) == SCRIPT_LATIN;
}

inline bool CyrillicScript(ELanguage language) {
    return ScriptByLanguage(language) == SCRIPT_CYRILLIC;
}
