#include "langs.h"

#include <library/cpp/digest/lower_case/hash_ops.h>

#include <util/generic/array_size.h>
#include <util/generic/hash.h>
#include <util/generic/singleton.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/system/defaults.h>

#include <array>
#include <cctype>

/*
 * define language by ELanguage
 */

namespace {
    struct TLanguageNameAndEnum {
        ELanguage Language;
        EScript Script;
        const char* EnglishName;
        const char* BiblioName;
        const char* IsoName;
        const char* Synonyms;
    };

    const TLanguageNameAndEnum LanguageNameAndEnum[] = {
        {LANG_UNK, SCRIPT_OTHER, "Unknown", "unk", "mis", nullptr},
        {LANG_RUS, SCRIPT_CYRILLIC, "Russian", "rus", "ru", "ru-RU"},
        {LANG_ENG, SCRIPT_LATIN, "English", "eng", "en", "en-US, en-GB, en-CA, en-NZ, en-AU"},
        {LANG_POL, SCRIPT_LATIN, "Polish", "pol", "pl", nullptr},
        {LANG_HUN, SCRIPT_LATIN, "Hungarian", "hun", "hu", nullptr},
        {LANG_UKR, SCRIPT_CYRILLIC, "Ukrainian", "ukr", "uk", "uk-UA"},
        {LANG_GER, SCRIPT_LATIN, "German", "ger", "de", "deu"},
        {LANG_FRE, SCRIPT_LATIN, "French", "fre", "fr", "fra, frn, fr-FR, fr-CA"},
        {LANG_TAT, SCRIPT_CYRILLIC, "Tatar", "tat", "tt", nullptr},
        {LANG_BEL, SCRIPT_CYRILLIC, "Belarusian", "bel", "be", "blr, Belorussian"},
        {LANG_KAZ, SCRIPT_CYRILLIC, "Kazakh", "kaz", "kk", "kk-Cyrl"},
        {LANG_ALB, SCRIPT_LATIN, "Albanian", "alb", "sq", nullptr},
        {LANG_SPA, SCRIPT_LATIN, "Spanish", "spa", "es", nullptr},
        {LANG_ITA, SCRIPT_LATIN, "Italian", "ita", "it", nullptr},
        {LANG_ARM, SCRIPT_ARMENIAN, "Armenian", "arm", "hy", "hye"},
        {LANG_DAN, SCRIPT_LATIN, "Danish", "dan", "da", nullptr},
        {LANG_POR, SCRIPT_LATIN, "Portuguese", "por", "pt", nullptr},
        {LANG_ICE, SCRIPT_LATIN, "Icelandic", "ice", "is", "isl"},
        {LANG_SLO, SCRIPT_LATIN, "Slovak", "slo", "sk", "slk"},
        {LANG_SLV, SCRIPT_LATIN, "Slovene", "slv", "sl", "Slovenian"},
        {LANG_DUT, SCRIPT_LATIN, "Dutch", "dut", "nl", "nld"},
        {LANG_BUL, SCRIPT_CYRILLIC, "Bulgarian", "bul", "bg", nullptr},
        {LANG_CAT, SCRIPT_LATIN, "Catalan", "cat", "ca", nullptr},
        {LANG_HRV, SCRIPT_LATIN, "Croatian", "hrv", "hr", "scr"},
        {LANG_CZE, SCRIPT_LATIN, "Czech", "cze", "cs", "ces"},
        {LANG_GRE, SCRIPT_GREEK, "Greek", "gre", "el", "ell"},
        {LANG_HEB, SCRIPT_HEBREW, "Hebrew", "heb", "he", "iw"}, // 'iw' is old ISO-639 code
        {LANG_NOR, SCRIPT_LATIN, "Norwegian", "nor", "no", nullptr},
        {LANG_MAC, SCRIPT_CYRILLIC, "Macedonian", "mac", "mk", nullptr},
        {LANG_SWE, SCRIPT_LATIN, "Swedish", "swe", "sv", nullptr},
        {LANG_KOR, SCRIPT_HANGUL, "Korean", "kor", "ko", nullptr},
        {LANG_LAT, SCRIPT_LATIN, "Latin", "lat", "la", nullptr},
        {LANG_BASIC_RUS, SCRIPT_CYRILLIC, "Basic Russian", "basic-rus", "bas-ru", nullptr},
        {LANG_BOS, SCRIPT_LATIN, "Bosnian", "bos", "bs", nullptr},
        {LANG_MLT, SCRIPT_LATIN, "Maltese", "mlt", "mt", nullptr},

        {LANG_EMPTY, SCRIPT_OTHER, "Empty", "empty", nullptr, nullptr},
        {LANG_UNK_LAT, SCRIPT_LATIN, "Unknown Latin", "unklat", nullptr, nullptr},
        {LANG_UNK_CYR, SCRIPT_CYRILLIC, "Unknown Cyrillic", "unkcyr", nullptr, nullptr},
        {LANG_UNK_ALPHA, SCRIPT_OTHER, "Unknown Alpha", "unkalpha", nullptr, nullptr},

        {LANG_FIN, SCRIPT_LATIN, "Finnish", "fin", "fi", nullptr},
        {LANG_EST, SCRIPT_LATIN, "Estonian", "est", "et", nullptr},
        {LANG_LAV, SCRIPT_LATIN, "Latvian", "lav", "lv", nullptr},
        {LANG_LIT, SCRIPT_LATIN, "Lithuanian", "lit", "lt", nullptr},
        {LANG_BAK, SCRIPT_CYRILLIC, "Bashkir", "bak", "ba", nullptr},
        {LANG_TUR, SCRIPT_LATIN, "Turkish", "tur", "tr", nullptr},
        {LANG_RUM, SCRIPT_LATIN, "Romanian", "rum", "ro", "ron"},
        {LANG_MON, SCRIPT_CYRILLIC, "Mongolian", "mon", "mn", nullptr},
        {LANG_UZB, SCRIPT_LATIN, "Uzbek", "uzb", "uz", "uz-Latn"},
        {LANG_KIR, SCRIPT_CYRILLIC, "Kirghiz", "kir", "ky", "Kyrgyz"},
        {LANG_TGK, SCRIPT_CYRILLIC, "Tajik", "tgk", "tg", nullptr},
        {LANG_TUK, SCRIPT_LATIN, "Turkmen", "tuk", "tk", nullptr},
        {LANG_SRP, SCRIPT_CYRILLIC, "Serbian", "srp", "sr", nullptr},
        {LANG_AZE, SCRIPT_LATIN, "Azerbaijani", "aze", "az", "Azeri"},
        {LANG_BASIC_ENG, SCRIPT_LATIN, "Basic English", "basic-eng", "bas-en", nullptr},
        {LANG_GEO, SCRIPT_GEORGIAN, "Georgian", "geo", "ka", "kat"},
        {LANG_ARA, SCRIPT_ARABIC, "Arabic", "ara", "ar", nullptr},
        {LANG_PER, SCRIPT_ARABIC, "Persian", "per", "fa", "fas"},
        {LANG_CHU, SCRIPT_CYRILLIC, "Church Slavonic", "chu", "cu", nullptr},
        {LANG_CHI, SCRIPT_HAN, "Chinese", "chi", "zh", "zho"},
        {LANG_JPN, SCRIPT_HIRAGANA, "Japanese", "jpn", "ja", nullptr},
        {LANG_IND, SCRIPT_LATIN, "Indonesian", "ind", "id", "in"}, // 'in' is old ISO-639 code
        {LANG_MAY, SCRIPT_LATIN, "Malay", "may", "ms", "msa"},
        {LANG_THA, SCRIPT_THAI, "Thai", "tha", "th", nullptr},
        {LANG_VIE, SCRIPT_LATIN, "Vietnamese", "vie", "vi", nullptr},
        {LANG_GLE, SCRIPT_LATIN, "Irish", "gle", "ga", nullptr},
        {LANG_TGL, SCRIPT_LATIN, "Tagalog", "tgl", "tl", "fil"},
        {LANG_HIN, SCRIPT_DEVANAGARI, "Hindi", "hin", "hi", nullptr},
        {LANG_AFR, SCRIPT_LATIN, "Afrikaans", "afr", "af", nullptr},
        {LANG_URD, SCRIPT_ARABIC, "Urdu", "urd", "ur", nullptr},
        {LANG_MYA, SCRIPT_MYANMAR, "Burmese", "mya", "my", nullptr},
        {LANG_KHM, SCRIPT_KHMER, "Khmer", "khm", "km", nullptr},
        {LANG_LAO, SCRIPT_LAO, "Lao", "lao", "lo", "Laotian, Laothian"},
        {LANG_TAM, SCRIPT_TAMIL, "Tamil", "tam", "ta", nullptr},
        {LANG_BEN, SCRIPT_BENGALI, "Bengali", "ben", "bn", nullptr},
        {LANG_GUJ, SCRIPT_GUJARATI, "Gujarati", "guj", "gu", nullptr},
        {LANG_KAN, SCRIPT_KANNADA, "Kannada", "kan", "kn", nullptr},
        {LANG_PAN, SCRIPT_GURMUKHI, "Punjabi", "pan", "pa", nullptr},
        {LANG_SIN, SCRIPT_SINHALA, "Sinhalese", "sin", "si", nullptr},
        {LANG_SWA, SCRIPT_LATIN, "Swahili", "swa", "sw", nullptr},
        {LANG_BAQ, SCRIPT_LATIN, "Basque", "baq", "eu", "eus"},
        {LANG_WEL, SCRIPT_LATIN, "Welsh", "wel", "cy", "cym"},
        {LANG_GLG, SCRIPT_LATIN, "Galician", "glg", "gl", nullptr},
        {LANG_HAT, SCRIPT_LATIN, "Haitian Creole", "hat", "ht", "Haitian"},
        {LANG_MLG, SCRIPT_LATIN, "Malagasy", "mlg", "mg", nullptr},
        {LANG_CHV, SCRIPT_CYRILLIC, "Chuvash", "chv", "cv", nullptr},
        {LANG_UDM, SCRIPT_CYRILLIC, "Udmurt", "udm", "udm", nullptr},
        {LANG_KPV, SCRIPT_CYRILLIC, "Komi-Zyrian", "kpv", "kv", "Komi, kom"},
        {LANG_MHR, SCRIPT_CYRILLIC, "Meadow Mari", "mhr", "mhr", "EasternMari, Mari, chm"},
        {LANG_SJN, SCRIPT_LATIN, "Sindarin", "sjn", "sjn", nullptr},
        {LANG_MRJ, SCRIPT_CYRILLIC, "Hill Mari", "mrj", "mrj", "WesternMari"},
        {LANG_KOI, SCRIPT_CYRILLIC, "Komi-Permyak", "koi", "koi", nullptr},
        {LANG_LTZ, SCRIPT_LATIN, "Luxembourgish", "ltz", "lb", "Luxemburgish"},
        {LANG_GLA, SCRIPT_LATIN, "Scottish Gaelic", "gla", "gd", "Gaelic"},
        {LANG_CEB, SCRIPT_LATIN, "Cebuano", "ceb", "ceb", "Bisaya, Binisaya, Visayan"},
        {LANG_PUS, SCRIPT_ARABIC, "Pashto", "pus", "ps", nullptr},
        {LANG_KMR, SCRIPT_LATIN, "Kurmanji", "kmr", "ku", "Kurdish"},
        {LANG_AMH, SCRIPT_ETHIOPIC, "Amharic", "amh", "am", nullptr},
        {LANG_ZUL, SCRIPT_LATIN, "Zulu", "zul", "zu", nullptr},
        {LANG_IBO, SCRIPT_LATIN, "Igbo", "ibo", "ig", "Ibo"},
        {LANG_YOR, SCRIPT_LATIN, "Yoruba", "yor", "yo", nullptr},
        {LANG_COS, SCRIPT_LATIN, "Corsican", "cos", "co", nullptr},
        {LANG_XHO, SCRIPT_LATIN, "Xhosa", "xho", "xh", nullptr},
        {LANG_JAV, SCRIPT_LATIN, "Javanese", "jav", "jv", nullptr}, // Also SCRIPT_JAVANESE and SCRIPT_ARABIC
        {LANG_NEP, SCRIPT_DEVANAGARI, "Nepali", "nep", "ne", nullptr},
        {LANG_SND, SCRIPT_DEVANAGARI, "Sindhi", "snd", "sd", nullptr}, // Also SCRIPT_ARABIC and SCRIPT_GUJARATI
        {LANG_SOM, SCRIPT_LATIN, "Somali", "som", "so", nullptr},
        {LANG_EPO, SCRIPT_LATIN, "Esperanto", "epo", "eo", nullptr},
        {LANG_TEL, SCRIPT_TELUGU, "Telugu", "tel", "te", nullptr},
        {LANG_MAR, SCRIPT_DEVANAGARI, "Marathi", "mar", "mr", nullptr},
        {LANG_HAU, SCRIPT_LATIN, "Hausa", "hau", "ha", nullptr},
        {LANG_YID, SCRIPT_HEBREW, "Yiddish", "yid", "yi", nullptr},
        {LANG_MAL, SCRIPT_MALAYALAM, "Malayalam", "mal", "ml", nullptr},
        {LANG_MAO, SCRIPT_LATIN, "Maori", "mao", "mi", "mri"},
        {LANG_SUN, SCRIPT_LATIN, "Sundanese", "sun", "su", nullptr},
        {LANG_PAP, SCRIPT_LATIN, "Papiamento", "pap", "pap", nullptr},
        {LANG_UZB_CYR, SCRIPT_CYRILLIC, "Cyrillic Uzbek", "uzbcyr", "uz-Cyrl", nullptr}, // https://tools.ietf.org/html/rfc5646
        {LANG_TRANSCR_IPA, SCRIPT_LATIN, "International Phonetic Alphabet Transcription", "ipa", "tr-ipa", nullptr},
        {LANG_EMJ, SCRIPT_LATIN, "Emoji", "emj", "emj", nullptr},
        {LANG_UYG, SCRIPT_ARABIC, "Uyghur", "uig", "ug", nullptr},
        {LANG_BRE, SCRIPT_LATIN, "Breton", "bre", "br", nullptr},
        {LANG_SAH, SCRIPT_CYRILLIC, "Yakut", "sah", "sah", nullptr},
        {LANG_KAZ_LAT, SCRIPT_LATIN, "Latin Kazakh", "kazlat", "kk-Latn", nullptr},
        {LANG_KJH, SCRIPT_CYRILLIC, "Khakas", "kjh", "kjh", "khk"},
        {LANG_OSS, SCRIPT_CYRILLIC, "Ossetian", "oss", "os", nullptr},
        {LANG_TYV, SCRIPT_CYRILLIC, "Tuvan", "tyv", "tyv", nullptr},
        {LANG_CHE, SCRIPT_CYRILLIC, "Chechen", "che", "ce", nullptr},
        {LANG_MNS, SCRIPT_CYRILLIC, "Mansi", "mns", "mns", nullptr},
        {LANG_ARZ, SCRIPT_ARABIC, "Egyptian Arabic", "arz", "arz", "ar-EG"},
        {LANG_KRC, SCRIPT_CYRILLIC, "Karachay–Balkar", "krc", "krc", nullptr},
        {LANG_KBD, SCRIPT_CYRILLIC, "Kabardino-Cherkess", "kbd", "kbd", nullptr},
        {LANG_NOG, SCRIPT_CYRILLIC, "Nogai", "nog", "nog", nullptr},
        {LANG_ABQ, SCRIPT_CYRILLIC, "Abaza", "abq", "abq", nullptr},
        {LANG_MYV, SCRIPT_CYRILLIC, "Erzya", "myv", "myv", nullptr},
        {LANG_MDF, SCRIPT_CYRILLIC, "Moksha", "mdf", "mdf", nullptr},
    };

    static_assert(static_cast<size_t>(LANG_MAX) == Y_ARRAY_SIZE(LanguageNameAndEnum), "Size doesn't match");

    class TLanguagesMap {
    private:
        static const char* const EMPTY_NAME;

        using TNamesHash = THashMap<TStringBuf, ELanguage, TCIOps, TCIOps>;
        TNamesHash Hash;

        using TNamesArray = std::array<const char*, static_cast<size_t>(LANG_MAX)>;
        TNamesArray BiblioNames;
        TNamesArray IsoNames;
        TNamesArray FullNames;

        using TScripts = std::array<EScript, static_cast<size_t>(LANG_MAX)>;
        TScripts Scripts;

    private:
        void AddNameToHash(const TStringBuf& name, ELanguage language) {
            if (Hash.find(name) != Hash.end()) {
                Y_ASSERT(Hash.find(name)->second == language);
                return;
            }

            Hash[name] = language;
        }

        void AddName(const char* name, ELanguage language, TNamesArray& names) {
            if (name == nullptr || strlen(name) == 0)
                return;

            Y_ASSERT(names[language] == EMPTY_NAME);
            names[language] = name;

            AddNameToHash(name, language);
        }

        void AddSynonyms(const char* syn, ELanguage language) {
            static const char* del = " ,;";
            if (!syn)
                return;
            while (*syn) {
                size_t len = strcspn(syn, del);
                AddNameToHash(TStringBuf(syn, len), language);
                syn += len;
                while (*syn && strchr(del, *syn))
                    ++syn;
            }
        }

    public:
        TLanguagesMap() {
            BiblioNames.fill(EMPTY_NAME);
            IsoNames.fill(EMPTY_NAME);
            FullNames.fill(EMPTY_NAME);
            Scripts.fill(SCRIPT_OTHER);

            for (size_t i = 0; i != Y_ARRAY_SIZE(LanguageNameAndEnum); ++i) {
                const TLanguageNameAndEnum& val = LanguageNameAndEnum[i];

                ELanguage language = val.Language;

                AddName(val.BiblioName, language, BiblioNames);
                AddName(val.IsoName, language, IsoNames);
                AddName(val.EnglishName, language, FullNames);
                AddSynonyms(val.Synonyms, language);

                if (Scripts[language] == SCRIPT_OTHER) {
                    Scripts[language] = val.Script;
                }
            }
        }

    public:
        inline ELanguage LanguageByName(const TStringBuf& name, ELanguage def) const {
            if (!name)
                return def;

            TNamesHash::const_iterator i = Hash.find(name);
            if (i == Hash.end()) {
                // Try to extract the primary language code from constructions like "en-cockney" or "zh_Hant"
                size_t dash_pos = name.find_first_of("_-");
                if (dash_pos != TStringBuf::npos)
                    i = Hash.find(name.substr(0, dash_pos));
                if (i == Hash.end())
                    return def;
            }

            return i->second;
        }

        inline const char* FullNameByLanguage(ELanguage language) const {
            if (static_cast<size_t>(language) >= FullNames.size())
                return nullptr;

            return FullNames[language];
        }
        inline const char* BiblioNameByLanguage(ELanguage language) const {
            if (static_cast<size_t>(language) >= BiblioNames.size())
                return nullptr;

            return BiblioNames[language];
        }
        inline const char* IsoNameByLanguage(ELanguage language) const {
            if (static_cast<size_t>(language) >= IsoNames.size())
                return nullptr;

            return IsoNames[language];
        }

        inline EScript Script(ELanguage language) const {
            return Scripts[language];
        }
    };
}

const char* const TLanguagesMap::EMPTY_NAME = "";

const char* FullNameByLanguage(ELanguage language) {
    return Singleton<TLanguagesMap>()->FullNameByLanguage(language);
}
const char* NameByLanguage(ELanguage language) {
    return Singleton<TLanguagesMap>()->BiblioNameByLanguage(language);
}
const char* IsoNameByLanguage(ELanguage language) {
    return Singleton<TLanguagesMap>()->IsoNameByLanguage(language);
}

ELanguage LanguageByNameStrict(const TStringBuf& name) {
    return Singleton<TLanguagesMap>()->LanguageByName(name, LANG_MAX);
}

ELanguage LanguageByNameOrDie(const TStringBuf& name) {
    ELanguage result = LanguageByNameStrict(name);
    if (result == LANG_MAX) {
        ythrow yexception() << "LanguageByNameOrDie: invalid language '" << name << "'";
    }
    return result;
}

ELanguage LanguageByName(const TStringBuf& name) {
    return Singleton<TLanguagesMap>()->LanguageByName(name, LANG_UNK);
}

EScript ScriptByLanguage(ELanguage language) {
    return Singleton<TLanguagesMap>()->Script(language);
}

namespace {
    const size_t MAX_GLYPH = 0x10000;
    class TScriptGlyphIndex {
    public:
        TScriptGlyphIndex() {
            NCharsetInternal::InitScriptData(Data, MAX_GLYPH);
        }

        EScript GetGlyphScript(wchar32 glyph) const {
            if (glyph >= MAX_GLYPH)
                return SCRIPT_UNKNOWN;
            return (EScript)Data[glyph];
        }

    private:
        ui8 Data[MAX_GLYPH];
    };
}

EScript ScriptByGlyph(wchar32 glyph) {
    return HugeSingleton<TScriptGlyphIndex>()->GetGlyphScript(glyph);
}

template <>
void Out<ELanguage>(IOutputStream& o, ELanguage lang) {
    o << NameByLanguage(lang);
}
