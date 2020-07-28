#include "langmask.h"

#include <util/generic/hash.h>
#include <util/generic/singleton.h>
#include <util/string/split.h>
#include <util/string/strip.h>
#include <util/system/compiler.h>

#include <array>

namespace NLanguageMasks {
    namespace {
        struct TScriptMapX: public TScriptMap {
            TScriptMapX() {
                for (size_t i = 0; i != LANG_MAX; ++i) {
                    ELanguage language = static_cast<ELanguage>(i);
                    if (!UnknownLanguage(language))
                        (*this)[ScriptByLanguage(language)].SafeSet(language);
                }
            }
        };
    }

    const TScriptMap& ScriptMap() {
        return *Singleton<TScriptMapX>();
    }

    const TLangMask& CyrillicLanguagesExt() {
        return ScriptMap().find(SCRIPT_CYRILLIC)->second;
    }

    const TLangMask& LatinLanguages() {
        return ScriptMap().find(SCRIPT_LATIN)->second;
    }

    const TLangMask& SameScriptLanguages(EScript scr) {
        static const TLangMask empty;
        TScriptMap::const_iterator it = ScriptMap().find(scr);
        return ScriptMap().end() == it ? empty : it->second;
    }

    TLangMask SameScriptLanguages(TLangMask src) {
        TLangMask dst;
        for (auto lg : src) {
            TScriptMap::const_iterator it = ScriptMap().find(ScriptByLanguage(lg));
            if (ScriptMap().end() != it) {
                dst |= it->second;
                src &= ~it->second; // don't need others using the same script
            }
        }
        return dst;
    }

    template <typename T>
    TLangMask CreateFromListImpl(const TString& list, T langGetter) {
        TLangMask result;
        TVector<TString> langVector;
        StringSplitter(list).Split(',').SkipEmpty().Collect(&langVector);
        for (const auto& i : langVector) {
            ELanguage lang = langGetter(Strip(i).data());
            if (lang == LANG_MAX)
                ythrow yexception() << "Unknown language: " << i;
            result.SafeSet(lang);
        }
        return result;
    }

    TLangMask CreateFromList(const TString& list) {
        return CreateFromListImpl(list, LanguageByNameStrict);
    }

    TLangMask SafeCreateFromList(const TString& list) {
        return CreateFromListImpl(list, LanguageByName);
    }

    TString ToString(const TLangMask& langMask) {
        if (langMask.Empty())
            return NameByLanguage(LANG_UNK);
        TString result;
        for (auto lang : langMask) {
            if (!!result)
                result += ",";
            result += NameByLanguage(lang);
        }
        return result;
    }
}

namespace {
    struct TNewLanguageEnumToOldLanguageHelper {
        TNewLanguageEnumToOldLanguageHelper() {
            static const TOldLanguageEncoder::TLanguageId LI_UNKNOWN = 0x00000000; // special code - shall be zero
            static const TOldLanguageEncoder::TLanguageId LI_ENGLISH = 0x00000001;
            static const TOldLanguageEncoder::TLanguageId LI_RUSSIAN = 0x00000002;
            static const TOldLanguageEncoder::TLanguageId LI_POLISH = 0x00000004;
            static const TOldLanguageEncoder::TLanguageId LI_UKRAINIAN = 0x00000008;
            static const TOldLanguageEncoder::TLanguageId LI_GERMAN = 0x00000010;
            static const TOldLanguageEncoder::TLanguageId LI_FRENCH = 0x00000020;
            // Beware: a hole should be left at 0x40 - 0x80,
            // to prevent overlap with CC_UPPERCASE / CC_TITLECASE
            static const TOldLanguageEncoder::TLanguageId LI_HUNGARIAN = 0x00000100;
            // static const TOldLanguageEncoder::TLanguageId LI_UKRAINIAN_ABBYY    = 0x00000200;
            static const TOldLanguageEncoder::TLanguageId LI_ITALIAN = 0x00000400;
            static const TOldLanguageEncoder::TLanguageId LI_BELORUSSIAN = 0x00000800;
            static const TOldLanguageEncoder::TLanguageId LI_KAZAKH = 0x00008000;

            Direct[LANG_UNK] = LI_UNKNOWN;
            Direct[LANG_ENG] = LI_ENGLISH;
            Direct[LANG_RUS] = LI_RUSSIAN;
            Direct[LANG_POL] = LI_POLISH;
            Direct[LANG_UKR] = LI_UKRAINIAN;
            Direct[LANG_GER] = LI_GERMAN;
            Direct[LANG_FRE] = LI_FRENCH;
            Direct[LANG_HUN] = LI_HUNGARIAN;
            // Direct[] = LI_UKRAINIAN_ABBYY;
            Direct[LANG_ITA] = LI_ITALIAN;
            Direct[LANG_BEL] = LI_BELORUSSIAN;
            Direct[LANG_KAZ] = LI_KAZAKH;

            for (auto i = Direct.size(); i > 0; --i) {
                Reverse[Direct[i - 1]] = static_cast<ELanguage>(i - 1);
            }

            Y_ENSURE(LANG_UNK == Reverse.find(LI_UNKNOWN)->second, "Must be equal");
        }

        THashMap< ::TOldLanguageEncoder::TLanguageId, ELanguage> Reverse;
        std::array< ::TOldLanguageEncoder::TLanguageId, static_cast<size_t>(LANG_MAX)> Direct;
    };
}

TOldLanguageEncoder::TLanguageId TOldLanguageEncoder::ToOld(ELanguage l) {
    const auto& helper = Default<TNewLanguageEnumToOldLanguageHelper>();
    if (Y_UNLIKELY(static_cast<size_t>(l) >= helper.Direct.size())) {
        l = LANG_UNK;
    }

    return helper.Direct[l];
}

ELanguage TOldLanguageEncoder::FromOld1(TOldLanguageEncoder::TLanguageId l) {
    const auto& helper = Default<TNewLanguageEnumToOldLanguageHelper>();
    const auto it = helper.Reverse.find(l);
    if (Y_UNLIKELY(it == helper.Reverse.end())) {
        return LANG_UNK;
    }

    return it->second;
}
