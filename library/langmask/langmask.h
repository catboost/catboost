#pragma once

#include <library/cpp/enumbitset/enumbitset.h>
#include <library/cpp/langs/langs.h>

#include <util/generic/fwd.h>

typedef TSfEnumBitSet<ELanguage, static_cast<ELanguage>(LANG_UNK + 1), LANG_MAX> TLangMask;

// Useful language sets
namespace NLanguageMasks {
    using TScriptMap = THashMap<EScript, TLangMask>;

    const TScriptMap& ScriptMap();

    inline const TLangMask& BasicLanguages() {
        const static TLangMask ret(LANG_ENG, LANG_RUS, LANG_UKR);
        return ret;
    }
    inline const TLangMask& DefaultRequestLanguages() {
        const static TLangMask ret = BasicLanguages() | TLangMask(LANG_KAZ, LANG_BEL, LANG_TAT);
        return ret;
    }
    inline const TLangMask& AllLanguages() {
        const static TLangMask ret = ~TLangMask() & ~TLangMask(LANG_BASIC_ENG, LANG_BASIC_RUS);
        return ret;
    }
    inline const TLangMask& CyrillicLanguages() {
        const static TLangMask ret = TLangMask(LANG_RUS, LANG_UKR, LANG_BEL);
        return ret;
    }
    const TLangMask& CyrillicLanguagesExt();
    const TLangMask& LatinLanguages();
    inline const TLangMask& LemmasInIndex() {
        const static TLangMask ret = TLangMask(LANG_RUS, LANG_ENG, LANG_UKR, LANG_TUR) |
                                     TLangMask(LANG_BASIC_RUS, LANG_BASIC_ENG);
        return ret;
    }
    inline const TLangMask& NoBastardsInSearch() {
        const static TLangMask ret = ~LemmasInIndex();
        return ret;
    }

    TLangMask SameScriptLanguages(TLangMask mask);

    inline TLangMask RestrictLangMaskWithSameScripts(const TLangMask& mask, const TLangMask& by) {
        return mask & ~SameScriptLanguages(by);
    }

    const TLangMask& SameScriptLanguages(EScript scr);

    inline TLangMask OtherSameScriptLanguages(const TLangMask& mask) {
        return ~mask & SameScriptLanguages(mask);
    }

    //List is string with list of languages names splinted by ','.
    TLangMask CreateFromList(const TString& list);     // throws exception on unknown name
    TLangMask SafeCreateFromList(const TString& list); // ignore unknown names

    TString ToString(const TLangMask& langMask);

}

#define LI_BASIC_LANGUAGES NLanguageMasks::BasicLanguages()
#define LI_DEFAULT_REQUEST_LANGUAGES NLanguageMasks::DefaultRequestLanguages()
#define LI_ALL_LANGUAGES NLanguageMasks::AllLanguages()
#define LI_CYRILLIC_LANGUAGES NLanguageMasks::CyrillicLanguages()
#define LI_CYRILLIC_LANGUAGES_EXT NLanguageMasks::CyrillicLanguagesExt()
#define LI_LATIN_LANGUAGES NLanguageMasks::LatinLanguages()

// Casing and composition of a word. Used in bitwise unions.
using TCharCategory = long;
const TCharCategory CC_EMPTY = 0x0000;
const TCharCategory CC_ALPHA = 0x0001;
const TCharCategory CC_NMTOKEN = 0x0002;
const TCharCategory CC_NUMBER = 0x0004;
const TCharCategory CC_NUTOKEN = 0x0008;
// Beware: CC_ASCII .. CC_TITLECASE shall occupy bits 4 to 6. Don't move them.
const TCharCategory CC_ASCII = 0x0010;
const TCharCategory CC_NONASCII = 0x0020;
const TCharCategory CC_TITLECASE = 0x0040;
const TCharCategory CC_UPPERCASE = 0x0080;
const TCharCategory CC_LOWERCASE = 0x0100;
const TCharCategory CC_MIXEDCASE = 0x0200;
const TCharCategory CC_COMPOUND = 0x0400;
const TCharCategory CC_HAS_DIACRITIC = 0x0800;
const TCharCategory CC_DIFFERENT_ALPHABET = 0x1000;

const TCharCategory CC_WHOLEMASK = 0x1FFF;

struct TOldLanguageEncoder {
    typedef long TLanguageId;

public:
    static TLanguageId ToOld(ELanguage l);

    static ELanguage FromOld1(TLanguageId l);

    static TLanguageId ToOld(const TLangMask& lm) {
        TLanguageId ret = 0;
        for (ELanguage lg : lm) {
            TLanguageId id = ToOld(lg);
            ret |= id;
        }
        return ret;
    }

    static TLangMask FromOld(TLanguageId lm) {
        static const TLanguageId allLangMask = TLanguageId(-1) & ~(0x40 | 0x80);
        static const size_t numBits = sizeof(TLanguageId) * CHAR_BIT;
        TLangMask ret;
        lm &= allLangMask;
        for (size_t i = 1; i < numBits; ++i) {
            TLanguageId id = TLanguageId(1) << (i - 1);
            if (lm & id)
                ret.SafeSet(FromOld1(id));
        }
        return ret;
    }
};
