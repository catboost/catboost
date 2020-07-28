#pragma once

#include "langmask.h"

class TIndexLanguageOptions {
private:
    TLangMask LangMask;
    ELanguage LangPriorityList[2]; //!< the first language used only, the second one is always LANG_UNK

private:
    void SetPriorityLang(ELanguage lang) {
        LangPriorityList[0] = lang;
        LangPriorityList[1] = LANG_UNK;
    }

    void SetLangMask(ELanguage lang) {
        LangMask = AdditionalLanguages(lang);
        LangMask.SafeSet(lang);
    }

public:
    explicit TIndexLanguageOptions(ELanguage lang) {
        SetLang(lang);
    }

    TIndexLanguageOptions(const TLangMask& mask, ELanguage lang)
        : LangMask(mask)
    {
        SetPriorityLang(lang);
    }

    void SetLang(ELanguage lang) {
        if (UnknownLanguage(lang)) // всякую упячку считать упячкой
            lang = LANG_UNK;

        SetPriorityLang(lang);
        SetLangMask(lang);
    }

    const TLangMask& GetLangMask() const {
        return LangMask;
    }

    ELanguage GetLanguage() const {
        return LangPriorityList[0];
    }

    //! the language priority list used for AnalyzeWord() calls only
    const ELanguage* GetLangPriorityList() const {
        return LangPriorityList;
    }

public: // static functions
    static bool CyrillicLanguagesPrecludesRussian(ELanguage lang) {
        switch (lang) {
            case LANG_RUS: // русский
            case LANG_BUL: // болгарский
            case LANG_MAC: // македонский
            case LANG_SRP: // сербский
                           // case LANG_UNK_CYR: // Упячка, но кириллическая
                return true;
            default:
                return false;
        }
    }

    static TLangMask AdditionalLanguages(ELanguage lang) {
        switch (lang) {
            case LANG_SRP: // сербский
                return TLangMask(LANG_HRV);
            case LANG_HRV: // хорватский
                return TLangMask(LANG_SRP);
            default:
                if (UnknownLanguage(lang))
                    return TLangMask();
                if (LatinScript(lang))
                    return TLangMask(LANG_BASIC_RUS);
                if (CyrillicLanguagesPrecludesRussian(lang))
                    return TLangMask(LANG_BASIC_ENG);
                return TLangMask(LANG_BASIC_RUS, LANG_BASIC_ENG);
        }
    }
};
