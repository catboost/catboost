#include "scripts.h"

#include <library/cpp/digest/lower_case/hash_ops.h>

#include <util/generic/hash.h>
#include <util/generic/singleton.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/system/defaults.h>

#include <array>

namespace {
    struct TScriptNameAndEnum {
        EScript Script;
        const char* EnglishName;
        const char* IsoName;
    };

    const TScriptNameAndEnum ScriptNameAndEnum[] = {
        {SCRIPT_UNKNOWN, "Unknown", "Zzzz"},
        {SCRIPT_LATIN, "Latin", "Latn"},
        {SCRIPT_CYRILLIC, "Cyrillic", "Cyrl"},

        {SCRIPT_GREEK, "Greek", "Grek"},
        {SCRIPT_ARABIC, "Arabic", "Arab"},
        {SCRIPT_HEBREW, "Hebrew", "Hebr"},
        {SCRIPT_ARMENIAN, "Armenian", "Armn"},
        {SCRIPT_GEORGIAN, "Georgian", "Geor"},

        {SCRIPT_HAN, "Han", "Hans"}, // We use more common Simpliied variant (as opposed to Traditional 'Hant')
        {SCRIPT_KATAKANA, "Katakana", "Kana"},
        {SCRIPT_HIRAGANA, "Hiragana", "Hira"},
        {SCRIPT_HANGUL, "Hangul", "Hang"},

        {SCRIPT_DEVANAGARI, "Devanagari", "Deva"},
        {SCRIPT_BENGALI, "Bengali", "Beng"},
        {SCRIPT_GUJARATI, "Gujarati", "Gujr"},
        {SCRIPT_GURMUKHI, "Gurmukhi", "Guru"},
        {SCRIPT_KANNADA, "Kannada", "Knda"},
        {SCRIPT_MALAYALAM, "Malayalam", "Mlym"},
        {SCRIPT_ORIYA, "Oriya", "Orya"},
        {SCRIPT_TAMIL, "Tamil", "Taml"},
        {SCRIPT_TELUGU, "Telugu", "Telu"},
        {SCRIPT_THAANA, "Thaana", "Thaa"},
        {SCRIPT_SINHALA, "Sinhala", "Sinh"},

        {SCRIPT_MYANMAR, "Myanmar", "Mymr"},
        {SCRIPT_THAI, "Thai", "Thai"},
        {SCRIPT_LAO, "Lao", "Laoo"},
        {SCRIPT_KHMER, "Khmer", "Khmr"},
        {SCRIPT_TIBETAN, "Tibetan", "Tibt"},
        {SCRIPT_MONGOLIAN, "Mongolian", "Mong"},

        {SCRIPT_ETHIOPIC, "Ethiopic", "Ethi"},
        {SCRIPT_RUNIC, "Runic", "Runr"},
        {SCRIPT_COPTIC, "Coptic", "Copt"},
        {SCRIPT_SYRIAC, "Syriac", "Syrc"},

        {SCRIPT_OTHER, "Other", "Zyyy"},
    };

    static_assert(static_cast<size_t>(SCRIPT_MAX) == Y_ARRAY_SIZE(ScriptNameAndEnum), "Size doesn't match");

    class TScriptsMap {
    private:
        static const char* const EMPTY_NAME;

        using TNamesHash = THashMap<TStringBuf, EScript, TCIOps, TCIOps>;
        TNamesHash Hash;

        using TNamesArray = std::array<const char*, static_cast<size_t>(SCRIPT_MAX)>;
        TNamesArray IsoNames;
        TNamesArray FullNames;

    private:
        void AddNameToHash(const TStringBuf& name, EScript script) {
            if (Hash.find(name) != Hash.end()) {
                Y_ASSERT(Hash.find(name)->second == script);
                return;
            }

            Hash[name] = script;
        }

        void AddName(const char* name, EScript script, TNamesArray& names) {
            if (name == nullptr || strlen(name) == 0)
                return;

            Y_ASSERT(names[script] == EMPTY_NAME);
            names[script] = name;

            AddNameToHash(name, script);
        }

    public:
        TScriptsMap() {
            IsoNames.fill(EMPTY_NAME);
            FullNames.fill(EMPTY_NAME);

            for (const auto& val : ScriptNameAndEnum) {
                EScript script = val.Script;

                AddName(val.IsoName, script, IsoNames);
                AddName(val.EnglishName, script, FullNames);
            }
        }

    public:
        inline EScript ScriptByName(const TStringBuf& name, EScript def) const {
            if (!name)
                return def;

            TNamesHash::const_iterator i = Hash.find(name);
            if (i == Hash.end()) {
                return def;
            }

            return i->second;
        }

        inline const char* FullNameByScript(EScript script) const {
            if (script < 0 || static_cast<size_t>(script) >= FullNames.size())
                return nullptr;

            return FullNames[script];
        }

        inline const char* IsoNameByScript(EScript script) const {
            if (script < 0 || static_cast<size_t>(script) >= IsoNames.size())
                return nullptr;

            return IsoNames[script];
        }
    };
}

const char* const TScriptsMap::EMPTY_NAME = "";

const char* FullNameByScript(EScript script) {
    return Singleton<TScriptsMap>()->FullNameByScript(script);
}

const char* IsoNameByScript(EScript script) {
    return Singleton<TScriptsMap>()->IsoNameByScript(script);
}

EScript ScriptByName(const TStringBuf& name) {
    return Singleton<TScriptsMap>()->ScriptByName(name, SCRIPT_UNKNOWN);
}

EScript ScriptByNameOrDie(const TStringBuf& name) {
    EScript result = ScriptByName(name);
    if (result == SCRIPT_UNKNOWN) {
        ythrow yexception() << "ScriptByNameOrDie: invalid script '" << name << "'";
    }
    return result;
}
