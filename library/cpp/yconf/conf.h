#pragma once

#include <library/cpp/logger/all.h>

#include <util/str_stl.h>
#include <library/cpp/charset/ci_string.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/stack.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <library/cpp/string_utils/parse_vector/vector_parser.h>
#include <util/generic/yexception.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>
#include <util/generic/noncopyable.h>

class TSectionDesc;

class TYandexConfig: public TSimpleRefCount<TYandexConfig>, TNonCopyable {
public:
    class Directives;
    typedef TMap<const char*, const char*, ci_less> SectionAttrs;
    struct Section;
    typedef TMultiMap<TCiString, Section*> TSectionsMap;

    struct Section {
        const char* Name;
        SectionAttrs Attrs;
        Directives* Cookie;
        Section* Parent;
        Section* Next;
        Section* Child;
        bool Owner;
        TSectionDesc* Desc;

        Section()
            : Name("")
            , Cookie(nullptr)
            , Parent(nullptr)
            , Next(nullptr)
            , Child(nullptr)
            , Owner(false)
            , Desc(nullptr)
        {
        }

        Directives& GetDirectives() {
            Y_ASSERT(Cookie);
            return *Cookie;
        }

        const Directives& GetDirectives() const {
            Y_ASSERT(Cookie);
            return *Cookie;
        }

        bool Parsed() const {
            return Cookie != nullptr;
        }

        TSectionsMap GetAllChildren() const;
    };

public:
    TYandexConfig()
        : FileData(nullptr)
    {
        Clear();
    }
    virtual ~TYandexConfig() {
        Clear();
    }
    [[nodiscard]] bool Read(const TString& path);
    [[nodiscard]] bool ReadMemory(const char* buffer, const char* configPath = nullptr);
    [[nodiscard]] bool ReadMemory(const TStringBuf& buffer, const char* configPath = nullptr);
    [[nodiscard]] bool Parse(const TString& path, bool process_directives = true);
    [[nodiscard]] bool ParseMemory(const char* buffer, bool process_directives = true, const char* configPath = nullptr);
    [[nodiscard]] bool ParseMemory(const TStringBuf& buffer, bool processDirectives = true, const char* configPath = nullptr);
    [[nodiscard]] bool ParseSection(const char* SecName, const char* idname = nullptr, const char* idvalue = nullptr);
    void AddSection(Section* sec);
    void Clear();
    void ReportError(const char* ptr, const char* err, bool warning = false);
    void ReportError(const char* ptr, bool warning, const char* format, ...) Y_PRINTF_FORMAT(4, 5);
    void PrintErrors(TLog* Log);
    void PrintErrors(TString& Err);

    template <typename TLogWriter>
    void PrintErrors(TLogWriter& writer) {
        for (const auto& s : Errors) {
            writer() << "In '" << ConfigPath << "': " << s << '\n';
        }
        Errors.clear();
    }

    Section* GetFirstChild(const char* Name, Section* CurSection = nullptr);
    const char* GetConfigPath() const {
        return ConfigPath.data();
    }
    Section* GetRootSection() {
        Y_ASSERT(!AllSections.empty());
        return AllSections[0];
    }
    const Section* GetRootSection() const {
        Y_ASSERT(!AllSections.empty());
        return AllSections[0];
    }
    void PrintConfig(IOutputStream& os) const;
    static void PrintSectionConfig(const TYandexConfig::Section* section, IOutputStream& os, bool printNextSection = true);

protected:
    //the followind three functions return 'false' only for fatal errors to break the parsing
    virtual bool AddKeyValue(Section& sec, const char* key, const char* value);
    virtual bool OnBeginSection(Section& sec); //keep sec.Cookie==0 to skip the section
    virtual bool OnEndSection(Section& sec);

private:
    bool PrepareLines();
    void ProcessComments();
    bool ProcessRoot(bool process_directives);
    bool ProcessAll(bool process_directives);
    bool ProcessBeginSection();
    bool ProcessEndSection();
    bool ProcessDirective();
    void ProcessLineBreak(char*& LineBreak, char toChange);
    bool FindEndOfSection(const char* SecName, const char* begin, char*& endsec, char*& endptr);

private:
    char* FileData;
    ui32 Len;
    char* CurrentMemoryPtr;
    TStack<Section*> CurSections;
    TVector<Section*> AllSections;
    TVector<TString> Errors;
    TVector<const char*> EndLines;
    TString ConfigPath;
};

class TYandexConfig::Directives: public TMap<TCiString, const char*, std::less<>> {
public:
    Directives(bool isStrict)
        : strict(isStrict)
    {
    }

    Directives()
        : strict(true)
    {
    }

    virtual ~Directives() = default;

    bool IsStrict() const {
        return strict;
    }

    bool AddKeyValue(const TString& key, const char* value);

    bool GetValue(TStringBuf key, TString& value) const;
    bool GetNonEmptyValue(TStringBuf key, TString& value) const;
    bool GetValue(TStringBuf key, bool& value) const;

    template <class T>
    inline bool GetValue(TStringBuf key, T& value) const {
        TString tmp;

        if (GetValue(key, tmp)) {
            value = FromString<T>(tmp);

            return true;
        }

        return false;
    }

    template <class T>
    inline T Value(TStringBuf key, T def = T()) const {
        GetValue(key, def);
        return def;
    }

    template <class T, class TDelim = char, bool emptyOK = true>
    bool TryFillArray(TStringBuf key, TVector<T>& result, const TDelim delim = ',') const {
        auto it = find(key);

        if (it != end() && (*it).second != nullptr) {
            TVector<T> localResult;
            if (!TryParseStringToVector((*it).second, localResult, delim)) {
                return false;
            } else {
                std::swap(localResult, result);
                return true;
            }
        } else {
            if (emptyOK) {
                result.clear();
            }
        }

        return emptyOK;
    }

    bool FillArray(TStringBuf key, TVector<TString>& values) const;

    void Clear();

    void declare(const char* directive_name) {
        insert(value_type(directive_name, nullptr));
    }

    virtual bool CheckOnEnd(TYandexConfig& yc, TYandexConfig::Section& sec);

protected:
    bool strict;
};

#define DECLARE_CONFIG(ConfigClass)                 \
    class ConfigClass: public TYandexConfig {       \
    public:                                         \
        ConfigClass()                               \
            : TYandexConfig() {                     \
        }                                           \
                                                    \
    protected:                                      \
        virtual bool OnBeginSection(Section& sec);  \
                                                    \
    private:                                        \
        ConfigClass(const ConfigClass&);            \
        ConfigClass& operator=(const ConfigClass&); \
    };

#define DECLARE_SECTION(SectionClass)                       \
    class SectionClass: public TYandexConfig::Directives {  \
    public:                                                 \
        SectionClass();                                     \
    };

#define DECLARE_SECTION_CHECK(SectionClass)                              \
    class SectionClass: public TYandexConfig::Directives {               \
    public:                                                              \
        SectionClass();                                                  \
        bool CheckOnEnd(TYandexConfig& yc, TYandexConfig::Section& sec); \
    };

#define BEGIN_CONFIG(ConfigClass)                       \
    bool ConfigClass::OnBeginSection(Section& sec) {    \
        if (sec.Parent == &sec) /* it's root */ {       \
            assert(*sec.Name == 0);                     \
            /* do not allow any directives at root */   \
            sec.Cookie = new TYandexConfig::Directives; \
            sec.Owner = true;                           \
            return true;                                \
        }

#define BEGIN_TOPSECTION2(SectionName, DirectivesClass)     \
    if (*sec.Parent->Name == 0) { /* it's placed at root */ \
        if (stricmp(sec.Name, #SectionName) == 0) {         \
            sec.Cookie = new DirectivesClass;               \
            sec.Owner = true;                               \
            return true;                                    \
        }                                                   \
    } else if (stricmp(sec.Parent->Name, #SectionName) == 0) {
#define BEGIN_SUBSECTION(SectionName, SubSectionName) \
    if (stricmp(sec.Parent->Name, #SubSectionName) == 0 && stricmp(sec.Parent->Parent->Name, #SectionName) == 0) {
#define SUBSECTION2(SubSectionName, DirectivesClass) \
    if (stricmp(sec.Name, #SubSectionName) == 0) {   \
        sec.Cookie = new DirectivesClass;            \
        sec.Owner = true;                            \
        return true;                                 \
    }

#define FAKESECTION(SubSectionName)                \
    if (stricmp(sec.Name, #SubSectionName) == 0) { \
        Y_ASSERT(sec.Cookie == 0);                 \
        return true;                               \
    }

#define END_SECTION() \
    }

#define END_CONFIG()                                                                              \
    if (!sec.Parent->Parsed())                                                                    \
        return true;                                                                              \
    ReportError(sec.Name, true, "section \'%s\' not allowed here and will be ignored", sec.Name); \
    return true;                                                                                  \
    }

#define SUBSECTION(SectionName) SUBSECTION2(SectionName, SectionName)
#define BEGIN_TOPSECTION(SectionName) BEGIN_TOPSECTION2(SectionName, SectionName)

#define BEGIN_SECTION(SectionClass) \
    SectionClass::SectionClass() {
#define DEFINE_SECTION(SectionClass)                        \
    class SectionClass: public TYandexConfig::Directives {  \
    public:                                                 \
        SectionClass() {
#define DIRECTIVE(DirectiveName) declare(#DirectiveName);
#define END_DEFINE_SECTION \
    }                      \
    }                      \
    ;
#define END_DEFINE_SECTION_CHECK                                     \
    }                                                                \
    bool CheckOnEnd(TYandexConfig& yc, TYandexConfig::Section& sec); \
    }                                                                \
    ;

#define DEFINE_INDEFINITE_SECTION(SectionClass)             \
    class SectionClass: public TYandexConfig::Directives {  \
    public:                                                 \
        SectionClass() {                                    \
            strict = false;                                 \
        }                                                   \
    };

#define BEGIN_SECTION_CHECK(SectionClass)                                           \
    bool SectionClass::CheckOnEnd(TYandexConfig& yc, TYandexConfig::Section& sec) { \
        (void)yc;                                                                   \
        (void)sec;                                                                  \
        SectionClass& type = *this;                                                 \
        (void)type;

#define DIR_ABSENT(DirectiveName) (type[#DirectiveName] == 0)
#define DIR_ARG_ABSENT(DirectiveName) (type[#DirectiveName] == 0 || *(type[#DirectiveName]) == 0)
#define DIR_PRESENT(DirectiveName) (type[#DirectiveName] != 0)
#define DIR_ARG_PRESENT(DirectiveName) (type[#DirectiveName] != 0 && *(type[#DirectiveName]) != 0)

#define DIRECTIVE_MUSTBE(DirectiveName)                                                         \
    if (DIR_ARG_ABSENT(DirectiveName)) {                                                        \
        yc.ReportError(sec.Name, true,                                                          \
                       "Section \'%s\' must include directive \'%s\'. Section will be ignored", \
                       sec.Name, #DirectiveName);                                               \
        return false;                                                                           \
    }

#define DIRECTIVE_ATLEAST(DirectiveName1, DirectiveName2)                                                  \
    if (DIR_ARG_ABSENT(DirectiveName1) && DIR_ARG_ABSENT(DirectiveName2)) {                                \
        yc.ReportError(sec.Name, true,                                                                     \
                       "Section \'%s\' must include directives \'%s\' or \'%s\'. Section will be ignored", \
                       sec.Name, #DirectiveName1, #DirectiveName2);                                        \
        return false;                                                                                      \
    }

#define DIRECTIVE_BOTH(DirectiveName1, DirectiveName2)                                                                                            \
    if (DIR_ARG_ABSENT(DirectiveName1) && DIR_ARG_PRESENT(DirectiveName2) || DIR_ARG_ABSENT(DirectiveName2) && DIR_ARG_PRESENT(DirectiveName1)) { \
        yc.ReportError(sec.Name, true,                                                                                                            \
                       "Section \'%s\' must include directives \'%s\' and \'%s\' simultaneously. Section will be ignored",                        \
                       sec.Name, #DirectiveName1, #DirectiveName2);                                                                               \
        return false;                                                                                                                             \
    }

#define END_SECTION_CHECK() \
    return true;            \
    }

class config_exception: public yexception {
public:
    config_exception(const char* fp) {
        filepoint = fp;
    }
    const char* where() const noexcept {
        return filepoint;
    }

private:
    const char* filepoint;
};

#define DEFINE_UNSTRICT_SECTION(SectionClasse)              \
    class SectionClasse                                     \
       : public TYandexConfig::Directives {                 \
    public:                                                 \
        SectionClasse(const TYandexConfig::Directives& obj) \
            : TYandexConfig::Directives(obj) {              \
            strict = false;                                 \
        }                                                   \
        SectionClasse() {                                   \
            strict = false;

DEFINE_UNSTRICT_SECTION(AnyDirectives)
END_DEFINE_SECTION

#define EMBEDDED_CONFIG(SectionName)                                \
    if (sec.Parent != &sec) /* not root not placed at root */ {     \
        Section* parent = sec.Parent;                               \
        while (*parent->Name != 0) { /* any child of SectionName */ \
            if (stricmp(parent->Name, #SectionName) == 0) {         \
                sec.Cookie = new AnyDirectives();                   \
                sec.Owner = true;                                   \
                return true;                                        \
            }                                                       \
            parent = parent->Parent;                                \
        }                                                           \
    }

#define ONE_EMBEDDED_CONFIG(SectionName)                            \
    if (sec.Parent != &sec) /* not root not placed at root */ {     \
        Section* parent = sec.Parent;                               \
        while (*parent->Name != 0) { /* any child of SectionName */ \
            if (stricmp(parent->Name, #SectionName) == 0) {         \
                if (!parent->Child->Next) {                         \
                    sec.Cookie = new AnyDirectives();               \
                    sec.Owner = true;                               \
                    return true;                                    \
                } else {                                            \
                    break;                                          \
                }                                                   \
            }                                                       \
            parent = parent->Parent;                                \
        }                                                           \
    }
