#pragma once

#include "doccodes.h"

#include <util/charset/recode_result.h>
#include <util/charset/unidata.h> // all wchar32 functions
#include <util/charset/utf8.h>
#include <util/generic/string.h>
#include <util/generic/ylimits.h>
#include <util/generic/yexception.h>
#include <util/system/yassert.h>
#include <util/system/defaults.h>

#include <cctype>

struct CodePage;
struct Recoder;
struct Encoder;

/*****************************************************************\
*                    struct CodePage                              *
\*****************************************************************/
struct CodePage {
    ECharset CPEnum;       // int MIBEnum;
    const char* Names[30]; // name[0] -- preferred mime-name
    wchar32 unicode[256];
    const char* DefaultChar; //[CCL_NUM]

    bool IsLower(unsigned char ch) const {
        return ::IsLower(unicode[ch]);
    }
    bool IsUpper(unsigned char ch) const {
        return ::IsUpper(unicode[ch]);
    }
    bool IsAlpha(unsigned char ch) const {
        return ::IsAlpha(unicode[ch]);
    }
    bool IsDigit(unsigned char ch) const {
        return ::IsDigit(unicode[ch]);
    }
    bool IsXdigit(unsigned char ch) const {
        return ::IsXdigit(unicode[ch]);
    }
    bool IsAlnum(unsigned char ch) const {
        return ::IsAlnum(unicode[ch]);
    }
    bool IsSpace(unsigned char ch) const {
        return ::IsSpace(unicode[ch]);
    }
    bool IsPunct(unsigned char ch) const {
        return ::IsPunct(unicode[ch]);
    }
    bool IsCntrl(unsigned char ch) const {
        return ::IsCntrl(unicode[ch]);
    }
    bool IsGraph(unsigned char ch) const {
        return ::IsGraph(unicode[ch]);
    }
    bool IsPrint(unsigned char ch) const {
        return ::IsPrint(unicode[ch]);
    }
    bool IsComposed(unsigned char ch) const {
        return ::IsComposed(unicode[ch]);
    }

    // return pointer to char after the last char
    char* ToLower(const char* begin, const char* end, char* to) const;
    char* ToLower(const char* begin, char* to) const;

    // return pointer to char after the last char
    char* ToUpper(const char* begin, const char* end, char* to) const;
    char* ToUpper(const char* begin, char* to) const;

    int stricmp(const char* s1, const char* s2) const;
    int strnicmp(const char* s1, const char* s2, size_t len) const;

    inline unsigned char ToUpper(unsigned char ch) const;
    inline unsigned char ToLower(unsigned char ch) const;
    inline unsigned char ToTitle(unsigned char ch) const;

    inline int ToDigit(unsigned char ch) const {
        return ::ToDigit(unicode[ch]);
    }

    static void Initialize();

    inline bool SingleByteCodepage() const {
        return DefaultChar != nullptr;
    }
    inline bool NativeCodepage() const {
        return SingleByteCodepage() || CPEnum == CODES_UTF8;
    }
};

class TCodePageHash;

namespace NCodepagePrivate {
    class TCodepagesMap {
    private:
        static const int DataShift = 2;
        static const int DataSize = CODES_MAX + DataShift;
        const CodePage* Data[DataSize];

    private:
        inline const CodePage* GetPrivate(ECharset e) const {
            Y_ASSERT(e + DataShift >= 0 && e + DataShift < DataSize);
            return Data[e + DataShift];
        }

        void SetData(const CodePage* cp);

    public:
        TCodepagesMap();

        inline const CodePage* Get(ECharset e) const {
            const CodePage* res = GetPrivate(e);
            if (!res->SingleByteCodepage()) {
                ythrow yexception() << "CodePage (" << (int)e << ") structure can only be used for single byte encodings";
            }

            return res;
        }

        inline bool SingleByteCodepage(ECharset e) const {
            return GetPrivate(e)->SingleByteCodepage();
        }
        inline bool NativeCodepage(ECharset e) const {
            return GetPrivate(e)->NativeCodepage();
        }
        inline const char* NameByCharset(ECharset e) const {
            return GetPrivate(e)->Names[0];
        }

        static const TCodepagesMap& Instance();

        friend class ::TCodePageHash;
    };

    inline bool NativeCodepage(ECharset e) {
        return ::NCodepagePrivate::TCodepagesMap::Instance().NativeCodepage(e);
    }
}

inline bool SingleByteCodepage(ECharset e) {
    return ::NCodepagePrivate::TCodepagesMap::Instance().SingleByteCodepage(e);
}

inline bool ValidCodepage(ECharset e) {
    return e >= 0 && e < CODES_MAX;
}

inline const CodePage* CodePageByCharset(ECharset e) {
    return ::NCodepagePrivate::TCodepagesMap::Instance().Get(e);
}

ECharset CharsetByName(TStringBuf name);

// Same as CharsetByName, but throws yexception() if name is invalid
ECharset CharsetByNameOrDie(TStringBuf name);

inline ECharset CharsetByCodePage(const CodePage* CP) {
    return CP->CPEnum;
}

inline const char* NameByCharset(ECharset e) {
    return ::NCodepagePrivate::TCodepagesMap::Instance().NameByCharset(e);
}

inline const char* NameByCharsetSafe(ECharset e) {
    if (CODES_UNKNOWN < e && e < CODES_MAX)
        return ::NCodepagePrivate::TCodepagesMap::Instance().NameByCharset(e);
    else
        ythrow yexception() << "unknown encoding: " << (int)e;
}

inline const char* NameByCodePage(const CodePage* CP) {
    return CP->Names[0];
}

inline const CodePage* CodePageByName(const char* name) {
    ECharset code = CharsetByName(name);
    if (code == CODES_UNKNOWN)
        return nullptr;

    return CodePageByCharset(code);
}

ECharset EncodingHintByName(const char* name);

/*****************************************************************\
*                    struct Encoder                               *
\*****************************************************************/
struct Encoder {
    char* Table[256];
    const char* DefaultChar;

    inline char Code(wchar32 ch) const {
        if (ch > 0xFFFF)
            return 0;
        return (unsigned char)Table[(ch >> 8) & 255][ch & 255];
    }

    inline char Tr(wchar32 ch) const {
        char code = Code(ch);
        if (code == 0 && ch != 0)
            code = DefaultChar[NUnicode::CharType(ch)];
        Y_ASSERT(code != 0 || ch == 0);
        return code;
    }

    inline unsigned char operator[](wchar32 ch) const {
        return Tr(ch);
    }

    void Tr(const wchar32* in, char* out, size_t len) const;
    void Tr(const wchar32* in, char* out) const;
    char* DefaultPlane;
};

/*****************************************************************\
*                    struct Recoder                               *
\*****************************************************************/
struct Recoder {
    unsigned char Table[257];

    void Create(const CodePage& source, const CodePage& target);
    void Create(const CodePage& source, const Encoder* wideTarget);

    void Create(const CodePage& page, wchar32 (*mapper)(wchar32));
    void Create(const CodePage& page, const Encoder* widePage, wchar32 (*mapper)(wchar32));

    inline unsigned char Tr(unsigned char c) const {
        return Table[c];
    }
    inline unsigned char operator[](unsigned char c) const {
        return Table[c];
    }
    void Tr(const char* in, char* out, size_t len) const;
    void Tr(const char* in, char* out) const;
    void Tr(char* in_out, size_t len) const;
    void Tr(char* in_out) const;
};

extern const struct Encoder& WideCharToYandex;

const Encoder& EncoderByCharset(ECharset enc);

namespace NCodepagePrivate {
    class TCodePageData {
    private:
        static const CodePage* const AllCodePages[];

        static const Recoder rcdr_to_yandex[];
        static const Recoder rcdr_from_yandex[];
        static const Recoder rcdr_to_lower[];
        static const Recoder rcdr_to_upper[];
        static const Recoder rcdr_to_title[];

        static const Encoder* const EncodeTo[];

        friend struct ::CodePage;
        friend class TCodepagesMap;
        friend RECODE_RESULT _recodeToYandex(ECharset, const char*, char*, size_t, size_t, size_t&, size_t&);
        friend RECODE_RESULT _recodeFromYandex(ECharset, const char*, char*, size_t, size_t, size_t&, size_t&);
        friend const Encoder& ::EncoderByCharset(ECharset enc);
    };
}

inline const Encoder& EncoderByCharset(ECharset enc) {
    if (!SingleByteCodepage(enc)) {
        ythrow yexception() << "Encoder structure can only be used for single byte encodings";
    }

    return *NCodepagePrivate::TCodePageData::EncodeTo[enc];
}

inline unsigned char CodePage::ToUpper(unsigned char ch) const {
    return NCodepagePrivate::TCodePageData::rcdr_to_upper[CPEnum].Table[ch];
}
inline unsigned char CodePage::ToLower(unsigned char ch) const {
    return NCodepagePrivate::TCodePageData::rcdr_to_lower[CPEnum].Table[ch];
}
inline unsigned char CodePage::ToTitle(unsigned char ch) const {
    return NCodepagePrivate::TCodePageData::rcdr_to_title[CPEnum].Table[ch];
}

extern const CodePage& csYandex;

/// these functions change (lowers) [end] position in case of utf-8
/// null character is NOT assumed or written at [*end]
void DecodeUnknownPlane(wchar16* start, wchar16*& end, const ECharset enc4unk);
void DecodeUnknownPlane(wchar32* start, wchar32*& end, const ECharset enc4unk);

inline void ToLower(char* s, size_t n, const CodePage& cp = csYandex) {
    char* const e = s + n;
    for (; s != e; ++s)
        *s = cp.ToLower(*s);
}

inline void ToUpper(char* s, size_t n, const CodePage& cp = csYandex) {
    char* const e = s + n;
    for (; s != e; ++s)
        *s = cp.ToUpper(*s);
}

inline TString ToLower(TString s, const CodePage& cp, size_t pos = 0, size_t n = TString::npos) {
    s.Transform([&cp](size_t, char c) { return cp.ToLower(c); }, pos, n);
    return s;
}

inline TString ToUpper(TString s, const CodePage& cp, size_t pos = 0, size_t n = TString::npos) {
    s.Transform([&cp](size_t, char c) { return cp.ToUpper(c); }, pos, n);
    return s;
}

inline TString ToTitle(TString s, const CodePage& cp, size_t pos = 0, size_t n = TString::npos) {
    s.Transform(
        [pos, &cp](size_t i, char c) {
            return i == pos ? cp.ToTitle(c) : cp.ToLower(c);
        },
        pos,
        n);
    return s;
}
