#include "ci_string.h"
#include "wide.h"
#include "recyr.hh"
#include "codepage.h"

#include <util/string/cast.h>
#include <util/string/subst.h>
#include <util/string/util.h>
#include <util/system/hi_lo.h>
#include <util/system/yassert.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/hash_set.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/memory/pool.h>

#include <cstring>

#include <ctype.h>

using namespace NCodepagePrivate;

void Recoder::Create(const CodePage& source, const CodePage& target) {
    const Encoder* wideTarget = &EncoderByCharset(target.CPEnum);
    Create(source, wideTarget);
}
void Recoder::Create(const CodePage& page, wchar32 (*mapfunc)(wchar32)) {
    const Encoder* widePage = &EncoderByCharset(page.CPEnum);
    Create(page, widePage, mapfunc);
}

template <class T, class T1>
static inline T1 Apply(T b, T e, T1 to, const Recoder& mapper) {
    while (b != e) {
        *to++ = mapper.Table[(unsigned char)*b++];
    }

    return to;
}

template <class T, class T1>
static inline T1 Apply(T b, T1 to, const Recoder& mapper) {
    while (*b != 0) {
        *to++ = mapper.Table[(unsigned char)*b++];
    }

    return to;
}

char* CodePage::ToLower(const char* b, const char* e, char* to) const {
    return Apply(b, e, to, TCodePageData::rcdr_to_lower[CPEnum]);
}
char* CodePage::ToLower(const char* b, char* to) const {
    return Apply(b, to, TCodePageData::rcdr_to_lower[CPEnum]);
}

char* CodePage::ToUpper(const char* b, const char* e, char* to) const {
    return Apply(b, e, to, TCodePageData::rcdr_to_upper[CPEnum]);
}
char* CodePage::ToUpper(const char* b, char* to) const {
    return Apply(b, to, TCodePageData::rcdr_to_upper[CPEnum]);
}

int CodePage::stricmp(const char* dst, const char* src) const {
    unsigned char f, l;
    do {
        f = ToLower(*dst++);
        l = ToLower(*src++);
    } while (f && (f == l));
    return f - l;
}

int CodePage::strnicmp(const char* dst, const char* src, size_t len) const {
    unsigned char f, l;
    if (len) {
        do {
            f = ToLower(*dst++);
            l = ToLower(*src++);
        } while (--len && f && (f == l));
        return f - l;
    }
    return 0;
}

static const CodePage UNSUPPORTED_CODEPAGE = {
    CODES_UNSUPPORTED,
    {
        "unsupported",
    },
    {},
    nullptr,
};

static const CodePage UNKNOWN_CODEPAGE = {
    CODES_UNKNOWN,
    {
        "unknown",
    },
    {},
    nullptr,
};

void NCodepagePrivate::TCodepagesMap::SetData(const CodePage* cp) {
    Y_ASSERT(cp);
    int code = static_cast<int>(cp->CPEnum) + DataShift;

    Y_ASSERT(code >= 0 && code < DataSize);
    Y_ASSERT(Data[code] == nullptr);

    Data[code] = cp;
}

NCodepagePrivate::TCodepagesMap::TCodepagesMap() {
    memset(Data, 0, sizeof(const CodePage*) * DataSize);
    SetData(&UNSUPPORTED_CODEPAGE);
    SetData(&UNKNOWN_CODEPAGE);

    for (size_t i = 0; i != CODES_MAX; ++i) {
        SetData(TCodePageData::AllCodePages[i]);
    }
}

const NCodepagePrivate::TCodepagesMap& NCodepagePrivate::TCodepagesMap::Instance() {
    return *Singleton<NCodepagePrivate::TCodepagesMap>();
}

class TCodePageHash {
private:
    using TData = THashMap<TStringBuf, ECharset, ci_hash, ci_equal_to>;

    TData Data;
    TMemoryPool Pool;

private:
    inline void AddNameWithCheck(const TString& name, ECharset code) {
        if (Data.find(name.c_str()) == Data.end()) {
            Data.insert(TData::value_type(Pool.Append(name.data(), name.size() + 1), code));
        } else {
            Y_ASSERT(Data.find(name.c_str())->second == code);
        }
    }

    inline void AddName(const TString& name, ECharset code) {
        AddNameWithCheck(name, code);

        TString temp = name;
        RemoveAll(temp, '-');
        RemoveAll(temp, '_');
        AddNameWithCheck(temp, code);

        temp = name;
        SubstGlobal(temp, '-', '_');
        AddNameWithCheck(temp, code);

        temp = name;
        SubstGlobal(temp, '_', '-');
        AddNameWithCheck(temp, code);
    }

public:
    inline TCodePageHash()
        : Pool(20 * 1024) /* Currently used: 17KB. */
    {
        TString xPrefix = "x-";
        const char* name;

        for (size_t i = 0; i != CODES_MAX; ++i) {
            ECharset e = static_cast<ECharset>(i);
            const CodePage* page = Singleton<NCodepagePrivate::TCodepagesMap>()->GetPrivate(e);

            AddName(ToString(static_cast<int>(i)), e);

            for (size_t j = 0; (name = page->Names[j]) != nullptr && name[0]; ++j) {
                AddName(name, e);

                AddName(xPrefix + name, e);
            }
        }
    }

    inline ECharset CharsetByName(TStringBuf name) {
        if (!name)
            return CODES_UNKNOWN;

        TData::const_iterator it = Data.find(name);
        if (it == Data.end())
            return CODES_UNKNOWN;

        return it->second;
    }
};

ECharset CharsetByName(TStringBuf name) {
    return Singleton<TCodePageHash>()->CharsetByName(name);
}

ECharset CharsetByNameOrDie(TStringBuf name) {
    ECharset result = CharsetByName(name);
    if (result == CODES_UNKNOWN)
        ythrow yexception() << "CharsetByNameOrDie: unknown charset '" << name << "'";
    return result;
}

template <typename TxChar>
static inline RECODE_RESULT utf8_read_rune_from_unknown_plane(TxChar& rune, size_t& rune_len, const TxChar* s, const TxChar* end) {
    if ((*s & 0xFF00) != 0xF000) {
        rune_len = 1;
        rune = *s;
        return RECODE_OK;
    }

    rune_len = 0;

    size_t _len = UTF8RuneLen((unsigned char)(*s));
    if (s + _len > end)
        return RECODE_EOINPUT; //[EOINPUT]
    if (_len == 0)
        return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in first byte

    wchar32 _rune = (ui8)(*s++); //[00000000 0XXXXXXX]
    if (_len > 1) {
        _rune &= UTF8LeadByteMask(_len);
        wchar32 ch = *s++;
        if ((ch & 0xFFC0) != 0xF080)
            return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in second byte
        _rune <<= 6;
        _rune |= ch & 0x3F; //[00000XXX XXYYYYYY]
        if (_len > 2) {
            ch = *s++;
            if ((ch & 0xFFC0) != 0xF080)
                return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in third byte
            _rune <<= 6;
            _rune |= ch & 0x3F; //[XXXXYYYY YYZZZZZZ]
            if (_len > 3) {
                ch = *s;
                if ((ch & 0xFFC0) != 0xF080)
                    return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in fourth byte
                _rune <<= 6;
                _rune |= ch & 0x3F; //[XXXYY YYYYZZZZ ZZQQQQQQ]
            }
        }
    }
    rune_len = _len;
    if (_rune > Max<TxChar>())
        rune = ' '; // maybe put sequence
    else
        rune = TxChar(_rune);
    return RECODE_OK;
}

template <typename TxChar>
void DoDecodeUnknownPlane(TxChar* str, TxChar*& ee, const ECharset enc) {
    TxChar* e = ee;
    if (SingleByteCodepage(enc)) {
        const CodePage* cp = CodePageByCharset(enc);
        for (TxChar* s = str; s < e; s++) {
            if (Hi8(Lo16(*s)) == 0xF0)
                *s = (TxChar)cp->unicode[Lo8(Lo16(*s))]; // NOT mb compliant
        }
    } else if (enc == CODES_UTF8) {
        TxChar* s;
        TxChar* d;

        for (s = d = str; s < e;) {
            size_t l = 0;

            if (utf8_read_rune_from_unknown_plane(*d, l, s, e) == RECODE_OK) {
                d++, s += l;
            } else {
                *d++ = BROKEN_RUNE;
                ++s;
            }
        }
        e = d;
    } else if (enc == CODES_UNKNOWN) {
        for (TxChar* s = str; s < e; s++) {
            if (Hi8(Lo16(*s)) == 0xF0)
                *s = Lo8(Lo16(*s));
        }
    } else {
        Y_ASSERT(!SingleByteCodepage(enc));

        TxChar* s = str;
        TxChar* d = str;

        TVector<char> buf;

        size_t read = 0;
        size_t written = 0;
        for (; s < e; ++s) {
            if (Hi8(Lo16(*s)) == 0xF0) {
                buf.push_back(Lo8(Lo16(*s)));
            } else {
                if (!buf.empty()) {
                    if (RecodeToUnicode(enc, buf.data(), d, buf.size(), e - d, read, written) == RECODE_OK) {
                        Y_ASSERT(read == buf.size());
                        d += written;
                    } else { // just copying broken symbols
                        Y_ASSERT(buf.size() <= static_cast<size_t>(e - d));
                        Copy(buf.data(), buf.size(), d);
                        d += buf.size();
                    }
                    buf.clear();
                }
                *d++ = *s;
            }
        }
    }
    ee = e;
}

void DecodeUnknownPlane(wchar16* str, wchar16*& ee, const ECharset enc) {
    DoDecodeUnknownPlane(str, ee, enc);
}
void DecodeUnknownPlane(wchar32* str, wchar32*& ee, const ECharset enc) {
    DoDecodeUnknownPlane(str, ee, enc);
}

namespace {
    class THashSetType: public THashSet<TString> {
    public:
        inline void Add(const TString& s) {
            insert(s);
        }

        inline bool Has(const TString& s) const noexcept {
            return find(s) != end();
        }
    };
}

class TWindowsPrefixesHashSet: public THashSetType {
public:
    inline TWindowsPrefixesHashSet() {
        Add("win");
        Add("wincp");
        Add("window");
        Add("windowcp");
        Add("windows");
        Add("windowscp");
        Add("ansi");
        Add("ansicp");
    }
};

class TCpPrefixesHashSet: public THashSetType {
public:
    inline TCpPrefixesHashSet() {
        Add("microsoft");
        Add("microsoftcp");
        Add("cp");
    }
};

class TIsoPrefixesHashSet: public THashSetType {
public:
    inline TIsoPrefixesHashSet() {
        Add("iso");
        Add("isolatin");
        Add("latin");
    }
};

class TLatinToIsoHash: public THashMap<const char*, TString, ci_hash, ci_equal_to> {
public:
    inline TLatinToIsoHash() {
        insert(value_type("latin1", "iso-8859-1"));
        insert(value_type("latin2", "iso-8859-2"));
        insert(value_type("latin3", "iso-8859-3"));
        insert(value_type("latin4", "iso-8859-4"));
        insert(value_type("latin5", "iso-8859-9"));
        insert(value_type("latin6", "iso-8859-10"));
        insert(value_type("latin7", "iso-8859-13"));
        insert(value_type("latin8", "iso-8859-14"));
        insert(value_type("latin9", "iso-8859-15"));
        insert(value_type("latin10", "iso-8859-16"));
    }
};

static inline void NormalizeEncodingPrefixes(TString& enc) {
    size_t preflen = enc.find_first_of("0123456789");
    if (preflen == TString::npos)
        return;

    TString prefix = enc.substr(0, preflen);
    for (size_t i = 0; i < prefix.length(); ++i) {
        if (prefix[i] == '-') {
            prefix.remove(i--);
        }
    }

    if (Singleton<TWindowsPrefixesHashSet>()->Has(prefix)) {
        enc.remove(0, preflen);
        enc.prepend("windows-");
        return;
    }

    if (Singleton<TCpPrefixesHashSet>()->Has(prefix)) {
        if (enc.length() > preflen + 3 && !strncmp(enc.c_str() + preflen, "125", 3) && isdigit(enc[preflen + 3])) {
            enc.remove(0, preflen);
            enc.prepend("windows-");
            return;
        }
        enc.remove(0, preflen);
        enc.prepend("cp");
        return;
    }

    if (Singleton<TIsoPrefixesHashSet>()->Has(prefix)) {
        if (enc.length() == preflen + 1 || enc.length() == preflen + 2) {
            TString enccopy = enc.substr(preflen);
            enccopy.prepend("latin");
            const TLatinToIsoHash* latinhash = Singleton<TLatinToIsoHash>();
            TLatinToIsoHash::const_iterator it = latinhash->find(enccopy.data());
            if (it != latinhash->end())
                enc.assign(it->second);
            return;
        } else if (enc.length() > preflen + 5 && enc[preflen] == '8') {
            enc.remove(0, preflen);
            enc.prepend("iso-");
            return;
        }
    }
}

class TEncodingNamesHashSet: public THashSetType {
public:
    TEncodingNamesHashSet() {
        Add("iso-8859-1");
        Add("iso-8859-2");
        Add("iso-8859-3");
        Add("iso-8859-4");
        Add("iso-8859-5");
        Add("iso-8859-6");
        Add("iso-8859-7");
        Add("iso-8859-8");
        Add("iso-8859-8-i");
        Add("iso-8859-9");
        Add("iso-8859-10");
        Add("iso-8859-11");
        Add("iso-8859-12");
        Add("iso-8859-13");
        Add("iso-8859-14");
        Add("iso-8859-15");
        Add("windows-1250");
        Add("windows-1251");
        Add("windows-1252");
        Add("windows-1253");
        Add("windows-1254");
        Add("windows-1255");
        Add("windows-1256");
        Add("windows-1257");
        Add("windows-1258");
        Add("windows-874");
        Add("iso-2022-jp");
        Add("euc-jp");
        Add("shift-jis");
        Add("shiftjis");
        Add("iso-2022-kr");
        Add("euc-kr");
        Add("gb-2312");
        Add("gb2312");
        Add("gb-18030");
        Add("gb18030");
        Add("gbk");
        Add("big5");
        Add("tis-620");
        Add("tis620");
    }
};

ECharset EncodingHintByName(const char* encname) {
    if (!encname)
        return CODES_UNKNOWN; // safety check

    // Common trouble: spurious "charset=" in the encoding name
    if (!strnicmp(encname, "charset=", 8)) {
        encname += 8;
    }

    // Strip everything up to the first alphanumeric, and after the last one
    while (*encname && !isalnum(*encname))
        ++encname;

    if (!*encname)
        return CODES_UNKNOWN;

    const char* lastpos = encname + strlen(encname) - 1;
    while (lastpos > encname && !isalnum(*lastpos))
        --lastpos;

    // Do some normalization
    TString enc(encname, lastpos - encname + 1);
    enc.to_lower();
    for (char* p = enc.begin(); p != enc.end(); ++p) {
        if (*p == ' ' || *p == '=' || *p == '_')
            *p = '-';
    }

    NormalizeEncodingPrefixes(enc);

    ECharset hint = CharsetByName(enc.c_str());
    if (hint != CODES_UNKNOWN)
        return hint;

    if (Singleton<TEncodingNamesHashSet>()->Has(enc))
        return CODES_UNSUPPORTED;
    return CODES_UNKNOWN;
}
