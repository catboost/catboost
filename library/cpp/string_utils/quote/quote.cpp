#include "quote.h"

#include <util/memory/tempbuf.h>
#include <util/string/ascii.h>
#include <util/string/cstriter.h>

/* note: (x & 0xdf) makes x upper case */
#define GETXC                                                           \
    do {                                                                \
        c *= 16;                                                        \
        c += (x[0] >= 'A' ? ((x[0] & 0xdf) - 'A') + 10 : (x[0] - '0')); \
        ++x;                                                            \
    } while (0)

#define GETSBXC                                                         \
    do {                                                                \
        c *= 16;                                                        \
        c += (x[0] >= 'A' ? ((x[0] & 0xdf) - 'A') + 10 : (x[0] - '0')); \
        x.Skip(1);                                                      \
    } while (0)


namespace {
    class TFromHexZeroTerm {
    public:
        static inline char x2c(const char*& x) {
            if (!IsAsciiHex((ui8)x[0]) || !IsAsciiHex((ui8)x[1]))
                return '%';
            ui8 c = 0;

            GETXC;
            GETXC;
            return c;
        }

        static inline char x2c(TStringBuf& x) {
            if (!IsAsciiHex((ui8)x[0]) || !IsAsciiHex((ui8)x[1]))
                return '%';
            ui8 c = 0;

            GETSBXC;
            GETSBXC;
            return c;
        }
    };

    class TFromHexLenLimited {
    public:
        explicit TFromHexLenLimited(const char* end)
            : End(end)
        {
        }

        inline char x2c(const char*& x) {
            if (x + 2 > End)
                return '%';
            return TFromHexZeroTerm::x2c(x);
        }

    private:
        const char* End;
    };
}

static inline char d2x(unsigned x) {
    return (char)((x < 10) ? ('0' + x) : ('A' + x - 10));
}

static inline const char* FixZero(const char* s) noexcept {
    return s ? s : "";
}

// we escape:
// '\"', '|', '(', ')',
// '%',  '&', '+', ',',
// '#',  '<', '=', '>',
// '[',  '\\',']', '?',
//  ':', '{', '}', '^'
// all below ' ' (0x20) and above '~' (0x7E).
// ' ' converted to '+'
static const bool chars_to_url_escape[256] = {
//  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //1
    0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, //2
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, //3

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //4
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, //5
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //6
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, //7

    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //8
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //9
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //A
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //B

    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //C
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //D
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //E
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //F
};

template <class It1, class It2, class It3>
static inline It1 Escape(It1 to, It2 from, It3 end, const bool* escape_map = chars_to_url_escape) {
    while (from != end) {
        if (escape_map[(unsigned char)*from]) {
            *to++ = '%';
            *to++ = d2x((unsigned char)*from >> 4);
            *to++ = d2x((unsigned char)*from & 0xF);
        } else {
            *to++ = (*from == ' ' ? '+' : *from);
        }

        ++from;
    }

    *to = 0;

    return to;
}

template <class It1, class It2, class It3, class FromHex>
static inline It1 Unescape(It1 to, It2 from, It3 end, FromHex fromHex) {
    (void)fromHex;

    while (from != end) {
        switch (*from) {
            case '%':
                ++from;
                *to++ = fromHex.x2c(from);
                break;
            case '+':
                *to++ = ' ';
                ++from;
                break;
            default:
                *to++ = *from++;
        }
    }
    *to = 0;
    return to;
}

// CGIEscape returns pointer to the end of the result string
// so as it could be possible to populate single long buffer
// with several calls to CGIEscape in a row.
char* CGIEscape(char* to, const char* from) {
    return Escape(to, FixZero(from), TCStringEndIterator());
}

char* CGIEscape(char* to, const char* from, size_t len) {
    return Escape(to, from, from + len);
}

void CGIEscape(TString& url) {
    TTempBuf tempBuf(CgiEscapeBufLen(url.size()));
    char* to = tempBuf.Data();

    url.AssignNoAlias(to, CGIEscape(to, url.data(), url.size()));
}

TString CGIEscapeRet(const TStringBuf url) {
    TString to;
    to.ReserveAndResize(CgiEscapeBufLen(url.size()));
    to.resize(CGIEscape(to.begin(), url.data(), url.size()) - to.data());
    return to;
}

TString& AppendCgiEscaped(const TStringBuf value, TString& to) {
    const size_t origLength = to.length();
    to.ReserveAndResize(origLength + CgiEscapeBufLen(value.size()));
    to.resize(CGIEscape(to.begin() + origLength, value.data(), value.size()) - to.data());
    return to;
}

// More general version of CGIEscape. The optional safe parameter specifies
// additional characters that should not be quoted — its default value is '/'.

// Also returns pointer to the end of result string.

template <class It1, class It2, class It3>
static inline It1 Quote(It1 to, It2 from, It3 end, const char* safe) {
    bool escape_map[256];
    memcpy(escape_map, chars_to_url_escape, 256);
    // RFC 3986 Uniform Resource Identifiers (URI): Generic Syntax
    // lists following reserved characters:
    const char* reserved = ":/?#[]@!$&\'()*+,;=";
    for (const char* p = reserved; *p; ++p) {
        escape_map[(unsigned char)*p] = true;
    }
    // characters we think are safe at the moment
    for (const char* p = safe; *p; ++p) {
        escape_map[(unsigned char)*p] = false;
    }

    return Escape(to, from, end, escape_map);
}

char* Quote(char* to, const char* from, const char* safe) {
    return Quote(to, FixZero(from), TCStringEndIterator(), safe);
}

char* Quote(char* to, const TStringBuf s, const char* safe) {
    return Quote(to, s.data(), s.data() + s.size(), safe);
}

void Quote(TString& url, const char* safe) {
    TTempBuf tempBuf(CgiEscapeBufLen(url.size()));
    char* to = tempBuf.Data();

    url.AssignNoAlias(to, Quote(to, url, safe));
}

char* CGIUnescape(char* to, const char* from) {
    return Unescape(to, FixZero(from), TCStringEndIterator(), TFromHexZeroTerm());
}

char* CGIUnescape(char* to, const char* from, size_t len) {
    return Unescape(to, from, from + len, TFromHexLenLimited(from + len));
}

void CGIUnescape(TString& url) {
    if (url.empty()) {
        return;
    }
    if (url.IsDetached()) { // in-place when refcount == 1
        char* resBegin = url.begin();
        const char* resEnd = CGIUnescape(resBegin, resBegin, url.size());
        url.resize(resEnd - resBegin);
    } else {
        url = CGIUnescapeRet(url);
    }
}

TString CGIUnescapeRet(const TStringBuf from) {
    TString to;
    to.ReserveAndResize(CgiUnescapeBufLen(from.size()));
    to.resize(CGIUnescape(to.begin(), from.data(), from.size()) - to.data());
    return to;
}

char* UrlUnescape(char* to, TStringBuf from) {
    while (!from.empty()) {
        char ch = from[0];
        from.Skip(1);
        if ('%' == ch && 2 <= from.length())
            ch = TFromHexZeroTerm::x2c(from);
        *to++ = ch;
    }

    *to = 0;

    return to;
}

void UrlUnescape(TString& url) {
    if (url.empty()) {
        return;
    }
    if (url.IsDetached()) { // in-place when refcount == 1
        char* resBegin = url.begin();
        const char* resEnd = UrlUnescape(resBegin, url);
        url.resize(resEnd - resBegin);
    } else {
        url = UrlUnescapeRet(url);
    }
}

TString UrlUnescapeRet(const TStringBuf from) {
    TString to;
    to.ReserveAndResize(CgiUnescapeBufLen(from.size()));
    to.resize(UrlUnescape(to.begin(), from) - to.data());
    return to;
}

char* UrlEscape(char* to, const char* from, bool forceEscape) {
    from = FixZero(from);

    while (*from) {
        const bool escapePercent = (*from == '%') &&
                                   (forceEscape || !((*(from + 1) && IsAsciiHex(*(from + 1)) && *(from + 2) && IsAsciiHex(*(from + 2)))));

        if (escapePercent || (unsigned char)*from <= ' ' || (unsigned char)*from > '~') {
            *to++ = '%';
            *to++ = d2x((unsigned char)*from >> 4);
            *to++ = d2x((unsigned char)*from & 0xF);
        } else
            *to++ = *from;
        ++from;
    }

    *to = 0;

    return to;
}

void UrlEscape(TString& url, bool forceEscape) {
    TTempBuf tempBuf(CgiEscapeBufLen(url.size()));
    char* to = tempBuf.Data();
    url.AssignNoAlias(to, UrlEscape(to, url.data(), forceEscape));
}

TString UrlEscapeRet(const TStringBuf from, bool forceEscape) {
    TString to;
    to.ReserveAndResize(CgiEscapeBufLen(from.size()));
    to.resize(UrlEscape(to.begin(), from.begin(), forceEscape) - to.data());
    return to;
}
