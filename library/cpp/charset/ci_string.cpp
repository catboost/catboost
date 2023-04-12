#include "ci_string.h"

int TCiString::compare(const TCiString& s1, const TCiString& s2, const CodePage& cp) {
    return cp.stricmp(s1.data(), s2.data());
}

int TCiString::compare(const char* p, const TCiString& s2, const CodePage& cp) {
    return cp.stricmp(p, s2.data());
}

int TCiString::compare(const TCiString& s1, const char* p, const CodePage& cp) {
    return cp.stricmp(s1.data(), p);
}

int TCiString::compare(const TStringBuf& p1, const TStringBuf& p2, const CodePage& cp) {
    int rv = cp.strnicmp(p1.data(), p2.data(), Min(p1.size(), p2.size()));
    return rv ? rv : p1.size() < p2.size() ? -1 : p1.size() == p2.size() ? 0 : 1;
}

bool TCiString::is_prefix(const TStringBuf& what, const TStringBuf& of, const CodePage& cp) {
    size_t len = what.size();
    return len <= of.size() && cp.strnicmp(what.data(), of.data(), len) == 0;
}

bool TCiString::is_suffix(const TStringBuf& what, const TStringBuf& of, const CodePage& cp) {
    size_t len = what.size();
    size_t slen = of.size();
    return (len <= slen) && (0 == cp.strnicmp(what.data(), of.data() + slen - len, len));
}

size_t TCiString::hashVal(const char* s, size_t len, const CodePage& cp) {
    size_t h = len;
    for (; /* (*s) && */ len--; ++s)
        h = 5 * h + cp.ToLower(*s);
    return h;
}

template <>
void Out<TCiString>(IOutputStream& o, const TCiString& p) {
    o.Write(p.data(), p.size());
}
