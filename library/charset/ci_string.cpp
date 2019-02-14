#include "ci_string.h"
#include "wide.h"

int TCiString::compare(const TCiString& s1, const TCiString& s2, const CodePage& cp) {
    return cp.stricmp(s1.Data_, s2.Data_);
}

int TCiString::compare(const char* p, const TCiString& s2, const CodePage& cp) {
    return cp.stricmp(p, s2.Data_);
}

int TCiString::compare(const TCiString& s1, const char* p, const CodePage& cp) {
    return cp.stricmp(s1.Data_, p);
}

int TCiString::compare(const TFixedString& p1, const TFixedString& p2, const CodePage& cp) {
    int rv = cp.strnicmp(p1.Start, p2.Start, Min(p1.Length, p2.Length));
    return rv ? rv : p1.Length < p2.Length ? -1 : p1.Length == p2.Length ? 0 : 1;
}

bool TCiString::is_prefix(const TFixedString& what, const TFixedString& of, const CodePage& cp) {
    size_t len = what.Length;
    return len <= of.Length && cp.strnicmp(what.Start, of.Start, len) == 0;
}

bool TCiString::is_suffix(const TFixedString& what, const TFixedString& of, const CodePage& cp) {
    size_t len = what.Length;
    size_t slen = of.Length;
    return (len <= slen) && (0 == cp.strnicmp(what.Start, of.Start + slen - len, len));
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
