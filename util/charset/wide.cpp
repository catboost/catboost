#include "wide.h"

#include <util/string/strip.h>

namespace {
    //! the constants are not zero-terminated
    const wchar16 LT[] = {'&', 'l', 't', ';'};
    const wchar16 GT[] = {'&', 'g', 't', ';'};
    const wchar16 AMP[] = {'&', 'a', 'm', 'p', ';'};
    const wchar16 BR[] = {'<', 'B', 'R', '>'};
    const wchar16 QUOT[] = {'&', 'q', 'u', 'o', 't', ';'};

    template <bool insertBr>
    inline size_t EscapedLen(wchar16 c) {
        switch (c) {
            case '<':
                return Y_ARRAY_SIZE(LT);
            case '>':
                return Y_ARRAY_SIZE(GT);
            case '&':
                return Y_ARRAY_SIZE(AMP);
            case '\"':
                return Y_ARRAY_SIZE(QUOT);
            default:
                if (insertBr && (c == '\r' || c == '\n'))
                    return Y_ARRAY_SIZE(BR);
                else
                    return 1;
        }
    }
}

void Collapse(TUtf16String& w) {
    CollapseImpl(w, w, 0, IsWhitespace);
}

size_t Collapse(wchar16* s, size_t n) {
    return CollapseImpl(s, n, IsWhitespace);
}

void Strip(TUtf16String& w) {
    const wchar16* p = w.c_str();
    const wchar16* pe = p + w.size();

    while (p != pe) {
        if (!IsWhitespace(*p)) {
            if (p != w.c_str()) {
                w.erase(w.c_str(), p);
            }

            pe = w.c_str() - 1;
            p = pe + w.size();
            while (p != pe) {
                if (!IsWhitespace(*p))
                    break;
                --p;
            }

            w.remove(p - pe); // it will not change the string if (p - pe) is not less than size
            return;
        }
        ++p;
    }

    // all characters are spaces
    w.clear();
}

template <bool insertBr>
void EscapeHtmlChars(TUtf16String& str) {
    static const TUtf16String lt(LT, Y_ARRAY_SIZE(LT));
    static const TUtf16String gt(GT, Y_ARRAY_SIZE(GT));
    static const TUtf16String amp(AMP, Y_ARRAY_SIZE(AMP));
    static const TUtf16String br(BR, Y_ARRAY_SIZE(BR));
    static const TUtf16String quot(QUOT, Y_ARRAY_SIZE(QUOT));

    size_t escapedLen = 0;

    const TUtf16String& cs = str;

    for (size_t i = 0; i < cs.size(); ++i)
        escapedLen += EscapedLen<insertBr>(cs[i]);

    if (escapedLen == cs.size())
        return;

    TUtf16String res;
    res.reserve(escapedLen);

    size_t start = 0;

    for (size_t i = 0; i < cs.size(); ++i) {
        const TUtf16String* ent = nullptr;
        switch (cs[i]) {
            case '<':
                ent = &lt;
                break;
            case '>':
                ent = &gt;
                break;
            case '&':
                ent = &amp;
                break;
            case '\"':
                ent = &quot;
                break;
            default:
                if (insertBr && (cs[i] == '\r' || cs[i] == '\n')) {
                    ent = &br;
                    break;
                } else
                    continue;
        }

        res.append(cs.begin() + start, cs.begin() + i);
        res.append(ent->begin(), ent->end());
        start = i + 1;
    }

    res.append(cs.begin() + start, cs.end());
    res.swap(str);
}

template void EscapeHtmlChars<false>(TUtf16String& str);
template void EscapeHtmlChars<true>(TUtf16String& str);
