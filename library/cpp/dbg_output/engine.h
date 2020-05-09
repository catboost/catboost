#pragma once

#include <util/stream/output.h>

#include <utility>
#include <util/generic/strbuf.h>

template <class T>
struct TDumper {
    template <class S>
    static inline void Dump(S& s, const T& t) {
        s.Stream() << t;
    }
};

namespace NDumpPrivate {
    template <class T, class V>
    inline void Dump(T& t, const V& v) {
        ::TDumper<V>::Dump(t, v);
    }

    template <class T, class V>
    inline T&& operator<<(T&& t, V&& v) {
        Dump(t, v);

        return std::forward<T>(t);
    }

    struct TADLBase {
    };
}

struct TDumpBase: public ::NDumpPrivate::TADLBase {
    inline TDumpBase(IOutputStream& out, bool indent) noexcept
        : Out(&out)
        , IndentLevel(0)
        , Indent(indent)
    {
    }

    inline IOutputStream& Stream() const noexcept {
        return *Out;
    }

    void Char(char ch);
    void Char(wchar16 ch);

    void String(const TStringBuf& s);
    void String(const TWtringBuf& s);

    void Raw(const TStringBuf& s);

    IOutputStream* Out;
    size_t IndentLevel;
    bool Indent;
};

struct TIndentScope {
    inline TIndentScope(TDumpBase& d)
        : D(&d)
    {
        ++(D->IndentLevel);
    }

    inline ~TIndentScope() {
        --(D->IndentLevel);
    }

    TDumpBase* D;
};

template <class TChar>
struct TRawLiteral {
    const TBasicStringBuf<TChar> S;
};

template <class TChar>
static inline TRawLiteral<TChar> DumpRaw(const TBasicStringBuf<TChar>& s) noexcept {
    return {s};
}

template <class TChar>
static inline TRawLiteral<TChar> DumpRaw(const TChar* s) noexcept {
    return {s};
}

template <class C>
struct TDumper<TRawLiteral<C>> {
    template <class S>
    static inline void Dump(S& s, const TRawLiteral<C>& v) {
        s.Raw(v.S);
    }
};

struct TIndentNewLine {
};

static inline TIndentNewLine IndentNewLine() noexcept {
    return {};
}

template <>
struct TDumper<TIndentNewLine> {
    template <class S>
    static inline void Dump(S& s, const TIndentNewLine&) {
        if (s.Indent) {
            s << DumpRaw("\n") << DumpRaw(TString(s.IndentLevel * 4, ' ').data());
        }
    }
};

template <class P>
struct TDumper<const P*> {
    template <class S>
    static inline void Dump(S& s, const P* p) {
        s.Pointer(p);
    }
};

template <class P>
struct TDumper<P*>: public TDumper<const P*> {
};

struct TCharDumper {
    template <class S, class V>
    static inline void Dump(S& s, const V& v) {
        s.Char(v);
    }
};

template <class S, class V>
static inline void OutSequence(S& s, const V& v, const char* openTag, const char* closeTag) {
    s.ColorScheme.Markup(s);
    s << DumpRaw(openTag);

    {
        TIndentScope scope(s);
        size_t cnt = 0;

        for (const auto& x : v) {
            if (cnt) {
                s.ColorScheme.Markup(s);
                s << DumpRaw(", ");
            }

            s << IndentNewLine();
            s.ColorScheme.Literal(s);
            s << x;
            ++cnt;
        }
    }

    s << IndentNewLine();
    s.ColorScheme.Markup(s);
    s << DumpRaw(closeTag);
    s.ColorScheme.ResetType(s);
}

struct TAssocDumper {
    template <class S, class V>
    static inline void Dump(S& s, const V& v) {
        ::OutSequence(s, v, "{", "}");
    }
};

struct TSeqDumper {
    template <class S, class V>
    static inline void Dump(S& s, const V& v) {
        ::OutSequence(s, v, "[", "]");
    }
};

struct TStrDumper {
    template <class S, class V>
    static inline void Dump(S& s, const V& v) {
        s.ColorScheme.String(s);
        s.String(v);
        s.ColorScheme.ResetType(s);
    }
};
