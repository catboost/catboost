#pragma once

#include "engine.h"

#include <util/generic/fwd.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>

//smart pointers
template <class T, class D>
struct TDumper<TAutoPtr<T, D>> {
    template <class S>
    static inline void Dump(S& s, const TAutoPtr<T, D>& v) {
        s << DumpRaw("TAutoPtr(") << v.Get() << DumpRaw(")");
    }
};

template <class T, class D>
struct TDumper<THolder<T, D>> {
    template <class S>
    static inline void Dump(S& s, const THolder<T, D>& v) {
        s << DumpRaw("THolder(") << v.Get() << DumpRaw(")");
    }
};

template <class T, class Ops>
struct TDumper<TIntrusivePtr<T, Ops>> {
    template <class S>
    static inline void Dump(S& s, const TIntrusivePtr<T, Ops>& v) {
        s << DumpRaw("TIntrusivePtr(") << v.Get() << DumpRaw(")");
    }
};

template <class T, class C, class D>
struct TDumper<TSharedPtr<T, C, D>> {
    template <class S>
    static inline void Dump(S& s, const TSharedPtr<T, C, D>& v) {
        s << DumpRaw("TSharedPtr(") << v.Get() << DumpRaw(")");
    }
};

template <class T, class D>
struct TDumper<TLinkedPtr<T, D>> {
    template <class S>
    static inline void Dump(S& s, const TLinkedPtr<T, D>& v) {
        s << DumpRaw("TLinkedPtr(") << v.Get() << DumpRaw(")");
    }
};

template <class T, class C, class D>
struct TDumper<TCopyPtr<T, C, D>> {
    template <class S>
    static inline void Dump(S& s, const TCopyPtr<T, C, D>& v) {
        s << DumpRaw("TCopyPtr(") << v.Get() << DumpRaw(")");
    }
};

//small ints
// Default dumper prints them via IOutputStream << (value), which results in raw
// chars, not integer values. Cast to a bigger int type to force printing as
// integers.
// NB: i8 = signed char != char != unsigned char = ui8
template <>
struct TDumper<ui8>: public TDumper<i32> {
};

template <>
struct TDumper<i8>: public TDumper<i32> {
};

//chars
template <>
struct TDumper<char>: public TCharDumper {
};

template <>
struct TDumper<wchar16>: public TCharDumper {
};

//pairs
template <class A, class B>
struct TDumper<std::pair<A, B>> {
    template <class S>
    static inline void Dump(S& s, const std::pair<A, B>& v) {
        s.ColorScheme.Key(s);
        s.ColorScheme.Literal(s);
        s << v.first;
        s.ColorScheme.ResetType(s);
        s.ColorScheme.ResetRole(s);
        s.ColorScheme.Markup(s);
        s << DumpRaw(" -> ");
        s.ColorScheme.Value(s);
        s.ColorScheme.Literal(s);
        s << v.second;
        s.ColorScheme.ResetType(s);
        s.ColorScheme.ResetRole(s);
    }
};

//sequences
template <class T, class A>
struct TDumper<yvector<T, A>>: public TSeqDumper {
};

template <class T, class A>
struct TDumper<std::vector<T, A>>: public TSeqDumper {
};

template <class T, size_t N>
struct TDumper<std::array<T, N>>: public TSeqDumper {
};

template <class T, class A>
struct TDumper<ydeque<T, A>>: public TSeqDumper {
};

template <class T, class A>
struct TDumper<ylist<T, A>>: public TSeqDumper {
};

//associatives
template <class K, class V, class P, class A>
struct TDumper<ymap<K, V, P, A>>: public TAssocDumper {
};

template <class K, class V, class P, class A>
struct TDumper<ymultimap<K, V, P, A>>: public TAssocDumper {
};

template <class T, class P, class A>
struct TDumper<yset<T, P, A>>: public TAssocDumper {
};

template <class T, class P, class A>
struct TDumper<ymultiset<T, P, A>>: public TAssocDumper {
};

template <class K, class V, class H, class P, class A>
struct TDumper<yhash<K, V, H, P, A>>: public TAssocDumper {
};

template <class K, class V, class H, class P, class A>
struct TDumper<yhash_mm<K, V, H, P, A>>: public TAssocDumper {
};

template <class T, class H, class P, class A>
struct TDumper<yhash_set<T, H, P, A>>: public TAssocDumper {
};

template <class T, class H, class P, class A>
struct TDumper<yhash_multiset<T, H, P, A>>: public TAssocDumper {
};

//strings
template <>
struct TDumper<TString>: public TStrDumper {
};

template <>
struct TDumper<const char*>: public TStrDumper {
};

template <>
struct TDumper<TUtf16String>: public TStrDumper {
};

template <>
struct TDumper<const wchar16*>: public TStrDumper {
};

template <class C, class T, class A>
struct TDumper<std::basic_string<C, T, A>>: public TStrDumper {
};

template <class TChar>
struct TDumper<TGenericStringBuf<TChar>>: public TStrDumper {
};
