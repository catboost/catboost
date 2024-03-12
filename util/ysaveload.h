#pragma once

#include <util/generic/fwd.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/generic/typetraits.h>
#include <util/generic/algorithm.h>
#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/system/compiler.h>

#ifndef __NVCC__
    // cuda is compiled in C++14 mode at the time
    #include <optional>
    #include <variant>
#endif

template <typename T>
class TSerializeTypeTraits {
public:
    /*
     *  pointer types cannot be serialized as POD-type
     */
    enum {
        IsSerializablePod = TTypeTraits<T>::IsPod && !std::is_pointer<T>::value
    };
};

struct TSerializeException: public yexception {
};

struct TLoadEOF: public TSerializeException {
};

template <class T>
static inline void Save(IOutputStream* out, const T& t);

template <class T>
static inline void SaveArray(IOutputStream* out, const T* t, size_t len);

template <class T>
static inline void Load(IInputStream* in, T& t);

template <class T>
static inline void LoadArray(IInputStream* in, T* t, size_t len);

template <class T, class TStorage>
static inline void Load(IInputStream* in, T& t, TStorage& pool);

template <class T, class TStorage>
static inline void LoadArray(IInputStream* in, T* t, size_t len, TStorage& pool);

template <class T>
static inline void SavePodType(IOutputStream* rh, const T& t) {
    rh->Write(&t, sizeof(T));
}

namespace NPrivate {
    [[noreturn]] void ThrowLoadEOFException(size_t typeSize, size_t realSize, TStringBuf structName);
    [[noreturn]] void ThrowUnexpectedVariantTagException(ui8 tagIndex);
}

template <class T>
static inline void LoadPodType(IInputStream* rh, T& t) {
    const size_t res = rh->Load(&t, sizeof(T));

    if (Y_UNLIKELY(res != sizeof(T))) {
        ::NPrivate::ThrowLoadEOFException(sizeof(T), res, TStringBuf("pod type"));
    }
}

template <class T>
static inline void SavePodArray(IOutputStream* rh, const T* arr, size_t count) {
    rh->Write(arr, sizeof(T) * count);
}

template <class T>
static inline void LoadPodArray(IInputStream* rh, T* arr, size_t count) {
    const size_t len = sizeof(T) * count;
    const size_t res = rh->Load(arr, len);

    if (Y_UNLIKELY(res != len)) {
        ::NPrivate::ThrowLoadEOFException(len, res, TStringBuf("pod array"));
    }
}

template <class It>
static inline void SaveIterRange(IOutputStream* rh, It b, It e) {
    while (b != e) {
        ::Save(rh, *b++);
    }
}

template <class It>
static inline void LoadIterRange(IInputStream* rh, It b, It e) {
    while (b != e) {
        ::Load(rh, *b++);
    }
}

template <class It, class TStorage>
static inline void LoadIterRange(IInputStream* rh, It b, It e, TStorage& pool) {
    while (b != e) {
        ::Load(rh, *b++, pool);
    }
}

template <class T, bool isPod>
struct TSerializerTakingIntoAccountThePodType {
    static inline void Save(IOutputStream* out, const T& t) {
        ::SavePodType(out, t);
    }

    static inline void Load(IInputStream* in, T& t) {
        ::LoadPodType(in, t);
    }

    template <class TStorage>
    static inline void Load(IInputStream* in, T& t, TStorage& /*pool*/) {
        ::LoadPodType(in, t);
    }

    static inline void SaveArray(IOutputStream* out, const T* t, size_t len) {
        ::SavePodArray(out, t, len);
    }

    static inline void LoadArray(IInputStream* in, T* t, size_t len) {
        ::LoadPodArray(in, t, len);
    }
};

namespace NHasSaveLoad {
    Y_HAS_MEMBER(SaveLoad);
}

template <class T, class = void>
struct TSerializerMethodSelector;

template <class T>
struct TSerializerMethodSelector<T, std::enable_if_t<NHasSaveLoad::THasSaveLoad<T>::value>> {
    static inline void Save(IOutputStream* out, const T& t) {
        //assume Save clause do not change t
        (const_cast<T&>(t)).SaveLoad(out);
    }

    static inline void Load(IInputStream* in, T& t) {
        t.SaveLoad(in);
    }

    template <class TStorage>
    static inline void Load(IInputStream* in, T& t, TStorage& pool) {
        t.SaveLoad(in, pool);
    }
};

template <class T>
struct TSerializerMethodSelector<T, std::enable_if_t<!NHasSaveLoad::THasSaveLoad<T>::value>> {
    static inline void Save(IOutputStream* out, const T& t) {
        t.Save(out);
    }

    static inline void Load(IInputStream* in, T& t) {
        t.Load(in);
    }

    template <class TStorage>
    static inline void Load(IInputStream* in, T& t, TStorage& pool) {
        t.Load(in, pool);
    }
};

template <class T>
struct TSerializerTakingIntoAccountThePodType<T, false>: public TSerializerMethodSelector<T> {
    static inline void SaveArray(IOutputStream* out, const T* t, size_t len) {
        ::SaveIterRange(out, t, t + len);
    }

    static inline void LoadArray(IInputStream* in, T* t, size_t len) {
        ::LoadIterRange(in, t, t + len);
    }

    template <class TStorage>
    static inline void LoadArray(IInputStream* in, T* t, size_t len, TStorage& pool) {
        ::LoadIterRange(in, t, t + len, pool);
    }
};

template <class It, bool isPtr>
struct TRangeSerialize {
    static inline void Save(IOutputStream* rh, It b, It e) {
        SaveArray(rh, b, e - b);
    }

    static inline void Load(IInputStream* rh, It b, It e) {
        LoadArray(rh, b, e - b);
    }

    template <class TStorage>
    static inline void Load(IInputStream* rh, It b, It e, TStorage& pool) {
        LoadArray(rh, b, e - b, pool);
    }
};

template <class It>
struct TRangeSerialize<It, false> {
    static inline void Save(IOutputStream* rh, It b, It e) {
        SaveIterRange(rh, b, e);
    }

    static inline void Load(IInputStream* rh, It b, It e) {
        LoadIterRange(rh, b, e);
    }

    template <class TStorage>
    static inline void Load(IInputStream* rh, It b, It e, TStorage& pool) {
        LoadIterRange(rh, b, e, pool);
    }
};

template <class It>
static inline void SaveRange(IOutputStream* rh, It b, It e) {
    TRangeSerialize<It, std::is_pointer<It>::value>::Save(rh, b, e);
}

template <class It>
static inline void LoadRange(IInputStream* rh, It b, It e) {
    TRangeSerialize<It, std::is_pointer<It>::value>::Load(rh, b, e);
}

template <class It, class TStorage>
static inline void LoadRange(IInputStream* rh, It b, It e, TStorage& pool) {
    TRangeSerialize<It, std::is_pointer<It>::value>::Load(rh, b, e, pool);
}

template <class T>
class TSerializer: public TSerializerTakingIntoAccountThePodType<T, TSerializeTypeTraits<T>::IsSerializablePod> {
};

template <class T>
class TArraySerializer: public TSerializerTakingIntoAccountThePodType<T, TSerializeTypeTraits<T>::IsSerializablePod> {
};

template <class T>
static inline void Save(IOutputStream* out, const T& t) {
    TSerializer<T>::Save(out, t);
}

template <class T>
static inline void SaveArray(IOutputStream* out, const T* t, size_t len) {
    TArraySerializer<T>::SaveArray(out, t, len);
}

template <class T>
static inline void Load(IInputStream* in, T& t) {
    TSerializer<T>::Load(in, t);
}

template <class T>
static inline void LoadArray(IInputStream* in, T* t, size_t len) {
    TArraySerializer<T>::LoadArray(in, t, len);
}

template <class T, class TStorage>
static inline void Load(IInputStream* in, T& t, TStorage& pool) {
    TSerializer<T>::Load(in, t, pool);
}

template <class T, class TStorage>
static inline void LoadArray(IInputStream* in, T* t, size_t len, TStorage& pool) {
    TArraySerializer<T>::LoadArray(in, t, len, pool);
}

static inline void SaveSize(IOutputStream* rh, size_t len) {
    if ((ui64)len < 0xffffffff) {
        ::Save(rh, (ui32)len);
    } else {
        ::Save(rh, (ui32)0xffffffff);
        ::Save(rh, (ui64)len);
    }
}

static inline size_t LoadSize(IInputStream* rh) {
    ui32 oldVerSize;
    ui64 newVerSize;
    ::Load(rh, oldVerSize);
    if (oldVerSize != 0xffffffff) {
        return oldVerSize;
    } else {
        ::Load(rh, newVerSize);
        return newVerSize;
    }
}

template <class C>
static inline void LoadSizeAndResize(IInputStream* rh, C& c) {
    c.resize(LoadSize(rh));
}

template <class TStorage>
static inline char* AllocateFromPool(TStorage& pool, size_t len) {
    return static_cast<char*>(pool.Allocate(len));
}

template <>
class TSerializer<const char*> {
public:
    static inline void Save(IOutputStream* rh, const char* s) {
        size_t length = strlen(s);
        ::SaveSize(rh, length);
        ::SavePodArray(rh, s, length);
    }

    template <class Char, class TStorage>
    static inline void Load(IInputStream* rh, Char*& s, TStorage& pool) {
        const size_t len = LoadSize(rh);

        char* res = AllocateFromPool(pool, len + 1);
        ::LoadPodArray(rh, res, len);
        res[len] = 0;
        s = res;
    }
};

template <class TVec>
class TVectorSerializer {
    using TIter = typename TVec::iterator;

public:
    static inline void Save(IOutputStream* rh, const TVec& v) {
        ::SaveSize(rh, v.size());
        ::SaveRange(rh, v.begin(), v.end());
    }

    static inline void Load(IInputStream* rh, TVec& v) {
        ::LoadSizeAndResize(rh, v);
        TIter b = v.begin();
        TIter e = (TIter)v.end();
        ::LoadRange(rh, b, e);
    }

    template <class TStorage>
    static inline void Load(IInputStream* rh, TVec& v, TStorage& pool) {
        ::LoadSizeAndResize(rh, v);
        TIter b = v.begin();
        TIter e = (TIter)v.end();
        ::LoadRange(rh, b, e, pool);
    }
};

template <class T, class A>
class TSerializer<TVector<T, A>>: public TVectorSerializer<TVector<T, A>> {
};

template <class T, class A>
class TSerializer<std::vector<T, A>>: public TVectorSerializer<std::vector<T, A>> {
};

template <class T, class A>
class TSerializer<TList<T, A>>: public TVectorSerializer<TList<T, A>> {
};

template <class T, class A>
class TSerializer<std::list<T, A>>: public TVectorSerializer<std::list<T, A>> {
};

template <>
class TSerializer<TString>: public TVectorSerializer<TString> {
};

template <>
class TSerializer<TUtf16String>: public TVectorSerializer<TUtf16String> {
};

template <class TChar>
class TSerializer<std::basic_string<TChar>>: public TVectorSerializer<std::basic_string<TChar>> {
};

template <class T, class A>
class TSerializer<TDeque<T, A>>: public TVectorSerializer<TDeque<T, A>> {
};

template <class T, class A>
class TSerializer<std::deque<T, A>>: public TVectorSerializer<std::deque<T, A>> {
};

template <class TArray>
class TStdArraySerializer {
public:
    static inline void Save(IOutputStream* rh, const TArray& a) {
        ::SaveArray(rh, a.data(), a.size());
    }

    static inline void Load(IInputStream* rh, TArray& a) {
        ::LoadArray(rh, a.data(), a.size());
    }
};

template <class T, size_t N>
class TSerializer<std::array<T, N>>: public TStdArraySerializer<std::array<T, N>> {
};

template <class A, class B>
class TSerializer<std::pair<A, B>> {
    using TPair = std::pair<A, B>;

public:
    static inline void Save(IOutputStream* rh, const TPair& p) {
        ::Save(rh, p.first);
        ::Save(rh, p.second);
    }

    static inline void Load(IInputStream* rh, TPair& p) {
        ::Load(rh, p.first);
        ::Load(rh, p.second);
    }

    template <class TStorage>
    static inline void Load(IInputStream* rh, TPair& p, TStorage& pool) {
        ::Load(rh, p.first, pool);
        ::Load(rh, p.second, pool);
    }
};

template <class T>
struct TTupleSerializer {
    template <class F, class Tuple, size_t... Indices>
    static inline void ReverseUseless(F&& f, Tuple&& t, std::index_sequence<Indices...>) {
        ApplyToMany(
            std::forward<F>(f),
            // We need to do this trick because we don't want to break backward compatibility.
            // Tuples are being packed in reverse order.
            std::get<std::tuple_size<T>::value - Indices - 1>(std::forward<Tuple>(t))...);
    }

    static inline void Save(IOutputStream* stream, const T& t) {
        ReverseUseless([&](const auto& v) { ::Save(stream, v); }, t,
                       std::make_index_sequence<std::tuple_size<T>::value>{});
    }

    static inline void Load(IInputStream* stream, T& t) {
        ReverseUseless([&](auto& v) { ::Load(stream, v); }, t,
                       std::make_index_sequence<std::tuple_size<T>::value>{});
    }
};

template <typename... TArgs>
struct TSerializer<std::tuple<TArgs...>>: TTupleSerializer<std::tuple<TArgs...>> {
};

template <>
class TSerializer<TBuffer> {
public:
    static void Save(IOutputStream* rh, const TBuffer& buf);
    static void Load(IInputStream* rh, TBuffer& buf);
};

template <class TSetOrMap, class TValue>
class TSetSerializerInserterBase {
public:
    inline TSetSerializerInserterBase(TSetOrMap& s)
        : S_(s)
    {
        S_.clear();
    }

    inline void Insert(const TValue& v) {
        S_.insert(v);
    }

protected:
    TSetOrMap& S_;
};

template <class TSetOrMap, class TValue, bool sorted>
class TSetSerializerInserter: public TSetSerializerInserterBase<TSetOrMap, TValue> {
    using TBase = TSetSerializerInserterBase<TSetOrMap, TValue>;

public:
    inline TSetSerializerInserter(TSetOrMap& s, size_t cnt)
        : TBase(s)
    {
        Y_UNUSED(cnt);
    }
};

template <class TSetType, class TValue>
class TSetSerializerInserter<TSetType, TValue, true>: public TSetSerializerInserterBase<TSetType, TValue> {
    using TBase = TSetSerializerInserterBase<TSetType, TValue>;

public:
    inline TSetSerializerInserter(TSetType& s, size_t cnt)
        : TBase(s)
    {
        Y_UNUSED(cnt);
        P_ = this->S_.begin();
    }

    inline void Insert(const TValue& v) {
        P_ = this->S_.insert(P_, v);
    }

private:
    typename TSetType::iterator P_;
};

template <class T1, class T2, class T3, class T4, class T5, class TValue>
class TSetSerializerInserter<THashMap<T1, T2, T3, T4, T5>, TValue, false>: public TSetSerializerInserterBase<THashMap<T1, T2, T3, T4, T5>, TValue> {
    using TMapType = THashMap<T1, T2, T3, T4, T5>;
    using TBase = TSetSerializerInserterBase<TMapType, TValue>;

public:
    inline TSetSerializerInserter(TMapType& m, size_t cnt)
        : TBase(m)
    {
        m.reserve(cnt);
    }
};

template <class T1, class T2, class T3, class T4, class T5, class TValue>
class TSetSerializerInserter<THashMultiMap<T1, T2, T3, T4, T5>, TValue, false>: public TSetSerializerInserterBase<THashMultiMap<T1, T2, T3, T4, T5>, TValue> {
    using TMapType = THashMultiMap<T1, T2, T3, T4, T5>;
    using TBase = TSetSerializerInserterBase<TMapType, TValue>;

public:
    inline TSetSerializerInserter(TMapType& m, size_t cnt)
        : TBase(m)
    {
        m.reserve(cnt);
    }
};

template <class T1, class T2, class T3, class T4, class TValue>
class TSetSerializerInserter<THashSet<T1, T2, T3, T4>, TValue, false>: public TSetSerializerInserterBase<THashSet<T1, T2, T3, T4>, TValue> {
    using TSetType = THashSet<T1, T2, T3, T4>;
    using TBase = TSetSerializerInserterBase<TSetType, TValue>;

public:
    inline TSetSerializerInserter(TSetType& s, size_t cnt)
        : TBase(s)
    {
        s.reserve(cnt);
    }
};

template <class TSetType, class TValue, bool sorted>
class TSetSerializerBase {
public:
    static inline void Save(IOutputStream* rh, const TSetType& s) {
        ::SaveSize(rh, s.size());
        ::SaveRange(rh, s.begin(), s.end());
    }

    static inline void Load(IInputStream* rh, TSetType& s) {
        const size_t cnt = ::LoadSize(rh);
        TSetSerializerInserter<TSetType, TValue, sorted> ins(s, cnt);

        TValue v;
        for (size_t i = 0; i != cnt; ++i) {
            ::Load(rh, v);
            ins.Insert(v);
        }
    }

    template <class TStorage>
    static inline void Load(IInputStream* rh, TSetType& s, TStorage& pool) {
        const size_t cnt = ::LoadSize(rh);
        TSetSerializerInserter<TSetType, TValue, sorted> ins(s, cnt);

        TValue v;
        for (size_t i = 0; i != cnt; ++i) {
            ::Load(rh, v, pool);
            ins.Insert(v);
        }
    }
};

template <class TMapType, bool sorted = false>
struct TMapSerializer: public TSetSerializerBase<TMapType, std::pair<typename TMapType::key_type, typename TMapType::mapped_type>, sorted> {
};

template <class TSetType, bool sorted = false>
struct TSetSerializer: public TSetSerializerBase<TSetType, typename TSetType::value_type, sorted> {
};

template <class T1, class T2, class T3, class T4>
class TSerializer<TMap<T1, T2, T3, T4>>: public TMapSerializer<TMap<T1, T2, T3, T4>, true> {
};

template <class K, class T, class C, class A>
class TSerializer<std::map<K, T, C, A>>: public TMapSerializer<std::map<K, T, C, A>, true> {
};

template <class T1, class T2, class T3, class T4>
class TSerializer<TMultiMap<T1, T2, T3, T4>>: public TMapSerializer<TMultiMap<T1, T2, T3, T4>, true> {
};

template <class K, class T, class C, class A>
class TSerializer<std::multimap<K, T, C, A>>: public TMapSerializer<std::multimap<K, T, C, A>, true> {
};

template <class T1, class T2, class T3, class T4, class T5>
class TSerializer<THashMap<T1, T2, T3, T4, T5>>: public TMapSerializer<THashMap<T1, T2, T3, T4, T5>, false> {
};

template <class T1, class T2, class T3, class T4, class T5>
class TSerializer<THashMultiMap<T1, T2, T3, T4, T5>>: public TMapSerializer<THashMultiMap<T1, T2, T3, T4, T5>, false> {
};

template <class K, class C, class A>
class TSerializer<TSet<K, C, A>>: public TSetSerializer<TSet<K, C, A>, true> {
};

template <class K, class C, class A>
class TSerializer<std::set<K, C, A>>: public TSetSerializer<std::set<K, C, A>, true> {
};

template <class T1, class T2, class T3, class T4>
class TSerializer<THashSet<T1, T2, T3, T4>>: public TSetSerializer<THashSet<T1, T2, T3, T4>, false> {
};

template <class T1, class T2>
class TSerializer<TQueue<T1, T2>> {
public:
    static inline void Save(IOutputStream* rh, const TQueue<T1, T2>& v) {
        ::Save(rh, v.Container());
    }
    static inline void Load(IInputStream* in, TQueue<T1, T2>& t) {
        ::Load(in, t.Container());
    }
};

template <class T1, class T2, class T3>
class TSerializer<TPriorityQueue<T1, T2, T3>> {
public:
    static inline void Save(IOutputStream* rh, const TPriorityQueue<T1, T2, T3>& v) {
        ::Save(rh, v.Container());
    }
    static inline void Load(IInputStream* in, TPriorityQueue<T1, T2, T3>& t) {
        ::Load(in, t.Container());
    }
};

#ifndef __NVCC__

template <typename T>
struct TSerializer<std::optional<T>> {
    static inline void Save(IOutputStream* os, const std::optional<T>& v) {
        ::Save(os, v.has_value());
        if (v.has_value()) {
            ::Save(os, *v);
        }
    }

    static inline void Load(IInputStream* is, std::optional<T>& v) {
        v.reset();

        bool hasValue;
        ::Load(is, hasValue);

        if (hasValue) {
            ::Load(is, v.emplace());
        }
    }
};

namespace NPrivate {
    template <class Variant, class T, size_t I>
    void LoadVariantAlternative(IInputStream* is, Variant& v) {
        T loaded;
        ::Load(is, loaded);
        v.template emplace<I>(std::move(loaded));
    }
}

template <typename... Args>
struct TSerializer<std::variant<Args...>> {
    using TVar = std::variant<Args...>;

    static_assert(sizeof...(Args) < 256, "We use ui8 to store tag");

    static void Save(IOutputStream* os, const TVar& v) {
        ::Save<ui8>(os, v.index());
        std::visit([os](const auto& data) {
            ::Save(os, data);
        }, v);
    }

    static void Load(IInputStream* is, TVar& v) {
        ui8 index;
        ::Load(is, index);
        if (Y_UNLIKELY(index >= sizeof...(Args))) {
            ::NPrivate::ThrowUnexpectedVariantTagException(index);
        }
        LoadImpl(is, v, index, std::index_sequence_for<Args...>{});
    }

private:
    template <size_t... Is>
    static void LoadImpl(IInputStream* is, TVar& v, ui8 index, std::index_sequence<Is...>) {
        using TLoader = void (*)(IInputStream*, TVar& v);
        constexpr TLoader loaders[] = {::NPrivate::LoadVariantAlternative<TVar, Args, Is>...};
        loaders[index](is, v);
    }
};

#endif

template <class T>
static inline void SaveLoad(IOutputStream* out, const T& t) {
    Save(out, t);
}

template <class T>
static inline void SaveLoad(IInputStream* in, T& t) {
    Load(in, t);
}

template <class S, class... Ts>
static inline void SaveMany(S* s, const Ts&... t) {
    ApplyToMany([&](const auto& v) { Save(s, v); }, t...);
}

template <class S, class... Ts>
static inline void LoadMany(S* s, Ts&... t) {
    ApplyToMany([&](auto& v) { Load(s, v); }, t...);
}

#define Y_SAVELOAD_DEFINE(...)                                    \
    inline void Save(IOutputStream* s) const {                    \
        [s](auto&&... args) {                                     \
            ::SaveMany(s, std::forward<decltype(args)>(args)...); \
        }(__VA_ARGS__);                                           \
    }                                                             \
                                                                  \
    inline void Load(IInputStream* s) {                           \
        [s](auto&&... args) {                                     \
            ::LoadMany(s, std::forward<decltype(args)>(args)...); \
        }(__VA_ARGS__);                                           \
    }                                                             \
    Y_SEMICOLON_GUARD

#define Y_SAVELOAD_DEFINE_OVERRIDE(...)                           \
    void Save(IOutputStream* s) const override {                  \
        [s](auto&&... args) {                                     \
            ::SaveMany(s, std::forward<decltype(args)>(args)...); \
        }(__VA_ARGS__);                                           \
    }                                                             \
                                                                  \
    void Load(IInputStream* s) override {                         \
        [s](auto&&... args) {                                     \
            ::LoadMany(s, std::forward<decltype(args)>(args)...); \
        }(__VA_ARGS__);                                           \
    }                                                             \
    Y_SEMICOLON_GUARD

template <class T>
struct TNonVirtualSaver {
    const T* Data;
    void Save(IOutputStream* out) const {
        Data->T::Save(out);
    }
};

template <typename S, typename T, typename... R>
inline void LoadMany(S* s, TNonVirtualSaver<T> t, R&... r) {
    const_cast<T*>(t.Data)->T::Load(s);
    ::LoadMany(s, r...);
}
