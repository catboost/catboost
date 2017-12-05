#pragma once

#include <util/generic/fwd.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/generic/typetraits.h>
#include <util/stream/output.h>
#include <util/stream/input.h>

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

template <class T>
static inline void LoadPodType(IInputStream* rh, T& t) {
    const size_t res = rh->Load(&t, sizeof(T));

    if (res != sizeof(T)) {
        ythrow TLoadEOF() << "can not load pod type(" << sizeof(T) << ", " << res << " bytes)";
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

    if (res != len) {
        ythrow TLoadEOF() << "can not load pod array(" << len << ", " << res << " bytes)";
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

template <class T, bool newStyle>
struct TSerializerMethodSelector {
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
struct TSerializerMethodSelector<T, false> {
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

namespace NHasSaveLoad {
    Y_HAS_MEMBER(SaveLoad);
}

template <class T>
struct TSerializerTakingIntoAccountThePodType<T, false>: public TSerializerMethodSelector<T, NHasSaveLoad::THasSaveLoad<T>::Result> {
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
        ythrow yexception() << "It's not allowed to save size which is more than or equal to max value of ui32";
    }
}

static inline size_t LoadSize(IInputStream* rh) {
    ui32 oldVerSize;
    ui64 newVerSize;
    ::Load(rh, oldVerSize);
    if (oldVerSize != 0xffffffff) {
        return oldVerSize;
    } else {
        ythrow yexception() << "It's not allowed to load size which is more than or equal to max value of ui32";
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

template <size_t I, typename... TArgs>
struct TTupleSerializer {
    static inline void Save(IOutputStream* stream, const std::tuple<TArgs...>& tuple) {
        ::Save(stream, std::get<I>(tuple));
        TTupleSerializer<I - 1, TArgs...>::Save(stream, tuple);
    }

    static inline void Load(IInputStream* stream, std::tuple<TArgs...>& tuple) {
        ::Load(stream, std::get<I>(tuple));
        TTupleSerializer<I - 1, TArgs...>::Load(stream, tuple);
    }
};

template <typename... TArgs>
struct TTupleSerializer<0, TArgs...> {
    static inline void Save(IOutputStream* stream, const std::tuple<TArgs...>& tuple) {
        ::Save(stream, std::get<0>(tuple));
    }

    static inline void Load(IInputStream* stream, std::tuple<TArgs...>& tuple) {
        ::Load(stream, std::get<0>(tuple));
    }
};

template <typename... TArgs>
struct TSerializer<std::tuple<TArgs...>> {
    static inline void Save(IOutputStream* stream, const std::tuple<TArgs...>& tuple) {
        TTupleSerializer<sizeof...(TArgs) - 1, TArgs...>::Save(stream, tuple);
    }

    static inline void Load(IInputStream* stream, std::tuple<TArgs...>& tuple) {
        TTupleSerializer<sizeof...(TArgs) - 1, TArgs...>::Load(stream, tuple);
    }
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
        P_ = TBase::S_.begin();
    }

    inline void Insert(const TValue& v) {
        P_ = TBase::S_.insert(P_, v);
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

template <typename... Args>
struct TSerializer<TVariant<Args...>> {
    using TVar = TVariant<Args...>;

    static_assert(sizeof...(Args) < 256, "We use ui8 to store tag");

    struct TTypeListTagLoader {
        IInputStream* InputStream;
        TVar& Target;
        int Tag;

        void Load(TTypeList<>) {
            ythrow TLoadEOF() << "Unexpected tag value " << Tag << " while loading TVariant";
        }

        template <class THead, class... TTail>
        void Load(TTypeList<THead, TTail...>) {
            if (Tag == TVar::template TagOf<THead>()) {
                THead x;
                ::Load(InputStream, x);
                Target = std::move(x);
            } else {
                Load(TTypeList<TTail...>{});
            }
        }
    };

    static void Save(IOutputStream* os, const TVar& v) {
        ::Save<ui8>(os, v.Tag());
        v.Visit([os](const auto& data) {
            ::Save(os, data);
        });
    }

    static void Load(IInputStream* is, TVar& v) {
        ui8 tag;
        ::Load(is, tag);
        TTypeListTagLoader{is, v, static_cast<int>(tag)}.Load(TTypeList<Args...>{});
    }
};

template <class T>
static inline void SaveLoad(IOutputStream* out, const T& t) {
    Save(out, t);
}

template <class T>
static inline void SaveLoad(IInputStream* in, T& t) {
    Load(in, t);
}

template <typename S>
static inline void SaveMany(S*) {
}

template <typename S, typename T, typename... R>
static inline void SaveMany(S* s, const T& t, const R&... r) {
    Save(s, t);
    ::SaveMany(s, r...);
}

template <typename S>
static inline void LoadMany(S*) {
}

template <typename S, typename T, typename... R>
static inline void LoadMany(S* s, T& t, R&... r) {
    Load(s, t);
    ::LoadMany(s, r...);
}

#define Y_SAVELOAD_DEFINE(...)                 \
    inline void Save(IOutputStream* s) const { \
        ::SaveMany(s, __VA_ARGS__);            \
    }                                          \
                                               \
    inline void Load(IInputStream* s) {        \
        ::LoadMany(s, __VA_ARGS__);            \
    }
