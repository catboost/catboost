#pragma once

#include <util/system/defaults.h>

#include <stlfwd>

template <typename TCharType, typename TTraits = std::char_traits<TCharType>>
class TBasicString;

using TString = TBasicString<char>;
using TUtf16String = TBasicString<wchar16>;
using TUtf32String = TBasicString<wchar32>;

template <typename TCharType, typename TTraits = std::char_traits<TCharType>>
class TBasicStringBuf;

using TStringBuf = TBasicStringBuf<char>;
using TWtringBuf = TBasicStringBuf<wchar16>;
using TUtf32StringBuf = TBasicStringBuf<wchar32>;

// misc
class TBuffer;

// functors
template <class T = void>
struct TLess;

template <class T = void>
struct TGreater;

template <class T = void>
struct TEqualTo;

template <class T>
struct THash;

// intrusive containers
struct TIntrusiveListDefaultTag;
template <class T, class Tag = TIntrusiveListDefaultTag>
class TIntrusiveList;

template <class T, class D, class Tag = TIntrusiveListDefaultTag>
class TIntrusiveListWithAutoDelete;

template <class T, class Tag = TIntrusiveListDefaultTag>
class TIntrusiveSList;

template <class T, class C>
class TAvlTree;

template <class TValue, class TCmp>
class TRbTree;

// containers
template <class T, class A = std::allocator<T>>
class TVector;

template <class T, class A = std::allocator<T>>
class TDeque;

template <class T, class S = TDeque<T>>
class TQueue;

template <class T, class S = TVector<T>, class C = TLess<T>>
class TPriorityQueue;

template <class Key, class T, class HashFcn = THash<Key>, class EqualKey = TEqualTo<Key>, class Alloc = std::allocator<Key>>
class THashMap;

template <class Key, class T, class HashFcn = THash<Key>, class EqualKey = TEqualTo<Key>, class Alloc = std::allocator<Key>>
class THashMultiMap;

template <class Value, class HashFcn = THash<Value>, class EqualKey = TEqualTo<Value>, class Alloc = std::allocator<Value>>
class THashSet;

template <class Value, class HashFcn = THash<Value>, class EqualKey = TEqualTo<Value>, class Alloc = std::allocator<Value>>
class THashMultiSet;

template <class T, class A = std::allocator<T>>
class TList;

template <class K, class V, class Less = TLess<K>, class A = std::allocator<K>>
class TMap;

template <class K, class V, class Less = TLess<K>, class A = std::allocator<K>>
class TMultiMap;

template <class K, class L = TLess<K>, class A = std::allocator<K>>
class TSet;

template <class K, class L = TLess<K>, class A = std::allocator<K>>
class TMultiSet;

template <class T, class S = TDeque<T>>
class TStack;

template <size_t BitCount, typename TChunkType = ui64>
class TBitMap;

// autopointers
class TDelete;
class TDeleteArray;
class TFree;
class TCopyNew;

template <class T, class D = TDelete>
class TAutoPtr;

template <class T, class D = TDelete>
class THolder;

template <class T, class C, class D = TDelete>
class TRefCounted;

template <class T>
class TDefaultIntrusivePtrOps;

template <class T, class Ops>
class TSimpleIntrusiveOps;

template <class T, class Ops = TDefaultIntrusivePtrOps<T>>
class TIntrusivePtr;

template <class T, class Ops = TDefaultIntrusivePtrOps<T>>
class TIntrusiveConstPtr;

template <class T, class Ops = TDefaultIntrusivePtrOps<T>>
using TSimpleIntrusivePtr = TIntrusivePtr<T, TSimpleIntrusiveOps<T, Ops>>;

template <class T, class C, class D = TDelete>
class TSharedPtr;

template <class T, class C = TCopyNew, class D = TDelete>
class TCopyPtr;

template <class TPtr, class TCopy = TCopyNew>
class TCowPtr;

template <typename T>
class TPtrArg;

template <typename T>
using TArrayHolder = THolder<T, TDeleteArray>;

template <typename T>
using TMallocHolder = THolder<T, TFree>;

template <typename T>
using TArrayPtr = TAutoPtr<T, TDeleteArray>;

template <typename T>
using TMallocPtr = TAutoPtr<T, TFree>;

// maybe
namespace NMaybe {
    struct TPolicyUndefinedExcept;
}

template <class T, class Policy = ::NMaybe::TPolicyUndefinedExcept>
class TMaybe;

struct TGUID;

template <class T>
class TArrayRef;

template <class T>
using TConstArrayRef = TArrayRef<const T>;
