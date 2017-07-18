#pragma once

#include "stlfwd.h"

#include <util/system/defaults.h>

//misc
class TBuffer;
class TString;
class TUtf16String;

//functors
template <class T = void>
struct TLess;

template <class T = void>
struct TGreater;

template <class T = void>
struct TEqualTo;

template <class T>
struct THash;

//strings
template <class TCharType>
class TCharTraits;

template <typename TChar, typename TTraits = TCharTraits<TChar>>
class TStringBufImpl;

using TStringBuf = TStringBufImpl<char>;
using TWtringBuf = TStringBufImpl<wchar16>;

//alias for compatibility with TGenericString
template <typename TChar>
using TGenericStringBuf = TStringBufImpl<TChar>;

//intrusive containers
template <class T>
class TIntrusiveList;

template <class T, class D>
class TIntrusiveListWithAutoDelete;

template <class T>
class TIntrusiveSList;

template <class T, class C>
class TAvlTree;

template <class TValue, class TCmp>
class TRbTree;

//containers
template <class T, class A = std::allocator<T>>
class yvector;

template <class T, class A = std::allocator<T>>
class ydeque;

template <class T, class S = ydeque<T>>
class yqueue;

template <class T, class S = yvector<T>, class C = TLess<T>>
class ypriority_queue;

template <class Key, class T, class HashFcn = THash<Key>, class EqualKey = TEqualTo<Key>, class Alloc = std::allocator<T>>
class yhash;

template <class Key, class T, class HashFcn = THash<Key>, class EqualKey = TEqualTo<Key>, class Alloc = std::allocator<T>>
class yhash_mm;

template <class Value, class HashFcn = THash<Value>, class EqualKey = TEqualTo<Value>, class Alloc = std::allocator<Value>>
class yhash_set;

template <class Value, class HashFcn = THash<Value>, class EqualKey = TEqualTo<Value>, class Alloc = std::allocator<Value>>
class yhash_multiset;

template <class T, class A = std::allocator<T>>
class ylist;

template <class K, class V, class Less = TLess<K>, class A = std::allocator<K>>
class ymap;

template <class K, class V, class Less = TLess<K>, class A = std::allocator<K>>
class ymultimap;

template <class K, class L = TLess<K>, class A = std::allocator<K>>
class yset;

template <class K, class L = TLess<K>, class A = std::allocator<K>>
class ymultiset;

template <class T, class S = ydeque<T>>
class ystack;

template <size_t BitCount, typename TChunkType = ui64>
class TBitMap;

//autopointers
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

template <class T, class D = TDelete>
class TLinkedPtr;

template <class T, class C = TCopyNew, class D = TDelete>
class TCopyPtr;

template <class TPtr, class TCopy = TCopyNew>
class TCowPtr;

template <typename T>
using TArrayHolder = THolder<T, TDeleteArray>;

template <typename T>
using TMallocHolder = THolder<T, TFree>;

template <typename T>
using TArrayPtr = TAutoPtr<T, TDeleteArray>;

template <typename T>
using TMallocPtr = TAutoPtr<T, TFree>;

//maybe
namespace NMaybe {
    struct TPolicyUndefinedExcept;
}
template <class T, class Policy = ::NMaybe::TPolicyUndefinedExcept>
class TMaybe;

struct TGUID;

template <class... Ts>
class TVariant;
