#pragma once

#include "public.h"

#include <util/system/compiler.h>

#include <atomic>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TFreeListItemBase
{
    std::atomic<T*> Next = nullptr;
};

// DCAS is supported in Clang with option -mcx16, is not supported in GCC. See following links.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=84522
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80878

using TAtomicUint128 = volatile unsigned __int128  __attribute__((aligned(16)));

template <class TItem>
class TFreeList
{
private:
    struct THead
    {
        std::atomic<TItem*> Pointer = {nullptr};
        std::atomic<size_t> PopCount = 0;

        THead() = default;

        explicit THead(TItem* pointer);

    };

    union
    {
        THead Head_;
        TAtomicUint128 AtomicHead_;
    };

    // Avoid false sharing.
    char Padding[CacheLineSize - sizeof(TAtomicUint128)];

public:
    TFreeList();

    TFreeList(TFreeList&& other);

    ~TFreeList();

    template <class TPredicate>
    bool PutIf(TItem* head, TItem* tail, TPredicate predicate);

    void Put(TItem* head, TItem* tail);

    void Put(TItem* item);

    TItem* Extract();

    TItem* ExtractAll();

    bool IsEmpty() const;

    void Append(TFreeList& other);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define FREE_LIST_INL_H_
#include "free_list-inl.h"
#undef FREE_LIST_INL_H_
