#pragma once

#include "public.h"
#include "tagged_ptr.h"

#include <atomic>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

#if defined(_64_)

using TFreeListPackedPairComponent = ui64;
using TFreeListPackedPair = volatile unsigned __int128  __attribute__((aligned(16)));

#elif defined(_32_)

using TFreeListPackedPairComponent = ui32;
using TFreeListPackedPair = std::atomic<ui64>;

#else
    #error Unsupported platform
#endif

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TFreeListItemBase
{
    std::atomic<T*> Next = nullptr;
};

template <class TItem>
class TFreeList
{
private:
    using TEpoch = NDetail::TFreeListPackedPairComponent;

    struct THead
    {
        std::atomic<TItem*> Pointer = nullptr;
        std::atomic<TEpoch> Epoch = 0;

        THead() = default;

        explicit THead(TItem* pointer);
    };

    static_assert(sizeof(THead) == sizeof(NDetail::TFreeListPackedPair));

    union
    {
        THead Head_;
        NDetail::TFreeListPackedPair PackedHead_;
    };

    // Avoid false sharing.
    [[maybe_unused]] char Padding_[CacheLineSize - sizeof(THead)];

public:
    TFreeList();
    TFreeList(TFreeList&& other) noexcept;

    ~TFreeList();

    template <class TPredicate>
    bool PutIf(TItem* head, TItem* tail, TPredicate predicate);
    void Put(TItem* head, TItem* tail);
    void Put(TItem* item);

    TItem* Extract();
    TItem* ExtractAll();

    void Append(TFreeList& other);

    bool IsEmpty() const;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define FREE_LIST_INL_H_
#include "free_list-inl.h"
#undef FREE_LIST_INL_H_
