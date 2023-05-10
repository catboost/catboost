#ifndef FREE_LIST_INL_H_
#error "Direct inclusion of this file is not allowed, include free_list.h"
// For the sake of sane code completion.
#include "free_list.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// DCAS is supported in Clang with option -mcx16, is not supported in GCC. See following links.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=84522
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80878

template <class T1, class T2>
Y_FORCE_INLINE bool CompareAndSet(
    TAtomicUint128* atomic,
    T1& expected1,
    T2& expected2,
    T1 new1,
    T2 new2)
{
#if defined(__x86_64__)
    bool success;
    __asm__ __volatile__
    (
        "lock cmpxchg16b %1\n"
        "setz %0"
        : "=q"(success)
        , "+m"(*atomic)
        , "+a"(expected1)
        , "+d"(expected2)
        : "b"(new1)
        , "c"(new2)
        : "cc"
    );
    return success;
#elif defined(__arm64__) || (defined(__aarch64__) && defined(RTE_ARM_FEATURE_ATOMICS))
    register ui64 x0 __asm("x0") = (ui64)expected1;
    register ui64 x1 __asm("x1") = (ui64)expected2;
    register ui64 x2 __asm("x2") = (ui64)new1;
    register ui64 x3 __asm("x3") = (ui64)new2;
    ui64 old1 = (ui64)expected1;
    ui64 old2 = (ui64)expected2;
    asm volatile
    (
#if defined(RTE_CC_CLANG)
        ".arch armv8-a+lse\n"
#endif
        "caspal %[old0], %[old1], %[upd0], %[upd1], [%[dst]]"
        : [old0] "+r" (x0)
        , [old1] "+r" (x1)
        : [upd0] "r" (x2)
        , [upd1] "r" (x3)
        , [dst] "r" (atomic)
        : "memory"
    );
    expected1 = (T1)x0;
    expected2 = (T2)x1;
    return x0 == old1 && x1 == old2;
#elif defined(__aarch64__)
    ui64 exp1 = reinterpret_cast<ui64>(expected1);
    ui64 exp2 = reinterpret_cast<ui64>(expected2);
    ui32 fail = 0;

    do {
        ui64 current1 = 0;
        ui64 current2 = 0;
        asm volatile (
            "ldaxp %[cur1], %[cur2], [%[src]]"
            : [cur1] "=r" (current1)
            , [cur2] "=r" (current2)
            : [src] "r" (atomic)
            : "memory"
        );

        if (current1 != exp1 || current2 != exp2) {
            expected1 = reinterpret_cast<T1>(current1);
            expected2 = reinterpret_cast<T2>(current2);
            return false;
        }

        asm volatile (
            "stlxp %w[fail], %[new1], %[new2], [%[dst]]"
            : [fail] "=&r" (fail)
            : [new1] "r" (new1)
            , [new2] "r" (new2)
            , [dst] "r" (atomic)
            : "memory"
        );

    } while (Y_UNLIKELY(fail));
    return true;
#else
#    error Unsupported platform
#endif
}

////////////////////////////////////////////////////////////////////////////////

template <class TItem>
TFreeList<TItem>::THead::THead(TItem* pointer)
    : Pointer(pointer)
{ }

template <class TItem>
TFreeList<TItem>::TFreeList()
    : Head_()
{ }

template <class TItem>
TFreeList<TItem>::TFreeList(TFreeList<TItem>&& other)
    : Head_(other.ExtractAll())
{ }

template <class TItem>
TFreeList<TItem>::~TFreeList()
{
    YT_VERIFY(IsEmpty());
}

template <class TItem>
template <class TPredicate>
Y_NO_SANITIZE("thread")
bool TFreeList<TItem>::PutIf(TItem* head, TItem* tail, TPredicate predicate)
{
    auto* current = Head_.Pointer.load(std::memory_order::relaxed);
    auto epoch = Head_.Epoch.load(std::memory_order::relaxed);

    while (predicate(current)) {
        tail->Next.store(current, std::memory_order::release);
        if (CompareAndSet(&AtomicHead_, current, epoch, head, epoch + 1)) {
            return true;
        }
    }

    tail->Next.store(nullptr, std::memory_order::release);

    return false;
}


template <class TItem>
Y_NO_SANITIZE("thread")
void TFreeList<TItem>::Put(TItem* head, TItem* tail)
{
    auto* current = Head_.Pointer.load(std::memory_order::relaxed);
    auto epoch = Head_.Epoch.load(std::memory_order::relaxed);

    do {
        tail->Next.store(current, std::memory_order::release);
    } while (!CompareAndSet(&AtomicHead_, current, epoch, head, epoch + 1));
}

template <class TItem>
void TFreeList<TItem>::Put(TItem* item)
{
    Put(item, item);
}

template <class TItem>
Y_NO_SANITIZE("thread")
TItem* TFreeList<TItem>::Extract()
{
    auto* current = Head_.Pointer.load(std::memory_order::relaxed);
    auto epoch = Head_.Epoch.load(std::memory_order::relaxed);

    while (current) {
        // If current node is already extracted by other thread
        // there can be any writes at address &current->Next.
        // The only guaranteed thing is that address is valid (memory is not freed).
        auto next = current->Next.load(std::memory_order::acquire);
        if (CompareAndSet(&AtomicHead_, current, epoch, next, epoch + 1)) {
            current->Next.store(nullptr, std::memory_order::release);
            return current;
        }
    }

    return nullptr;
}

template <class TItem>
TItem* TFreeList<TItem>::ExtractAll()
{
    auto* current = Head_.Pointer.load(std::memory_order::relaxed);
    auto epoch = Head_.Epoch.load(std::memory_order::relaxed);

    while (current) {
        if (CompareAndSet<TItem*, size_t>(&AtomicHead_, current, epoch, nullptr, epoch + 1)) {
            return current;
        }
    }

    return nullptr;
}

template <class TItem>
bool TFreeList<TItem>::IsEmpty() const
{
    return Head_.Pointer.load() == nullptr;
}

template <class TItem>
void TFreeList<TItem>::Append(TFreeList<TItem>& other)
{
    auto* head = other.ExtractAll();

    if (!head) {
        return;
    }

    auto* tail = head;
    while (tail->Next) {
        tail = tail->Next;
    }

    Put(head, tail);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
