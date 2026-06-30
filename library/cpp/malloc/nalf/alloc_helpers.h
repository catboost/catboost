#pragma once

#include "nalf_alloc.h"

struct TNoHeapAlloc {
    void* operator new(size_t);
    void* operator new[](size_t);

    // implemented and available for gcc virtual destructors
protected:
    void operator delete(void*) {
        Y_ABORT();
    }
    void operator delete[](void*) {
        Y_ABORT();
    }

    void operator delete(void*, const std::nothrow_t&) {
        Y_ABORT();
    }
    void operator delete[](void*, const std::nothrow_t&) {
        Y_ABORT();
    }
};

template <typename TFinal>
struct TSystemAllocHelper {
    // override new/delete to happen with system-alloc, system-free, useful for structures which could be allocated before allocator setup is complete
    // (allocator themself)

    void* operator new(size_t sz) {
        Y_ABORT_UNLESS(sz == sizeof(TFinal));
        return NNumaAwareLockFreeAllocator::SystemAllocation(sz);
    }

    void* operator new[](size_t sz) {
        Y_ABORT_UNLESS(sz == sizeof(TFinal));
        return NNumaAwareLockFreeAllocator::SystemAllocation(sz);
    }

    void operator delete(void* mem) {
        NNumaAwareLockFreeAllocator::SystemFree(mem, sizeof(TFinal));
    }

    void operator delete[](void* mem) {
        NNumaAwareLockFreeAllocator::SystemFree(mem, sizeof(TFinal));
    }

    void operator delete(void* mem, const std::nothrow_t&) {
        NNumaAwareLockFreeAllocator::SystemFree(mem, sizeof(TFinal));
    }

    void operator delete[](void* mem, const std::nothrow_t&) {
        NNumaAwareLockFreeAllocator::SystemFree(mem, sizeof(TFinal));
    }
};

template <NNumaAwareLockFreeAllocator::TAllocHint::EHint H>
struct TWithNalfAlloc {
#if !defined(NALF_FORCE_MALLOC_FREE)
    // override new/delete to happen with nalf
    void* operator new(size_t sz) {
        return NNumaAwareLockFreeAllocator::Allocate(sz, H);
    }

    void* operator new[](size_t sz) {
        return NNumaAwareLockFreeAllocator::Allocate(sz, H);
    }

    void operator delete(void* mem) {
        NNumaAwareLockFreeAllocator::Free(mem);
    }

    void operator delete[](void* mem) {
        NNumaAwareLockFreeAllocator::Free(mem);
    }

    void operator delete(void* mem, const std::nothrow_t&) {
        NNumaAwareLockFreeAllocator::Free(mem);
    }

    void operator delete[](void* mem, const std::nothrow_t&) {
        NNumaAwareLockFreeAllocator::Free(mem);
    }
#endif // NALF_FORCE_MALLOC_FREE
};

struct TWithNalfIncrementalAlloc : TWithNalfAlloc<NNumaAwareLockFreeAllocator::TAllocHint::Incremental> {};
struct TWithNalfForceIncrementalAlloc : TWithNalfAlloc<NNumaAwareLockFreeAllocator::TAllocHint::ForceIncremental> {};
struct TWithNalfChunkedAlloc : TWithNalfAlloc<NNumaAwareLockFreeAllocator::TAllocHint::Chunked> {};
struct TWithNalfForceChunkedAlloc : TWithNalfAlloc<NNumaAwareLockFreeAllocator::TAllocHint::ForceChunked> {};
struct TWithNalfSystemAlloc : TWithNalfAlloc<NNumaAwareLockFreeAllocator::TAllocHint::System> {};
struct TWithNalfForceSystemAlloc : TWithNalfAlloc<NNumaAwareLockFreeAllocator::TAllocHint::ForceSystem> {};

using TAllocSwapIncremental = NNumaAwareLockFreeAllocator::TSwapHint<NNumaAwareLockFreeAllocator::TAllocHint::Incremental>;
using TAllocSwapChunked = NNumaAwareLockFreeAllocator::TSwapHint<NNumaAwareLockFreeAllocator::TAllocHint::Chunked>;

template <typename Type, NNumaAwareLockFreeAllocator::TAllocHint::EHint Hint>
struct TNalfAllocator {
    typedef Type value_type;
    typedef Type* pointer;
    typedef const Type* const_pointer;
    typedef Type& reference;
    typedef const Type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    TNalfAllocator() noexcept = default;
    ~TNalfAllocator() noexcept = default;

    template <typename U>
    explicit TNalfAllocator(TNalfAllocator<U, Hint>) noexcept {}
    template <typename U>
    struct rebind { typedef TNalfAllocator<U, Hint> other; };

    static pointer allocate(size_type n, const void* = nullptr) {
        return static_cast<pointer>(NNumaAwareLockFreeAllocator::Allocate(n * sizeof(value_type), Hint));
    }

    static void deallocate(pointer p, size_type = 0U) {
        return NNumaAwareLockFreeAllocator::Free(p);
    }

    static constexpr size_type max_size() noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }

    template <typename U, typename... Args>
    static void construct(U* p, Args&&... args) {
        ::new ((void*)p) U(std::forward<Args>(args)...);
    }

    template <typename U>
    static void destroy(U* p) {
        return p->~U();
    }

    static pointer address(reference x) noexcept {
        return std::addressof(x);
    }

    static const_pointer address(const_reference x) noexcept {
        return std::addressof(x);
    }
};

template <typename Type>
using TNalfChunkedAllocator = TNalfAllocator<Type, NNumaAwareLockFreeAllocator::TAllocHint::Chunked>;
template <typename Type>
using TNalfIncrementalAllocator = TNalfAllocator<Type, NNumaAwareLockFreeAllocator::TAllocHint::Incremental>;
