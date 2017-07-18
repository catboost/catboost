#pragma once

#include "pool.h"
#include "alloc.h"

#include <util/generic/utility.h>
#include <util/generic/intrlist.h>

class TFixedSizeAllocator {
    struct TAlloc: public TIntrusiveSListItem<TAlloc> {
        inline void* ToPointer() noexcept {
            return this;
        }

        static inline TAlloc* FromPointer(void* ptr) noexcept {
            return (TAlloc*)ptr;
        }

        static constexpr size_t EntitySize(size_t alloc) noexcept {
            return Max(sizeof(TAlloc), alloc);
        }

        static constexpr size_t EntityAlign(size_t align) noexcept {
            return Max(alignof(TAlloc), align);
        }

        static inline TAlloc* Construct(void* ptr) noexcept {
            return (TAlloc*)ptr;
        }
    };

public:
    using IGrowPolicy = TMemoryPool::IGrowPolicy;

    TFixedSizeAllocator(size_t allocSize, IAllocator* alloc)
        : TFixedSizeAllocator(allocSize, alignof(TAlloc), TMemoryPool::TExpGrow::Instance(), alloc)
    {
    }

    TFixedSizeAllocator(size_t allocSize, size_t alignSize, IAllocator* alloc)
        : TFixedSizeAllocator(allocSize, alignSize, TMemoryPool::TExpGrow::Instance(), alloc)
    {
    }

    TFixedSizeAllocator(size_t allocSize, IGrowPolicy* grow, IAllocator* alloc)
        : TFixedSizeAllocator(allocSize, alignof(TAlloc), grow, alloc)
    {
    }

    TFixedSizeAllocator(size_t allocSize, size_t alignSize, IGrowPolicy* grow, IAllocator* alloc)
        : Pool_(allocSize, grow, alloc)
        , AlignSize_(TAlloc::EntityAlign(alignSize))
        , AllocSize_(TAlloc::EntitySize(allocSize))
    {
    }

    inline void* Allocate() {
        if (Y_UNLIKELY(Free_.Empty())) {
            return Pool_.Allocate(AllocSize_, AlignSize_);
        }

        return Free_.PopFront()->ToPointer();
    }

    inline void Release(void* ptr) noexcept {
        Free_.PushFront(TAlloc::FromPointer(ptr));
    }

    inline size_t Size() const noexcept {
        return AllocSize_;
    }

private:
    TMemoryPool Pool_;
    const size_t AlignSize_;
    const size_t AllocSize_;
    TIntrusiveSList<TAlloc> Free_;
};

template <class T>
class TSmallObjAllocator {
public:
    using IGrowPolicy = TFixedSizeAllocator::IGrowPolicy;

    inline TSmallObjAllocator(IAllocator* alloc)
        : Alloc_(sizeof(T), alignof(T), alloc)
    {
    }

    inline TSmallObjAllocator(IGrowPolicy* grow, IAllocator* alloc)
        : Alloc_(sizeof(T), alignof(T), grow, alloc)
    {
    }

    inline T* Allocate() {
        return (T*)Alloc_.Allocate();
    }

    inline void Release(T* t) noexcept {
        Alloc_.Release(t);
    }

private:
    TFixedSizeAllocator Alloc_;
};

template <class T>
class TObjectFromPool {
public:
    struct THeader {
        void* Pool;
        // Can't just use T because THeader must be standard layout type for offsetof to work.
        alignas(T) char Obj[sizeof(T)];
    };
    using TPool = TSmallObjAllocator<THeader>;

    inline void* operator new(size_t, TPool* pool) {
        THeader* ret = pool->Allocate();
        ret->Pool = pool;
        return &ret->Obj;
    }

    inline void operator delete(void* ptr, size_t) noexcept {
        DoDelete(ptr);
    }

    inline void operator delete(void* ptr, TPool*) noexcept {
        /*
         * this delete operator can be called automagically by compiler
         */

        DoDelete(ptr);
    }

private:
    static inline void DoDelete(void* ptr) noexcept {
        static_assert(std::is_standard_layout<THeader>::value, "offsetof is only defined for standard layout types");
        THeader* header = (THeader*)((char*)ptr - offsetof(THeader, Obj));
        ((TPool*)header->Pool)->Release(header);
    }
};
