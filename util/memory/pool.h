#pragma once

#include "alloc.h"

#include <util/system/align.h>
#include <util/system/yassert.h>
#include <util/generic/bitops.h>
#include <util/generic/utility.h>
#include <util/generic/intrlist.h>
#include <util/generic/strbuf.h>
#include <util/generic/singleton.h>

#include <new>
#include <string>
#include <utility>

/**
 * Memory pool implements a memory allocation scheme that is very fast, but
 * limited in its usage.
 *
 * A common use case is when you want to allocate a bunch of small objects, and
 * then release them all at some point of your program. Using memory pool, you
 * can just drop them off into oblivion without calling any destructors,
 * provided that all associated memory was allocated on the pool.
 */
class TMemoryPool {
private:
    using TBlock = IAllocator::TBlock;

    class TChunk: public TIntrusiveListItem<TChunk> {
    public:
        inline TChunk(size_t len = 0) noexcept
            : Cur_((char*)(this + 1))
            , Left_(len)
        {
            Y_ASSERT((((size_t)Cur_) % PLATFORM_DATA_ALIGN) == 0);
        }

        inline void* Allocate(size_t len) noexcept {
            if (Left_ >= len) {
                char* ret = Cur_;

                Cur_ += len;
                Left_ -= len;

                return ret;
            }

            return nullptr;
        }

        inline void* Allocate(size_t len, size_t align) noexcept {
            size_t pad = AlignUp(Cur_, align) - Cur_;

            void* ret = Allocate(pad + len);
            if (ret) {
                return static_cast<char*>(ret) + pad;
            }

            return nullptr;
        }

        inline size_t BlockLength() const noexcept {
            return (Cur_ + Left_) - (char*)this;
        }

        inline size_t Used() const noexcept {
            return Cur_ - (const char*)this;
        }

        inline size_t Left() const noexcept {
            return Left_;
        }

        inline const char* Data() const noexcept {
            return (const char*)(this + 1);
        }

        inline char* Data() noexcept {
            return (char*)(this + 1);
        }

        inline size_t DataSize() const noexcept {
            return Cur_ - Data();
        }

        inline void ResetChunk() noexcept {
            size_t total = DataSize() + Left();
            Cur_ = Data();
            Left_ = total;
        }

    private:
        char* Cur_;
        size_t Left_;
    };

    using TChunkList = TIntrusiveList<TChunk>;

public:
    class IGrowPolicy {
    public:
        virtual ~IGrowPolicy() = default;

        virtual size_t Next(size_t prev) const noexcept = 0;
    };

    class TLinearGrow: public IGrowPolicy {
    public:
        size_t Next(size_t prev) const noexcept override {
            return prev;
        }

        static IGrowPolicy* Instance() noexcept;
    };

    class TExpGrow: public IGrowPolicy {
    public:
        size_t Next(size_t prev) const noexcept override {
            return prev * 2;
        }

        static IGrowPolicy* Instance() noexcept;
    };

    struct TOptions {
        bool RoundUpToNextPowerOfTwo;
        TOptions()
            : RoundUpToNextPowerOfTwo(true)
        {
        }
    };

    // When a bookmark is destroyed, the memory pool returns to the state when the bookmark was created.
    class TBookmark {
    public:
        inline TBookmark(TMemoryPool& memoryPool)
            : OwnerPoolRef_(memoryPool)
            , BookmarkChunk_(memoryPool.Current_)
            , BookmarkChunkDataSize_(memoryPool.Current_->DataSize())
        {
        }

        inline ~TBookmark() {
            OwnerPoolRef_.Current_->ResetChunk();
            if (OwnerPoolRef_.Current_ == BookmarkChunk_) {
                Y_UNUSED(BookmarkChunk_->Allocate(BookmarkChunkDataSize_));
            }
        }

    private:
        TMemoryPool& OwnerPoolRef_;
        TMemoryPool::TChunk* BookmarkChunk_;
        size_t BookmarkChunkDataSize_;
    };

    inline TMemoryPool(size_t initial, IGrowPolicy* grow = TExpGrow::Instance(), IAllocator* alloc = TDefaultAllocator::Instance(), const TOptions& options = TOptions())
        : Current_(&Empty_)
        , BlockSize_(initial)
        , GrowPolicy_(grow)
        , Alloc_(alloc)
        , Options_(options)
        , Origin_(initial)
        , MemoryAllocatedBeforeCurrent_(0)
        , MemoryWasteBeforeCurrent_(0)
    {
    }

    inline ~TMemoryPool() {
        Clear();
    }

    inline void* Allocate(size_t len) {
        return RawAllocate(AlignUp<size_t>(len, PLATFORM_DATA_ALIGN));
    }

    inline void* Allocate(size_t len, size_t align) {
        return RawAllocate(AlignUp<size_t>(len, PLATFORM_DATA_ALIGN), align);
    }

    template <typename T>
    inline T* Allocate() {
        return (T*)this->Allocate(sizeof(T), alignof(T));
    }

    template <typename T>
    inline T* Allocate(size_t align) {
        return (T*)this->Allocate(sizeof(T), Max(align, alignof(T)));
    }

    template <typename T>
    inline T* AllocateArray(size_t count) {
        return (T*)this->Allocate(sizeof(T) * count, alignof(T));
    }

    template <typename T>
    inline T* AllocateArray(size_t count, size_t align) {
        return (T*)this->Allocate(sizeof(T) * count, Max(align, alignof(T)));
    }

    template <typename T>
    inline T* AllocateZeroArray(size_t count) {
        T* ptr = AllocateArray<T>(count);
        memset(ptr, 0, count * sizeof(T));
        return ptr;
    }

    template <typename T>
    inline T* AllocateZeroArray(size_t count, size_t align) {
        T* ptr = AllocateArray<T>(count, align);
        memset(ptr, 0, count * sizeof(T));
        return ptr;
    }

    template <typename T, typename... Args>
    inline T* New(Args&&... args) {
        return new (Allocate<T>()) T(std::forward<Args>(args)...);
    }

    template <typename T>
    inline T* NewArray(size_t count) {
        T* arr = (T*)AllocateArray<T>(count);

        for (T *ptr = arr, *end = arr + count; ptr != end; ++ptr) {
            new (ptr) T;
        }

        return arr;
    }

    template <typename TChar>
    inline TChar* Append(const TChar* str) {
        return Append(str, std::char_traits<TChar>::length(str) + 1); // include terminating zero byte
    }

    template <typename TChar>
    inline TChar* Append(const TChar* str, size_t len) {
        TChar* ret = AllocateArray<TChar>(len);

        std::char_traits<TChar>::copy(ret, str, len);
        return ret;
    }

    template <typename TChar>
    inline TBasicStringBuf<TChar> AppendString(const TBasicStringBuf<TChar>& buf) {
        return TBasicStringBuf<TChar>(Append(buf.data(), buf.size()), buf.size());
    }

    template <typename TChar>
    inline TBasicStringBuf<TChar> AppendCString(const TBasicStringBuf<TChar>& buf) {
        TChar* ret = static_cast<TChar*>(Allocate((buf.size() + 1) * sizeof(TChar)));

        std::char_traits<TChar>::copy(ret, buf.data(), buf.size());
        *(ret + buf.size()) = 0;
        return TBasicStringBuf<TChar>(ret, buf.size());
    }

    inline size_t Available() const noexcept {
        return Current_->Left();
    }

    inline void Clear() noexcept {
        DoClear(false);
    }

    inline void ClearKeepFirstChunk() noexcept {
        DoClear(true);
    }

    inline size_t ClearReturnUsedChunkCount(bool keepFirstChunk) noexcept {
        return DoClear(keepFirstChunk);
    }

    inline size_t MemoryAllocated() const noexcept {
        return MemoryAllocatedBeforeCurrent_ + (Current_ != &Empty_ ? Current_->Used() : 0);
    }

    inline size_t MemoryWaste() const noexcept {
        return MemoryWasteBeforeCurrent_ + (Current_ != &Empty_ ? Current_->Left() : 0);
    }

    template <class TOp>
    inline void Traverse(TOp& op) const noexcept {
        for (TChunkList::TConstIterator i = Chunks_.Begin(); i != Chunks_.End(); ++i) {
            op(i->Data(), i->DataSize());
        }
    }

    inline IAllocator* RealAllocator() const noexcept {
        return Alloc_;
    }

protected:
    inline void* RawAllocate(size_t len) {
        void* ret = Current_->Allocate(len);

        if (ret) {
            return ret;
        }

        AddChunk(len);

        return Current_->Allocate(len);
    }

    inline void* RawAllocate(size_t len, size_t align) {
        Y_ASSERT(align > 0);
        void* ret = Current_->Allocate(len, align);

        if (ret) {
            return ret;
        }

        AddChunk(len + align - 1);

        return Current_->Allocate(len, align);
    }

private:
    void AddChunk(size_t hint);
    size_t DoClear(bool keepfirst) noexcept;

private:
    TChunk Empty_;
    TChunk* Current_;
    size_t BlockSize_;
    IGrowPolicy* GrowPolicy_;
    IAllocator* Alloc_;
    TOptions Options_;
    TChunkList Chunks_;
    const size_t Origin_;
    size_t MemoryAllocatedBeforeCurrent_;
    size_t MemoryWasteBeforeCurrent_;
};

template <typename TPool>
struct TPoolableBase {
    inline void* operator new(size_t bytes, TPool& pool) {
        return pool.Allocate(bytes);
    }

    inline void* operator new(size_t bytes, std::align_val_t align, TPool& pool) {
        return pool.Allocate(bytes, (size_t)align);
    }

    inline void operator delete(void*, size_t) noexcept {
    }

    inline void operator delete(void*, TPool&) noexcept {
    }

private:
    inline void* operator new(size_t); // disallow default allocation
};

struct TPoolable: public TPoolableBase<TMemoryPool> {
};

class TMemoryPoolAllocator: public IAllocator {
public:
    inline TMemoryPoolAllocator(TMemoryPool* pool)
        : Pool_(pool)
    {
    }

    TBlock Allocate(size_t len) override {
        TBlock ret = {Pool_->Allocate(len), len};

        return ret;
    }

    void Release(const TBlock& block) override {
        (void)block;
    }

private:
    TMemoryPool* Pool_;
};

template <class T, class TPool>
class TPoolAllocBase {
public:
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using value_type = T;

    inline TPoolAllocBase(TPool* pool)
        : Pool_(pool)
    {
    }

    template <typename TOther>
    inline TPoolAllocBase(const TPoolAllocBase<TOther, TPool>& o)
        : Pool_(o.GetPool())
    {
    }

    inline T* allocate(size_t n) {
        return (T*)Pool_->Allocate(n * sizeof(T), alignof(T));
    }

    inline void deallocate(pointer /*p*/, size_t /*n*/) {
    }

    template <class T1>
    struct rebind {
        using other = TPoolAllocBase<T1, TPool>;
    };

    inline size_type max_size() const noexcept {
        return size_type(-1) / sizeof(T);
    }

    template <typename... Args>
    inline void construct(pointer p, Args&&... args) {
        new (p) T(std::forward<Args>(args)...);
    }

    inline void destroy(pointer p) noexcept {
        (void)p; /* Make MSVC happy. */
        p->~T();
    }

    inline TPool* GetPool() const {
        return Pool_;
    }

    inline friend bool operator==(const TPoolAllocBase& l, const TPoolAllocBase& r) {
        return l.Pool_ == r.Pool_;
    }

    inline friend bool operator!=(const TPoolAllocBase& l, const TPoolAllocBase& r) {
        return !(l == r);
    }

private:
    TPool* Pool_;
};

template <class T>
using TPoolAlloc = TPoolAllocBase<T, TMemoryPool>;

// Any type since it is supposed to be rebound anyway.
using TPoolAllocator = TPoolAlloc<int>;

template <class T>
inline bool operator==(const TPoolAlloc<T>&, const TPoolAlloc<T>&) noexcept {
    return true;
}

template <class T>
inline bool operator!=(const TPoolAlloc<T>&, const TPoolAlloc<T>&) noexcept {
    return false;
}
