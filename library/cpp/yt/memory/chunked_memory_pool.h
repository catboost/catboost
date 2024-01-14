#pragma once

#include "public.h"
#include "ref.h"

#include <library/cpp/yt/misc/port.h>

#include <util/generic/size_literals.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TDefaultChunkedMemoryPoolTag { };

// TAllocationHolder is polymorphic. So we cannot use TWithExtraSpace mixin
// because it needs the most derived type as a template argument and
// it would require GetExtraSpacePtr/GetRef methods to be virtual.

class TAllocationHolder
{
public:
    TAllocationHolder(TMutableRef ref, TRefCountedTypeCookie cookie);
    TAllocationHolder(const TAllocationHolder&) = delete;
    TAllocationHolder(TAllocationHolder&&) = default;
    virtual ~TAllocationHolder();

    void operator delete(void* ptr) noexcept;

    TMutableRef GetRef() const;

    template <class TDerived>
    static TDerived* Allocate(size_t size, TRefCountedTypeCookie cookie);

private:
    const TMutableRef Ref_;
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    const TRefCountedTypeCookie Cookie_;
#endif
};

////////////////////////////////////////////////////////////////////////////////

struct IMemoryChunkProvider
    : public TRefCounted
{
    virtual std::unique_ptr<TAllocationHolder> Allocate(size_t size, TRefCountedTypeCookie cookie) = 0;
};

DEFINE_REFCOUNTED_TYPE(IMemoryChunkProvider)

const IMemoryChunkProviderPtr& GetDefaultMemoryChunkProvider();

////////////////////////////////////////////////////////////////////////////////

class TChunkedMemoryPool
    : private TNonCopyable
{
public:
    static constexpr size_t DefaultStartChunkSize = 4_KB;
    static constexpr size_t RegularChunkSize = 36_KB - 512;

    TChunkedMemoryPool(
        TRefCountedTypeCookie tagCookie,
        IMemoryChunkProviderPtr chunkProvider,
        size_t startChunkSize = DefaultStartChunkSize);

    explicit TChunkedMemoryPool(
        TRefCountedTypeCookie tagCookie,
        size_t startChunkSize = DefaultStartChunkSize);

    TChunkedMemoryPool();

    template <class TTag>
    explicit TChunkedMemoryPool(
        TTag,
        size_t startChunkSize = DefaultStartChunkSize);

    //! Allocates #sizes bytes without any alignment.
    char* AllocateUnaligned(size_t size);

    //! Allocates #size bytes aligned with 8-byte granularity.
    char* AllocateAligned(size_t size, int align = 8);

    //! Allocates #n uninitialized instances of #T.
    template <class T>
    T* AllocateUninitialized(int n, int align = alignof(T));

    //! Allocates space and copies #src inside it.
    template <class T>
    TMutableRange<T> Capture(TRange<T> src, int align = alignof(T));

    //! Frees memory range if possible: namely, if the free region is a suffix of last allocated region.
    void Free(char* from, char* to);

    //! Marks all previously allocated small chunks as free for subsequent allocations but
    //! does not deallocate them.
    //! Purges all large blocks.
    void Clear();

    //! Purges all allocated memory, including small chunks.
    void Purge();

    //! Returns the number of allocated bytes.
    size_t GetSize() const;

    //! Returns the number of reserved bytes.
    size_t GetCapacity() const;

    //! Returns the number of bytes that can be acquired in the current chunk
    //! without additional allocations.
    size_t GetCurrentChunkSpareSize() const;

    //! Moves all the allocated memory from other memory pool to the current one.
    //! The other pool becomes empty, like after Purge() call.
    void Absorb(TChunkedMemoryPool&& other);

private:
    const TRefCountedTypeCookie TagCookie_;
    // A common usecase is to construct TChunkedMemoryPool with the default
    // memory chunk provider. The latter is ref-counted and is shared between
    // a multitude of TChunkedMemoryPool instances. This could potentially
    // lead to a contention over IMemoryChunkProvider's ref-counter.
    // To circumvent this, we keep both an owning (#ChunkProviderHolder_) and
    // a non-owning (#ChunkProvider_) reference to the underlying provider.
    // In case of the default chunk provider, the owning reference is not used.
    const IMemoryChunkProviderPtr ChunkProviderHolder_;
    IMemoryChunkProvider* const ChunkProvider_;

    int NextChunkIndex_ = 0;
    size_t NextSmallSize_;

    size_t Size_ = 0;
    size_t Capacity_ = 0;

    // Chunk memory layout:
    //   |AAAA|....|UUUU|
    // Legend:
    //   A aligned allocations
    //   U unaligned allocations
    //   . free zone
    char* FreeZoneBegin_;
    char* FreeZoneEnd_;

    std::vector<std::unique_ptr<TAllocationHolder>> Chunks_;
    std::vector<std::unique_ptr<TAllocationHolder>> OtherBlocks_;

    void Initialize(size_t startChunkSize);

    char* AllocateUnalignedSlow(size_t size);
    char* AllocateAlignedSlow(size_t size, int align);
    char* AllocateSlowCore(size_t size);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CHUNKED_MEMORY_POOL_INL_H_
#include "chunked_memory_pool-inl.h"
#undef CHUNKED_MEMORY_POOL_INL_H_
