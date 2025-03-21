#pragma once

#include "ref.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TDefaultChunkedMemoryAllocatorTag
{ };

//! Mimics TChunkedMemoryPool but acts as an allocator returning shared refs.
class TChunkedMemoryAllocator
    : private TNonCopyable
{
public:
    static const i64 DefaultChunkSize;
    static const double DefaultMaxSmallBlockSizeRatio;

    explicit TChunkedMemoryAllocator(
        i64 chunkSize = DefaultChunkSize,
        double maxSmallBlockSizeRatio = DefaultMaxSmallBlockSizeRatio,
        TRefCountedTypeCookie tagCookie = GetRefCountedTypeCookie<TDefaultChunkedMemoryAllocatorTag>());

    template <class TTag>
    explicit TChunkedMemoryAllocator(
        TTag /*tag*/ = TTag(),
        i64 chunkSize = DefaultChunkSize,
        double maxSmallBlockSizeRatio = DefaultMaxSmallBlockSizeRatio)
        : TChunkedMemoryAllocator(
            chunkSize,
            maxSmallBlockSizeRatio,
            GetRefCountedTypeCookie<TTag>())
    {
        static_assert(IsEmptyClass<TTag>());
    }

    //! Allocates #sizes bytes without any alignment.
    TSharedMutableRef AllocateUnaligned(i64 size);

    //! Allocates #size bytes aligned with 8-byte granularity.
    TSharedMutableRef AllocateAligned(i64 size, int align = 8);

private:
    const i64 ChunkSize_;
    const i64 MaxSmallBlockSize_;
    const TRefCountedTypeCookie TagCookie_;

    // Zero-sized arrays are prohibited by C++ standard
    // Also even if supported they actually occupy some space (usually 1 byte)
    char EmptyBuf_[1];

    // Chunk memory layout:
    //   |AAAA|....|UUUU|
    // Legend:
    //   A aligned allocations
    //   U unaligned allocations
    //   . free zone
    char* FreeZoneBegin_ = EmptyBuf_;
    char* FreeZoneEnd_ = EmptyBuf_;
    TSharedMutableRef Chunk_;

    TSharedMutableRef AllocateUnalignedSlow(i64 size);
    TSharedMutableRef AllocateAlignedSlow(i64 size, int align);
    TSharedMutableRef AllocateSlowCore(i64 size);

};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CHUNKED_MEMORY_ALLOCATOR_INL_H_
#include "chunked_memory_allocator-inl.h"
#undef CHUNKED_MEMORY_ALLOCATOR_INL_H_
