#include "chunked_memory_allocator.h"
#include "serialize.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

const i64 TChunkedMemoryAllocator::DefaultChunkSize = 4096;
const double TChunkedMemoryAllocator::DefaultMaxSmallBlockSizeRatio = 0.25;

////////////////////////////////////////////////////////////////////////////////

TChunkedMemoryAllocator::TChunkedMemoryAllocator(
    i64 chunkSize,
    double maxSmallBlockSizeRatio,
    TRefCountedTypeCookie tagCookie)
    : ChunkSize_(chunkSize)
    , MaxSmallBlockSize_(static_cast<i64>(ChunkSize_ * maxSmallBlockSizeRatio))
    , TagCookie_(tagCookie)
{  }

TSharedMutableRef TChunkedMemoryAllocator::AllocateUnalignedSlow(i64 size)
{
    auto large = AllocateSlowCore(size);
    if (large) {
        return large;
    }
    return AllocateUnaligned(size);
}

TSharedMutableRef TChunkedMemoryAllocator::AllocateAlignedSlow(i64 size, int align)
{
    // NB: Do not rely on any particular alignment of chunks.
    auto large = AllocateSlowCore(size + align);
    if (large) {
        auto* alignedBegin = AlignUp(large.Begin(), align);
        return large.Slice(alignedBegin, alignedBegin + size);
    }
    return AllocateAligned(size, align);
}

TSharedMutableRef TChunkedMemoryAllocator::AllocateSlowCore(i64 size)
{
    if (size > MaxSmallBlockSize_) {
        return TSharedMutableRef::Allocate(size, {.InitializeStorage = false}, TagCookie_);
    }

    Chunk_ = TSharedMutableRef::Allocate(ChunkSize_, {.InitializeStorage = false}, TagCookie_);
    FreeZoneBegin_ = Chunk_.Begin();
    FreeZoneEnd_ = Chunk_.End();

    return TSharedMutableRef();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
