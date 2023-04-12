#ifndef CHUNKED_MEMORY_ALLOCATOR_INL_H_
#error "Direct inclusion of this file is not allowed, include chunked_memory_allocator.h"
// For the sake of sane code completion.
#include "chunked_memory_allocator.h"
#endif

#include "serialize.h"

#include <util/system/align.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline TSharedMutableRef TChunkedMemoryAllocator::AllocateUnaligned(i64 size)
{
    // Fast path.
    if (FreeZoneEnd_ >= FreeZoneBegin_ + size) {
        FreeZoneEnd_ -= size;
        return Chunk_.Slice(FreeZoneEnd_, FreeZoneEnd_ + size);
    }

    // Slow path.
    return AllocateUnalignedSlow(size);
}

inline TSharedMutableRef TChunkedMemoryAllocator::AllocateAligned(i64 size, int align)
{
    // NB: This can lead to FreeZoneBegin_ >= FreeZoneEnd_ in which case the chunk is full.
    FreeZoneBegin_ = AlignUp(FreeZoneBegin_, align);

    // Fast path.
    if (FreeZoneBegin_ + size <= FreeZoneEnd_) {
        FreeZoneBegin_ += size;
        return Chunk_.Slice(FreeZoneBegin_ - size, FreeZoneBegin_);
    }

    // Slow path.
    return AllocateAlignedSlow(size, align);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
