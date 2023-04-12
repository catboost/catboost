#include "chunked_memory_pool_output.h"

#include "chunked_memory_pool.h"

#include <library/cpp/yt/memory/ref.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TChunkedMemoryPoolOutput::TChunkedMemoryPoolOutput(TChunkedMemoryPool* pool, size_t chunkSize)
    : Pool_(pool)
    , ChunkSize_(chunkSize)
{ }

size_t TChunkedMemoryPoolOutput::DoNext(void** ptr)
{
    // Check if the current chunk is exhausted.
    if (Current_ == End_) {
        // Emplace the (whole) last chunk, if any.
        if (Begin_) {
            Refs_.emplace_back(Begin_, Current_);
        }
        // Allocate a new chunk.
        // Use |AllocateAligned| to get a chance to free some memory afterwards.
        // Tune the number of bytes requested from the pool to try avoid allocations.
        auto spareSize = Pool_->GetCurrentChunkSpareSize();
        auto allocationSize = (spareSize == 0 ? ChunkSize_ : std::min(ChunkSize_, spareSize));
        Begin_ = Pool_->AllocateAligned(allocationSize, /* align */ 1);
        Current_ = Begin_;
        End_ = Begin_ + allocationSize;
    }

    // Return the unused part of the current chunk.
    // This could be the whole chunk allocated above.
    *ptr = Current_;
    auto size = End_ - Current_;
    Current_ = End_;
    return size;
}

void TChunkedMemoryPoolOutput::DoUndo(size_t size)
{
    // Just rewind the current pointer.
    Current_ -= size;
    YT_VERIFY(Current_ >= Begin_);
}

std::vector<TMutableRef> TChunkedMemoryPoolOutput::Finish()
{
    // Emplace the used part of the last chunk, if any.
    if (Begin_) {
        Refs_.emplace_back(Begin_, Current_);
    }
    // Try to free the unused part of the last chunk, if possible.
    if (Current_ < End_) {
        Pool_->Free(Current_, End_);
    }
    return std::move(Refs_);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

