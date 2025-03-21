#include "chunked_output_stream.h"

#include <util/system/sanitizers.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TChunkedOutputStream::TChunkedOutputStream(
    TRefCountedTypeCookie tagCookie,
    IMemoryUsageTrackerPtr memoryUsageTracker,
    size_t initialReserveSize,
    size_t maxReserveSize)
    : MemoryUsageTracker_(std::move(memoryUsageTracker))
    , CurrentChunkMemoryUsageGuard_(TMemoryUsageTrackerGuard::Build(MemoryUsageTracker_))
    , MaxReserveSize_(RoundUpToPage(maxReserveSize))
    , CurrentReserveSize_(RoundUpToPage(initialReserveSize))
    , CurrentChunk_(tagCookie, /*size*/ 0)
{
    YT_VERIFY(MaxReserveSize_ > 0);

    if (CurrentReserveSize_ > MaxReserveSize_) {
        CurrentReserveSize_ = MaxReserveSize_;
    }
}

std::vector<TSharedRef> TChunkedOutputStream::Finish()
{
    FinishedChunks_.push_back(TrackMemory(MemoryUsageTracker_, TSharedRef::FromBlob(std::move(CurrentChunk_))));
    CurrentChunkMemoryUsageGuard_.Release();

    YT_ASSERT(CurrentChunk_.IsEmpty());
    FinishedSize_ = 0;

    for (const auto& chunk : FinishedChunks_) {
        NSan::CheckMemIsInitialized(chunk.Begin(), chunk.Size());
    }

    return std::move(FinishedChunks_);
}

size_t TChunkedOutputStream::GetSize() const
{
    return FinishedSize_ + CurrentChunk_.Size();
}

size_t TChunkedOutputStream::GetCapacity() const
{
    return FinishedSize_ + CurrentChunk_.Capacity();
}

void TChunkedOutputStream::ReserveNewChunk(size_t spaceRequired)
{
    YT_ASSERT(CurrentChunk_.Size() == CurrentChunk_.Capacity());
    FinishedSize_ += CurrentChunk_.Size();
    FinishedChunks_.push_back(TrackMemory(MemoryUsageTracker_, TSharedRef::FromBlob(std::move(CurrentChunk_))));
    CurrentReserveSize_ = std::min(2 * CurrentReserveSize_, MaxReserveSize_);
    CurrentChunk_.Reserve(std::max(RoundUpToPage(spaceRequired), CurrentReserveSize_));
    UpdateCurrentChunkMemoryUsage();
}

void TChunkedOutputStream::DoWrite(const void* buffer, size_t length)
{
    if (CurrentChunk_.Capacity() == 0) {
        CurrentChunk_.Reserve(CurrentReserveSize_);
    }

    auto spaceAvailable = std::min(length, CurrentChunk_.Capacity() - CurrentChunk_.Size());
    CurrentChunk_.Append(buffer, spaceAvailable);

    auto spaceRequired = length - spaceAvailable;
    if (spaceRequired > 0) {
        ReserveNewChunk(spaceRequired);
        CurrentChunk_.Append(static_cast<const char*>(buffer) + spaceAvailable, spaceRequired);
    }
    UpdateCurrentChunkMemoryUsage();
}

size_t TChunkedOutputStream::DoNext(void** ptr)
{
    if (CurrentChunk_.Size() == CurrentChunk_.Capacity()) {
        if (CurrentChunk_.Capacity() == 0) {
            CurrentChunk_.Reserve(CurrentReserveSize_);
        } else {
            ReserveNewChunk(0);
        }
    }

    auto spaceAvailable = CurrentChunk_.Capacity() - CurrentChunk_.Size();
    YT_ASSERT(spaceAvailable > 0);
    *ptr = CurrentChunk_.End();
    CurrentChunk_.Resize(CurrentChunk_.Capacity(), /*initializeStorage*/ false);
    UpdateCurrentChunkMemoryUsage();
    return spaceAvailable;
}

void TChunkedOutputStream::DoUndo(size_t len)
{
    YT_VERIFY(CurrentChunk_.Size() >= len);
    CurrentChunk_.Resize(CurrentChunk_.Size() - len);
    UpdateCurrentChunkMemoryUsage();
}

char* TChunkedOutputStream::Preallocate(size_t size)
{
    size_t available = CurrentChunk_.Capacity() - CurrentChunk_.Size();
    if (available < size) {
        FinishedSize_ += CurrentChunk_.Size();
        FinishedChunks_.push_back(TrackMemory(MemoryUsageTracker_, TSharedRef::FromBlob(std::move(CurrentChunk_))));

        CurrentReserveSize_ = std::min(2 * CurrentReserveSize_, MaxReserveSize_);

        CurrentChunk_.Reserve(std::max(RoundUpToPage(size), CurrentReserveSize_));
    }
    UpdateCurrentChunkMemoryUsage();
    return CurrentChunk_.End();
}

void TChunkedOutputStream::Advance(size_t size)
{
    YT_ASSERT(CurrentChunk_.Size() + size <= CurrentChunk_.Capacity());
    CurrentChunk_.Resize(CurrentChunk_.Size() + size, false);
    UpdateCurrentChunkMemoryUsage();
}

void TChunkedOutputStream::UpdateCurrentChunkMemoryUsage()
{
    CurrentChunkMemoryUsageGuard_.SetSize(CurrentChunk_.Capacity());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
