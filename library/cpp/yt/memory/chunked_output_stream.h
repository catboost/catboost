#pragma once

#include "blob.h"

#include <library/cpp/yt/memory/ref.h>

#include <util/stream/zerocopy_output.h>

#include <util/generic/size_literals.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TDefaultChunkedOutputStreamTag
{ };

class TChunkedOutputStream
    : public IZeroCopyOutput
{
public:
    explicit TChunkedOutputStream(
        TRefCountedTypeCookie tagCookie = GetRefCountedTypeCookie<TDefaultChunkedOutputStreamTag>(),
        size_t initialReserveSize = 4_KB,
        size_t maxReserveSize = 64_KB);

    TChunkedOutputStream(TChunkedOutputStream&&) = default;
    TChunkedOutputStream& operator=(TChunkedOutputStream&&) = default;

    //! Returns a sequence of written chunks.
    //! The stream is no longer usable after this call.
    std::vector<TSharedRef> Finish();

    //! Returns the number of bytes actually written.
    size_t GetSize() const;

    //! Returns the number of bytes actually written plus unused capacity in the
    //! last chunk.
    size_t GetCapacity() const;

    //! Returns a pointer to a contiguous memory block of a given #size.
    //! Do not forget to call #Advance after use.
    char* Preallocate(size_t size);

    //! Marks #size bytes (which were previously preallocated) as used.
    void Advance(size_t size);

private:
    size_t MaxReserveSize_;
    size_t CurrentReserveSize_;

    size_t FinishedSize_ = 0;

    TBlob CurrentChunk_;
    std::vector<TSharedRef> FinishedChunks_;


    void ReserveNewChunk(size_t spaceRequired);

    void DoWrite(const void* buf, size_t len) override;
    size_t DoNext(void** ptr) override;
    void DoUndo(size_t len) override;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
