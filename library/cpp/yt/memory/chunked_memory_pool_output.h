#pragma once

#include "public.h"
#include "ref.h"

#include <util/stream/zerocopy_output.h>

#include <util/generic/size_literals.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TChunkedMemoryPoolOutput
    : public IZeroCopyOutput
{
public:
    static constexpr size_t DefaultChunkSize = 4_KB;

    explicit TChunkedMemoryPoolOutput(
        TChunkedMemoryPool* pool,
        size_t chunkSize = DefaultChunkSize);

    std::vector<TMutableRef> Finish();

private:
    size_t DoNext(void** ptr) override;
    void DoUndo(size_t size) override;

private:
    TChunkedMemoryPool* const Pool_;
    const size_t ChunkSize_;

    char* Begin_ = nullptr;
    char* Current_ = nullptr;
    char* End_ = nullptr;
    std::vector<TMutableRef> Refs_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

