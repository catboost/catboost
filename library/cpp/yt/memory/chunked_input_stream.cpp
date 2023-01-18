#include "chunked_input_stream.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TChunkedInputStream::TChunkedInputStream(std::vector<TSharedRef> blocks)
    : Blocks_(std::move(blocks))
{ }

size_t TChunkedInputStream::DoNext(const void** ptr, size_t len)
{
    SkipCompletedBlocks();
    if (Index_ == Blocks_.size()) {
        *ptr = nullptr;
        return 0;
    }
    *ptr = Blocks_[Index_].Begin() + Position_;
    size_t toSkip = std::min(Blocks_[Index_].Size() - Position_, len);
    Position_ += toSkip;
    return toSkip;
}

void TChunkedInputStream::SkipCompletedBlocks()
{
    while (Index_ < Blocks_.size() && Position_ == Blocks_[Index_].Size()) {
        Index_ += 1;
        Position_ = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
