#pragma once

#include "ref.h"

#include <util/stream/zerocopy.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TChunkedInputStream
    : public IZeroCopyInput
{
public:
    explicit TChunkedInputStream(std::vector<TSharedRef> blocks);

    size_t DoNext(const void** ptr, size_t len) override;

private:
    const std::vector<TSharedRef> Blocks_;
    size_t Index_ = 0;
    size_t Position_ = 0;

    void SkipCompletedBlocks();
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
