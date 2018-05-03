#pragma once

#include "output.h"

#include <util/memory/tempbuf.h>

class TGrowingTempBufOutput: public IOutputStream, public TTempBuf {
public:
    inline TGrowingTempBufOutput() = default;

    explicit TGrowingTempBufOutput(size_t size)
        : TTempBuf(size)
    {
    }

    TGrowingTempBufOutput(TGrowingTempBufOutput&&) noexcept = default;
    TGrowingTempBufOutput& operator=(TGrowingTempBufOutput&&) noexcept = default;

protected:
    void DoWrite(const void* data, size_t len) override;
};
