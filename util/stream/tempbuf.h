#pragma once

#include "output.h"

#include <util/memory/tempbuf.h>

class TTempBufOutput: public IOutputStream, public TTempBuf {
public:
    inline TTempBufOutput() = default;

    explicit TTempBufOutput(size_t size)
        : TTempBuf(size)
    {
    }

    TTempBufOutput(TTempBufOutput&&) noexcept = default;
    TTempBufOutput& operator=(TTempBufOutput&&) noexcept = default;

protected:
    void DoWrite(const void* data, size_t len) override;
};
