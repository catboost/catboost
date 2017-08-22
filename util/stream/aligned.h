#pragma once

#include "input.h"
#include "output.h"

#include <util/system/yassert.h>
#include <util/generic/bitops.h>

/**
 * @addtogroup Streams
 * @{
 */

/**
 * Proxy input stream that provides additional functions that make reading
 * aligned data easier.
 */
class TAlignedInput: public IInputStream {
public:
    TAlignedInput(IInputStream* s)
        : Stream_(s)
        , Position_(0)
    {
    }

    /**
     * Ensures alignment of the position in the input stream by skipping
     * some input.
     *
     * @param alignment                 Alignment. Must be a power of 2.
     */
    void Align(size_t alignment = sizeof(void*)) {
        Y_ASSERT(IsPowerOf2(alignment));

        if (Position_ & (alignment - 1)) {
            size_t len = alignment - (Position_ & (alignment - 1));

            do {
                len -= DoSkip(len);
            } while (len);
        }
    }

private:
    size_t DoRead(void* ptr, size_t len) override;
    size_t DoSkip(size_t len) override;
    size_t DoReadTo(TString& st, char ch) override;
    ui64 DoReadAll(IOutputStream& out) override;

private:
    IInputStream* Stream_;
    ui64 Position_;
};

/**
 * Proxy output stream that provides additional functions that make writing
 * aligned data easier.
 */
class TAlignedOutput: public IOutputStream {
public:
    TAlignedOutput(IOutputStream* s)
        : Stream_(s)
        , Position_(0)
    {
    }

    TAlignedOutput(TAlignedOutput&&) noexcept = default;
    TAlignedOutput& operator=(TAlignedOutput&&) noexcept = default;

    size_t GetCurrentOffset() const {
        return Position_;
    }

    /**
     * Ensures alignment of the position in the output stream by writing
     * some data.
     *
     * @param alignment                 Alignment. Must be a power of 2.
     */
    void Align(size_t alignment = sizeof(void*)) {
        Y_ASSERT(IsPowerOf2(alignment));

        static char unused[sizeof(void*) * 2];
        Y_ASSERT(alignment <= sizeof(unused));

        if (Position_ & (alignment - 1)) {
            DoWrite(unused, alignment - (Position_ & (alignment - 1)));
        }
    }

private:
    void DoWrite(const void* ptr, size_t len) override;

private:
    IOutputStream* Stream_;
    ui64 Position_;
};

/** @} */
