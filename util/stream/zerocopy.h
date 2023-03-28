#pragma once

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/generic/ylimits.h>

#include "input.h"

class IOutputStream;

/**
 * @addtogroup Streams
 * @{
 */

/**
 * Input stream with direct access to the input buffer.
 *
 * Derived classes must implement `DoNext` method.
 */
class IZeroCopyInput: public IInputStream {
public:
    IZeroCopyInput() noexcept = default;
    ~IZeroCopyInput() override;

    IZeroCopyInput(IZeroCopyInput&&) noexcept = default;
    IZeroCopyInput& operator=(IZeroCopyInput&&) noexcept = default;

    /**
     * Returns the next data chunk from this input stream.
     *
     * Note that this function is not guaranteed to return the requested number
     * of bytes, even if they are in fact available in the stream.
     *
     * @param ptr[out]                  Pointer to the start of the data chunk.
     * @param len[in]                   Maximal size of the data chunk to be returned, in bytes.
     * @returns                         Size of the returned data chunk, in bytes.
     *                                  Return value of zero signals end of stream.
     */
    template <class T>
    inline size_t Next(T** ptr, size_t len) {
        Y_ASSERT(ptr);

        return DoNext((const void**)ptr, len);
    }

    template <class T>
    inline size_t Next(T** ptr) {
        return Next(ptr, Max<size_t>());
    }

protected:
    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;
    ui64 DoReadAll(IOutputStream& out) override;
    virtual size_t DoNext(const void** ptr, size_t len) = 0;
};

/**
 * Input stream with direct access to the input buffer and ability to undo read
 *
 * Derived classes must implement `DoUndo` method.
 */
class IZeroCopyInputFastReadTo: public IZeroCopyInput {
public:
    IZeroCopyInputFastReadTo() noexcept = default;
    ~IZeroCopyInputFastReadTo() override;

    IZeroCopyInputFastReadTo(IZeroCopyInputFastReadTo&&) noexcept = default;
    IZeroCopyInputFastReadTo& operator=(IZeroCopyInputFastReadTo&&) noexcept = default;

protected:
    size_t DoReadTo(TString& st, char ch) override;

private:
    /**
     * Undo read.
     *
     * Note that this function not check if you try undo more that read. In fact Undo used for undo read in last chunk.
     *
     * @param len[in]                   Bytes to undo.
     */
    inline void Undo(size_t len) {
        if (len) {
            DoUndo(len);
        }
    }
    virtual void DoUndo(size_t len) = 0;
};

/** @} */
