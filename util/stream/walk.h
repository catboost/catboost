#pragma once

#include "zerocopy.h"

/**
 * Zero-copy stream that simplifies implementation of derived classes.
 *
 * Derived classes must implement `DoUnboundedNext` method.
 */
class IWalkInput: public IZeroCopyInputFastReadTo {
public:
    IWalkInput()
        : Buf_(nullptr)
        , Len_(0)
    {
    }

protected:
    void DoUndo(size_t len) override;
    size_t DoNext(const void** ptr, size_t len) override;

    /**
     * Returns the next data chunk from this input stream. There are no
     * restrictions on the size of the data chunk.
     *
     * @param ptr[out]                  Pointer to the start of the data chunk.
     * @returns                         Size of the returned data chunk, in bytes.
     *                                  Return value of zero signals end of stream.
     */
    virtual size_t DoUnboundedNext(const void** ptr) = 0;

private:
    const void* Buf_;
    size_t Len_;
};
