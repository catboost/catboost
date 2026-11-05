#pragma once

#include <util/system/types.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! MSB-first bit writer over a caller-owned buffer.
/*!
 *  Bits are flushed to the buffer in 4-byte chunks, so the buffer must have room
 *  for up to 3 bytes beyond the last logically-written byte before #Finish is
 *  called.
 */
class TBitWriter
{
public:
    explicit TBitWriter(char* ptr);

    //! Appends the #width low bits of #value. Requires 0 <= #width <= 32 and
    //! #value < 2^#width (for #width == 32 any value is accepted).
    void WriteBits(ui32 value, int width);

    //! Pads the last partial byte with zero low bits and returns the
    //! one-past-end pointer.
    char* Finish();

private:
    char* Ptr_;
    ui64 Accumulator_ = 0;
    int BitCount_ = 0;
};

//! MSB-first bit reader.
/*!
 *  Reads bits written by #TBitWriter. Assumes up to 8 bytes past the logical end
 *  of the stream are safe to read.
 */
class TBitReader
{
public:
    explicit TBitReader(const char* ptr);

    //! Reads and returns #width bits (0 <= #width <= 32).
    ui32 ReadBits(int width);

    //! Returns the one-past-end pointer; the (< 8) buffered sub-byte bits, which
    //! are the writer's zero padding, are dropped.
    const char* Finish();

private:
    const ui8* Ptr_;
    ui64 Accumulator_ = 0;
    int BitCount_ = 0;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define BIT_IO_INL_H_
#include "bit_io-inl.h"
#undef BIT_IO_INL_H_
