#ifndef BIT_IO_INL_H_
#error "Direct inclusion of this file is not allowed, include bit_io.h"
// For the sake of sane code completion.
#include "bit_io.h"
#endif

#include <util/system/compiler.h>
#include <util/system/unaligned_mem.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline TBitWriter::TBitWriter(char* ptr)
    : Ptr_(ptr)
{ }

Y_FORCE_INLINE void TBitWriter::WriteBits(ui32 value, int width)
{
    // Flush a whole 32-bit word at once (a single unaligned store) instead of
    // looping over individual bytes with a data-dependent trip count.
    Accumulator_ = (Accumulator_ << width) | value;
    BitCount_ += width;
    if (BitCount_ >= 32) {
        BitCount_ -= 32;
        ui32 word = __builtin_bswap32(static_cast<ui32>(Accumulator_ >> BitCount_));
        WriteUnaligned<ui32>(Ptr_, word);
        Ptr_ += sizeof(word);
    }
}

inline char* TBitWriter::Finish()
{
    while (BitCount_ >= 8) {
        BitCount_ -= 8;
        *Ptr_++ = static_cast<char>((Accumulator_ >> BitCount_) & 0xff);
    }
    if (BitCount_ > 0) {
        *Ptr_++ = static_cast<char>((Accumulator_ << (8 - BitCount_)) & 0xff);
        BitCount_ = 0;
    }
    Accumulator_ = 0;
    return Ptr_;
}

////////////////////////////////////////////////////////////////////////////////

inline TBitReader::TBitReader(const char* ptr)
    : Ptr_(reinterpret_cast<const ui8*>(ptr))
{ }

Y_FORCE_INLINE ui32 TBitReader::ReadBits(int width)
{
    while (BitCount_ < width) {
        Accumulator_ = (Accumulator_ << 8) | *Ptr_++;
        BitCount_ += 8;
    }
    BitCount_ -= width;
    return static_cast<ui32>((Accumulator_ >> BitCount_) & ((1ull << width) - 1));
}

inline const char* TBitReader::Finish()
{
    return reinterpret_cast<const char*>(Ptr_);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
