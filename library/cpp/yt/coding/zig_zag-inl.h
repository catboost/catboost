#ifndef ZIG_ZAG_INL_H_
#error "Direct inclusion of this file is not allowed, include zig_zag.h"
// For the sake of sane code completion.
#include "zig_zag.h"
#endif
#undef ZIG_ZAG_INL_H_

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline ui32 ZigZagEncode32(i32 n)
{
    // Note:  the right-shift must be arithmetic.
    // Note:  left shift must be unsigned because of overflow.
    return (static_cast<ui32>(n) << 1) ^ static_cast<ui32>(n >> 31);
}

inline i32 ZigZagDecode32(ui32 n)
{
    // Note:  using unsigned types prevent undefined behavior.
    return static_cast<i32>((n >> 1) ^ (~(n & 1) + 1));
}

inline ui64 ZigZagEncode64(i64 n)
{
    // Note:  the right-shift must be arithmetic.
    // Note:  left shift must be unsigned because of overflow.
    return (static_cast<ui64>(n) << 1) ^ static_cast<ui64>(n >> 63);
}

inline i64 ZigZagDecode64(ui64 n)
{
    // Note:  using unsigned types prevent undefined behavior.
    return static_cast<i64>((n >> 1) ^ (~(n & 1) + 1));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
