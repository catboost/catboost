#ifndef GUID_INL_H_
#error "Direct inclusion of this file is not allowed, include guid.h"
// For the sake of sane code completion.
#include "guid.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE constexpr TGuid::TGuid()
    : Parts32{}
{ }

Y_FORCE_INLINE constexpr TGuid::TGuid(ui32 part0, ui32 part1, ui32 part2, ui32 part3)
    : Parts32{part0, part1, part2, part3}
{ }

Y_FORCE_INLINE constexpr TGuid::TGuid(ui64 part0, ui64 part1)
    : Parts64{part0, part1}
{ }

Y_FORCE_INLINE bool TGuid::IsEmpty() const
{
    return Parts64[0] == 0 && Parts64[1] == 0;
}

Y_FORCE_INLINE TGuid::operator bool() const
{
    return !IsEmpty();
}

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE bool operator == (TGuid lhs, TGuid rhs) noexcept
{
    return
        lhs.Parts64[0] == rhs.Parts64[0] &&
        lhs.Parts64[1] == rhs.Parts64[1];
}

Y_FORCE_INLINE std::strong_ordering operator <=> (TGuid lhs, TGuid rhs) noexcept
{
#ifdef __GNUC__
    ui64 lhs0 = __builtin_bswap64(lhs.Parts64[0]);
    ui64 rhs0 = __builtin_bswap64(rhs.Parts64[0]);
    if (lhs0 < rhs0) {
        return std::strong_ordering::less;
    }
    if (lhs0 > rhs0) {
        return std::strong_ordering::greater;
    }
    ui64 lhs1 = __builtin_bswap64(lhs.Parts64[1]);
    ui64 rhs1 = __builtin_bswap64(rhs.Parts64[1]);
    if (lhs1 < rhs1) {
        return std::strong_ordering::less;
    }
    if (lhs1 > rhs1) {
        return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
#else
    int cmp = memcmp(&lhs, &rhs, sizeof(TGuid));
    if (cmp < 0) {
        return std::strong_ordering::less;
    }
    if (cmp > 0) {
        return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
#endif
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

Y_FORCE_INLINE size_t THash<NYT::TGuid>::operator()(const NYT::TGuid& guid) const
{
    const size_t p = 1000000009; // prime number
    return guid.Parts32[0] +
           guid.Parts32[1] * p +
           guid.Parts32[2] * p * p +
           guid.Parts32[3] * p * p * p;
}
