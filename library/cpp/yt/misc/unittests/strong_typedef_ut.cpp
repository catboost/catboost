#include <library/cpp/yt/misc/strong_typedef.h>

#include <limits>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

YT_DEFINE_STRONG_TYPEDEF(TMyInt1, int);
YT_DEFINE_STRONG_TYPEDEF(TMyInt2, TMyInt1);

static_assert(std::numeric_limits<TMyInt1>::is_specialized);
static_assert(std::numeric_limits<TMyInt2>::is_specialized);

#define XX(name) \
    static_assert(std::numeric_limits<TMyInt1>::name == std::numeric_limits<int>::name); \
    static_assert(std::numeric_limits<TMyInt2>::name == std::numeric_limits<int>::name);

XX(is_signed)
XX(digits)
XX(digits10)
XX(max_digits10)
XX(is_integer)
XX(is_exact)
XX(radix)
XX(min_exponent)
XX(min_exponent10)
XX(max_exponent)
XX(max_exponent10)
XX(has_infinity)
XX(has_quiet_NaN)
XX(has_signaling_NaN)
XX(has_denorm)
XX(has_denorm_loss)
XX(is_iec559)
XX(is_bounded)
XX(is_modulo)
XX(traps)
XX(tinyness_before)
XX(round_style)

#undef XX

#define XX(name) \
    static_assert(std::numeric_limits<TMyInt1>::name() == TMyInt1(std::numeric_limits<int>::name())); \
    static_assert(std::numeric_limits<TMyInt2>::name() == TMyInt2(TMyInt1(std::numeric_limits<int>::name())));

XX(min)
XX(max)
XX(lowest)
XX(epsilon)
XX(round_error)
XX(infinity)
XX(quiet_NaN)
XX(signaling_NaN)
XX(denorm_min)

#undef XX

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
