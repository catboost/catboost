/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh> // implicit_prom_t
#include <cub/util_type.cuh> // _CCCL_HAS_INT128()

#include <cuda/cmath> // cuda::std::ceil_div
#include <cuda/std/bit> // cuda::std::has_single_bit
#include <cuda/std/climits> // CHAR_BIT
#include <cuda/std/cstdint> // uint64_t
#include <cuda/std/limits> // numeric_limits
#include <cuda/std/type_traits> // ::cuda::std::is_integral

#if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS

CUB_NAMESPACE_BEGIN

namespace detail
{

/***********************************************************************************************************************
 * larger_unsigned_type
 **********************************************************************************************************************/

template <typename T, typename = void>
struct larger_unsigned_type
{
  using type = void;
};

template <typename T>
struct larger_unsigned_type<T, ::cuda::std::enable_if_t<(sizeof(T) < 4)>>
{
  using type = ::cuda::std::uint32_t;
};

template <typename T>
struct larger_unsigned_type<T, ::cuda::std::enable_if_t<(sizeof(T) == 4)>>
{
  using type = ::cuda::std::uint64_t;
};

#if _CCCL_HAS_INT128()

template <typename T>
struct larger_unsigned_type<T, ::cuda::std::enable_if_t<(sizeof(T) == 8)>>
{
  using type = __uint128_t;
};

#endif // _CCCL_HAS_INT128()

template <typename T>
using larger_unsigned_type_t = typename larger_unsigned_type<T>::type;

template <typename T>
using unsigned_implicit_prom_t = ::cuda::std::make_unsigned_t<implicit_prom_t<T>>;

template <typename T>
using supported_integral =
  ::cuda::std::bool_constant<::cuda::std::is_integral_v<T> && !::cuda::std::is_same_v<T, bool> && (sizeof(T) <= 8)>;

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename DivisorType, typename T, typename R>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE unsigned_implicit_prom_t<DivisorType>
multiply_extract_higher_bits(T value, R multiplier)
{
  static_assert(supported_integral<T>::value, "unsupported type");
  static_assert(supported_integral<R>::value, "unsupported type");
  if constexpr (_CCCL_TRAIT(::cuda::std::is_signed, T))
  {
    _CCCL_ASSERT(value >= 0, "value must be non-negative");
  }
  if constexpr (_CCCL_TRAIT(::cuda::std::is_signed, R))
  {
    _CCCL_ASSERT(multiplier >= 0, "multiplier must be non-negative");
  }
  static constexpr int NumBits = sizeof(DivisorType) * CHAR_BIT;
  using unsigned_t             = unsigned_implicit_prom_t<DivisorType>;
  using larger_t               = larger_unsigned_type_t<DivisorType>;
  // clang-format off
  NV_IF_TARGET(
    NV_IS_HOST,
      (return static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> NumBits);),
    //NV_IS_DEVICE
      (return (sizeof(T) == 8)
        ? static_cast<unsigned_t>(__umul64hi(value, multiplier))
        : static_cast<unsigned_t>((static_cast<larger_t>(value) * multiplier) >> NumBits);));
  // clang-format on
}

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4127) /* conditional expression is constant */

template <typename T1>
class fast_div_mod
{
  static_assert(supported_integral<T1>::value, "unsupported type");

  // uint16_t is a special case that would requires complex logic. Workaround: convert to int
  using T          = ::cuda::std::conditional_t<::cuda::std::is_same_v<T1, ::cuda::std::uint16_t>, int, T1>;
  using unsigned_t = unsigned_implicit_prom_t<T>;

public:
  template <typename R>
  struct result
  {
    using common_t = decltype(R{} / T{});
    common_t quotient;
    common_t remainder;
  };

  fast_div_mod() = delete;

  [[nodiscard]] _CCCL_HOST_DEVICE explicit fast_div_mod(T divisor) noexcept
      : _divisor{static_cast<unsigned_t>(divisor)}
  {
    using larger_t = larger_unsigned_type_t<T>;
    _CCCL_ASSERT(divisor > 0, "divisor must be positive");
    auto udivisor = static_cast<unsigned_t>(divisor);
    // the following branches are needed to avoid negative shift
    if (::cuda::std::has_single_bit(udivisor)) // power of two
    {
      _shift_right = ::cuda::std::bit_width(udivisor) - 1;
      return;
    }
    else if (sizeof(T) == 8 && divisor == 3)
    {
      return;
    }
    constexpr int BitSize   = sizeof(T) * CHAR_BIT; // 32
    constexpr int BitOffset = BitSize / 16; // 2
    int num_bits            = ::cuda::std::bit_width(udivisor) + 1;
    _CCCL_ASSERT(static_cast<size_t>(num_bits + BitSize - BitOffset) < sizeof(larger_t) * CHAR_BIT, "overflow error");
    // without explicit power-of-two check, num_bits needs to replace +1 with !::cuda::std::has_single_bit(udivisor)
    _multiplier  = static_cast<unsigned_t>(::cuda::ceil_div(larger_t{1} << (num_bits + BitSize - BitOffset), //
                                                           static_cast<larger_t>(divisor)));
    _shift_right = num_bits - BitOffset;
    _CCCL_ASSERT(_multiplier != 0, "overflow error");
  }

  fast_div_mod(const fast_div_mod&) noexcept = default;

  fast_div_mod(fast_div_mod&&) noexcept = default;

  template <typename R>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result<R> operator()(R dividend) const noexcept
  {
    static_assert(supported_integral<R>::value, "unsupported type");
    using common_t  = decltype(R{} / T{});
    using ucommon_t = ::cuda::std::make_unsigned_t<common_t>;
    using result_t  = result<R>;
    _CCCL_ASSERT(dividend >= 0, "divisor must be non-negative");
    auto udividend = static_cast<ucommon_t>(dividend);
    if (_divisor == 1)
    {
      return result_t{static_cast<common_t>(dividend), common_t{}};
    }
    else if (_divisor > unsigned_t{::cuda::std::numeric_limits<T>::max() / 2})
    {
      auto quotient = udividend >= static_cast<ucommon_t>(_divisor);
      return result_t{static_cast<common_t>(quotient), static_cast<common_t>(udividend - (quotient * _divisor))};
    }
    else if (sizeof(T) == 8 && _divisor == 3)
    {
      return result_t{static_cast<common_t>(udividend / 3), static_cast<common_t>(udividend % 3)};
    }
    auto higher_bits = (_multiplier == 0) ? udividend : multiply_extract_higher_bits<T>(dividend, _multiplier);
    auto quotient    = higher_bits >> _shift_right;
    auto remainder   = udividend - (quotient * _divisor);
    _CCCL_ASSERT(quotient == udividend / _divisor, "wrong quotient");
    _CCCL_ASSERT(remainder < (ucommon_t) _divisor, "remainder out of range");
    return result_t{static_cast<common_t>(quotient), static_cast<common_t>(remainder)};
  }

  template <typename R>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend implicit_prom_t<T> operator/(R dividend, fast_div_mod div) noexcept
  {
    return div(dividend).quotient;
  }

  template <typename R>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend implicit_prom_t<T> operator%(R dividend, fast_div_mod div) noexcept
  {
    return div(dividend).remainder;
  }

private:
  unsigned_t _divisor    = 1;
  unsigned_t _multiplier = 0;
  unsigned _shift_right  = 0;
};
_CCCL_DIAG_POP

} // namespace detail

CUB_NAMESPACE_END

#if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
_CCCL_END_NV_DIAG_SUPPRESS()
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS
