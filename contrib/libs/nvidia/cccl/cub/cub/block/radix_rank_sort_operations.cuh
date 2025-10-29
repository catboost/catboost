/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * radix_rank_sort_operations.cuh contains common abstractions, definitions and
 * operations used for radix sorting and ranking.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/integer_sequence.h>

#include <cuda/bit>
#include <cuda/functional>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/type_traits>

CUB_NAMESPACE_BEGIN

/** \brief Base struct for digit extractor. Contains common code to provide
    special handling for floating-point -0.0.

    \note This handles correctly both the case when the keys are
    bitwise-complemented after twiddling for descending sort (in onesweep) as
    well as when the keys are not bit-negated, but the implementation handles
    descending sort separately (in other implementations in CUB). Twiddling
    alone maps -0.0f to 0x7fffffff and +0.0f to 0x80000000 for float, which are
    subsequent bit patterns and bitwise complements of each other. For onesweep,
    both -0.0f and +0.0f are mapped to the bit pattern of +0.0f (0x80000000) for
    ascending sort, and to the pattern of -0.0f (0x7fffffff) for descending
    sort. For all other sorting implementations in CUB, both are always mapped
    to +0.0f. Since bit patterns for both -0.0f and +0.0f are next to each other
    and only one of them is used, the sorting works correctly. For double, the
    same applies, but with 64-bit patterns.
*/
template <typename KeyT, bool IsFP = ::cuda::is_floating_point_v<KeyT>>
struct BaseDigitExtractor
{
  using TraitsT      = Traits<KeyT>;
  using UnsignedBits = typename TraitsT::UnsignedBits;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits ProcessFloatMinusZero(UnsignedBits key)
  {
    return key;
  }
};

template <typename KeyT>
struct BaseDigitExtractor<KeyT, true>
{
  using TraitsT      = Traits<KeyT>;
  using UnsignedBits = typename TraitsT::UnsignedBits;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits ProcessFloatMinusZero(UnsignedBits key)
  {
    UnsignedBits TWIDDLED_MINUS_ZERO_BITS =
      TraitsT::TwiddleIn(UnsignedBits(1) << UnsignedBits(8 * sizeof(UnsignedBits) - 1));
    UnsignedBits TWIDDLED_ZERO_BITS = TraitsT::TwiddleIn(0);
    return key == TWIDDLED_MINUS_ZERO_BITS ? TWIDDLED_ZERO_BITS : key;
  }
};

/** \brief A wrapper type to extract digits. Uses the BFE intrinsic to extract a
 * key from a digit. */
template <typename KeyT>
struct BFEDigitExtractor : BaseDigitExtractor<KeyT>
{
  using typename BaseDigitExtractor<KeyT>::UnsignedBits;

  ::cuda::std::uint32_t bit_start;
  ::cuda::std::uint32_t num_bits;

  explicit _CCCL_DEVICE _CCCL_FORCEINLINE
  BFEDigitExtractor(::cuda::std::uint32_t bit_start = 0, ::cuda::std::uint32_t num_bits = 0)
      : bit_start(bit_start)
      , num_bits(num_bits)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t Digit(UnsignedBits key) const
  {
    return ::cuda::bitfield_extract(this->ProcessFloatMinusZero(key), bit_start, num_bits);
  }
};

/** \brief A wrapper type to extract digits. Uses a combination of shift and
 * bitwise and to extract digits. */
template <typename KeyT>
struct ShiftDigitExtractor : BaseDigitExtractor<KeyT>
{
  using typename BaseDigitExtractor<KeyT>::UnsignedBits;

  ::cuda::std::uint32_t bit_start;
  ::cuda::std::uint32_t mask;

  explicit _CCCL_DEVICE _CCCL_FORCEINLINE
  ShiftDigitExtractor(::cuda::std::uint32_t bit_start = 0, ::cuda::std::uint32_t num_bits = 0)
      : bit_start(bit_start)
      , mask((1 << num_bits) - 1)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t Digit(UnsignedBits key) const
  {
    return ::cuda::std::uint32_t(this->ProcessFloatMinusZero(key) >> UnsignedBits(bit_start)) & mask;
  }
};

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{

template <bool... Bs>
struct logic_helper_t;

template <bool>
struct true_t
{
  static constexpr bool value = true;
};

template <bool... Bs>
using all_t = //
  ::cuda::std::is_same< //
    logic_helper_t<Bs...>, //
    logic_helper_t<true_t<Bs>::value...>>;

struct identity_decomposer_t
{
  template <class T>
  _CCCL_HOST_DEVICE T& operator()(T& key) const
  {
    return key;
  }
};

template <class F, class... Ts, ::cuda::std::size_t... Is>
_CCCL_HOST_DEVICE void
for_each_member_impl_helper(F f, const ::cuda::std::tuple<Ts&...>& tpl, THRUST_NS_QUALIFIER::index_sequence<Is...>)
{
  [[maybe_unused]] auto sink = {(f(::cuda::std::get<Is>(tpl)), 0)...};
}

template <class F, class... Ts>
_CCCL_HOST_DEVICE void for_each_member_impl(F f, const ::cuda::std::tuple<Ts&...>& tpl)
{
  static_assert(sizeof...(Ts), "Empty aggregates are not supported");

  // Most radix operations are indifferent to the order of operations.
  // Conversely, the digit extractor traverses fields from the least significant
  // to the most significant to imitate bitset printing where higher bits are on
  // the left. It also maps to intuition, where something coming first is more
  // important. Therefore, we traverse fields on the opposite order.
  for_each_member_impl_helper(f, tpl, THRUST_NS_QUALIFIER::make_reversed_index_sequence<sizeof...(Ts)>{});
}

template <class F, class DecomposerT, class T>
_CCCL_HOST_DEVICE void for_each_member(F f, DecomposerT decomposer, T& aggregate)
{
  for_each_member_impl(f, decomposer(aggregate));
}

namespace radix
{
template <class T, class = void>
struct is_fundamental_type
{
  static constexpr bool value = false;
};

template <class T>
struct is_fundamental_type<T, ::cuda::std::void_t<typename Traits<T>::UnsignedBits>>
{
  static constexpr bool value = true;
};

template <class T, class = void>
struct is_tuple_of_references_to_fundamental_types_t : ::cuda::std::false_type
{};

template <class... Ts>
struct is_tuple_of_references_to_fundamental_types_t< //
  ::cuda::std::tuple<Ts&...>, //
  ::cuda::std::enable_if_t< //
    all_t<is_fundamental_type<Ts>::value...>::value //
    >> //
    : ::cuda::std::true_type
{};

template <class KeyT, class DecomposerT>
using decomposer_check_t =
  is_tuple_of_references_to_fundamental_types_t<_CUDA_VSTD::invoke_result_t<DecomposerT, KeyT&>>;

template <class T>
struct bit_ordered_conversion_policy_t
{
  using bit_ordered_type = typename Traits<T>::UnsignedBits;

  static _CCCL_HOST_DEVICE bit_ordered_type to_bit_ordered(detail::identity_decomposer_t, bit_ordered_type val)
  {
    return Traits<T>::TwiddleIn(val);
  }

  static _CCCL_HOST_DEVICE bit_ordered_type from_bit_ordered(detail::identity_decomposer_t, bit_ordered_type val)
  {
    return Traits<T>::TwiddleOut(val);
  }
};

template <class T>
struct bit_ordered_inversion_policy_t
{
  using bit_ordered_type = typename Traits<T>::UnsignedBits;

  static _CCCL_HOST_DEVICE bit_ordered_type inverse(detail::identity_decomposer_t, bit_ordered_type val)
  {
    return ~val;
  }
};

template <class T, bool = is_fundamental_type<T>::value>
struct traits_t
{
  using bit_ordered_type              = typename Traits<T>::UnsignedBits;
  using bit_ordered_conversion_policy = bit_ordered_conversion_policy_t<T>;
  using bit_ordered_inversion_policy  = bit_ordered_inversion_policy_t<T>;

  template <class FundamentalExtractorT, class /* DecomposerT */>
  using digit_extractor_t = FundamentalExtractorT;

  static _CCCL_HOST_DEVICE bit_ordered_type min_raw_binary_key(detail::identity_decomposer_t)
  {
    return Traits<T>::LOWEST_KEY;
  }

  static _CCCL_HOST_DEVICE bit_ordered_type max_raw_binary_key(detail::identity_decomposer_t)
  {
    return Traits<T>::MAX_KEY;
  }

  static _CCCL_HOST_DEVICE int default_end_bit(detail::identity_decomposer_t)
  {
    return sizeof(T) * 8;
  }

  template <class FundamentalExtractorT>
  static _CCCL_HOST_DEVICE digit_extractor_t<FundamentalExtractorT, detail::identity_decomposer_t>
  digit_extractor(int begin_bit, int num_bits, detail::identity_decomposer_t)
  {
    return FundamentalExtractorT(begin_bit, num_bits);
  }
};

template <class DecomposerT>
struct min_raw_binary_key_f
{
  DecomposerT decomposer;

  template <class T>
  _CCCL_HOST_DEVICE void operator()(T& field)
  {
    using traits                               = traits_t<::cuda::std::remove_cv_t<T>>;
    using bit_ordered_type                     = typename traits::bit_ordered_type;
    reinterpret_cast<bit_ordered_type&>(field) = traits::min_raw_binary_key(detail::identity_decomposer_t{});
  }
};

template <class DecomposerT, class T>
_CCCL_HOST_DEVICE void min_raw_binary_key(DecomposerT decomposer, T& aggregate)
{
  detail::for_each_member(min_raw_binary_key_f<DecomposerT>{decomposer}, decomposer, aggregate);
}

template <class DecomposerT>
struct max_raw_binary_key_f
{
  DecomposerT decomposer;

  template <class T>
  _CCCL_HOST_DEVICE void operator()(T& field)
  {
    using traits                               = traits_t<::cuda::std::remove_cv_t<T>>;
    using bit_ordered_type                     = typename traits::bit_ordered_type;
    reinterpret_cast<bit_ordered_type&>(field) = traits::max_raw_binary_key(detail::identity_decomposer_t{});
  }
};

template <class DecomposerT, class T>
_CCCL_HOST_DEVICE void max_raw_binary_key(DecomposerT decomposer, T& aggregate)
{
  detail::for_each_member(max_raw_binary_key_f<DecomposerT>{decomposer}, decomposer, aggregate);
}

template <class DecomposerT>
struct to_bit_ordered_f
{
  DecomposerT decomposer;

  template <class T>
  _CCCL_HOST_DEVICE void operator()(T& field)
  {
    using traits                 = traits_t<::cuda::std::remove_cv_t<T>>;
    using bit_ordered_type       = typename traits::bit_ordered_type;
    using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

    auto& ordered_field = reinterpret_cast<bit_ordered_type&>(field);
    ordered_field       = bit_ordered_conversion::to_bit_ordered(detail::identity_decomposer_t{}, ordered_field);
  }
};

template <class DecomposerT, class T>
_CCCL_HOST_DEVICE void to_bit_ordered(DecomposerT decomposer, T& aggregate)
{
  detail::for_each_member(to_bit_ordered_f<DecomposerT>{decomposer}, decomposer, aggregate);
}

template <class DecomposerT>
struct from_bit_ordered_f
{
  DecomposerT decomposer;

  template <class T>
  _CCCL_HOST_DEVICE void operator()(T& field)
  {
    using traits                 = traits_t<::cuda::std::remove_cv_t<T>>;
    using bit_ordered_type       = typename traits::bit_ordered_type;
    using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

    auto& ordered_field = reinterpret_cast<bit_ordered_type&>(field);
    ordered_field       = bit_ordered_conversion::from_bit_ordered(detail::identity_decomposer_t{}, ordered_field);
  }
};

template <class DecomposerT, class T>
_CCCL_HOST_DEVICE void from_bit_ordered(DecomposerT decomposer, T& aggregate)
{
  detail::for_each_member(from_bit_ordered_f<DecomposerT>{decomposer}, decomposer, aggregate);
}

template <class DecomposerT>
struct inverse_f
{
  DecomposerT decomposer;

  template <class T>
  _CCCL_HOST_DEVICE void operator()(T& field)
  {
    using traits           = traits_t<::cuda::std::remove_cv_t<T>>;
    using bit_ordered_type = typename traits::bit_ordered_type;

    auto& ordered_field = reinterpret_cast<bit_ordered_type&>(field);
    ordered_field       = ~ordered_field;
  }
};

template <class DecomposerT, class T>
_CCCL_HOST_DEVICE void inverse(DecomposerT decomposer, T& aggregate)
{
  detail::for_each_member(inverse_f<DecomposerT>{decomposer}, decomposer, aggregate);
}

template <class DecomposerT>
struct default_end_bit_f
{
  int& result;
  DecomposerT decomposer;

  template <class T>
  _CCCL_HOST_DEVICE void operator()(T& field)
  {
    result += sizeof(field) * 8;
  }
};

template <class DecomposerT, class T>
_CCCL_HOST_DEVICE int default_end_bit(DecomposerT decomposer, T& aggregate)
{
  int result{};
  detail::for_each_member(default_end_bit_f<DecomposerT>{result, decomposer}, decomposer, aggregate);
  return result;
}

struct digit_f
{
  ::cuda::std::uint32_t& dst;
  ::cuda::std::uint32_t& dst_bit_start;
  ::cuda::std::uint32_t& src_bit_start;
  ::cuda::std::uint32_t& num_bits;

  template <class T>
  _CCCL_HOST_DEVICE void operator()(T& src)
  {
    constexpr ::cuda::std::uint32_t src_size = sizeof(T) * 8;

    if (src_bit_start >= src_size)
    {
      src_bit_start -= src_size;
    }
    else
    {
      using traits           = traits_t<::cuda::std::remove_cv_t<T>>;
      using bit_ordered_type = typename traits::bit_ordered_type;

      const ::cuda::std::uint32_t bits_to_copy = (::cuda::std::min) (src_size - src_bit_start, num_bits);

      if (bits_to_copy)
      {
        bit_ordered_type ordered_src =
          BaseDigitExtractor<T>::ProcessFloatMinusZero(reinterpret_cast<bit_ordered_type&>(src));

        const ::cuda::std::uint32_t mask = (1 << bits_to_copy) - 1;
        dst                              = dst | (((ordered_src >> src_bit_start) & mask) << dst_bit_start);

        num_bits -= bits_to_copy;
        dst_bit_start += bits_to_copy;
      }
      src_bit_start = 0;
    }
  }
};

template <class DecomposerT, class T>
_CCCL_HOST_DEVICE void
digit(DecomposerT decomposer,
      ::cuda::std::uint32_t& dst,
      T& src,
      ::cuda::std::uint32_t& dst_bit_start,
      ::cuda::std::uint32_t& src_bit_start,
      ::cuda::std::uint32_t& num_bits)
{
  detail::for_each_member(digit_f{dst, dst_bit_start, src_bit_start, num_bits}, decomposer, src);
}

template <class DecomposerT>
struct custom_digit_extractor_t
{
  DecomposerT decomposer;
  ::cuda::std::uint32_t bit_start;
  ::cuda::std::uint32_t num_bits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  custom_digit_extractor_t(DecomposerT decomposer, ::cuda::std::uint32_t bit_start, ::cuda::std::uint32_t num_bits)
      : decomposer(decomposer)
      , bit_start(bit_start)
      , num_bits(num_bits)
  {}

  template <class T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t Digit(T& key) const
  {
    ::cuda::std::uint32_t result{};
    ::cuda::std::uint32_t dst_bit_start{};
    ::cuda::std::uint32_t src_bit_start = bit_start;
    ::cuda::std::uint32_t bits_remaining{num_bits};
    digit(decomposer, result, key, dst_bit_start, src_bit_start, bits_remaining);
    return result;
  }
};

struct custom_bit_conversion_policy_t
{
  template <class DecomposerT, class T>
  static _CCCL_HOST_DEVICE T to_bit_ordered(DecomposerT decomposer, T val)
  {
    detail::radix::to_bit_ordered(decomposer, val);
    return val;
  }

  template <class DecomposerT, class T>
  static _CCCL_HOST_DEVICE T from_bit_ordered(DecomposerT decomposer, T val)
  {
    detail::radix::from_bit_ordered(decomposer, val);
    return val;
  }
};

struct custom_bit_inversion_policy_t
{
  template <class DecomposerT, class T>
  static _CCCL_HOST_DEVICE T inverse(DecomposerT decomposer, T val)
  {
    detail::radix::inverse(decomposer, val);
    return val;
  }
};

template <class T>
struct traits_t<T, false /* is_fundamental */>
{
  using bit_ordered_type              = T;
  using bit_ordered_conversion_policy = custom_bit_conversion_policy_t;
  using bit_ordered_inversion_policy  = custom_bit_inversion_policy_t;

  template <class FundamentalExtractorT, class DecomposerT>
  using digit_extractor_t = custom_digit_extractor_t<DecomposerT>;

  template <class DecomposerT>
  static _CCCL_HOST_DEVICE bit_ordered_type min_raw_binary_key(DecomposerT decomposer)
  {
    T val{};
    detail::radix::min_raw_binary_key(decomposer, val);
    return val;
  }

  template <class DecomposerT>
  static _CCCL_HOST_DEVICE bit_ordered_type max_raw_binary_key(DecomposerT decomposer)
  {
    T val{};
    detail::radix::max_raw_binary_key(decomposer, val);
    return val;
  }

  template <class DecomposerT>
  static _CCCL_HOST_DEVICE int default_end_bit(DecomposerT decomposer)
  {
    T aggregate{};
    return detail::radix::default_end_bit(decomposer, aggregate);
  }

  template <class FundamentalExtractorT, class DecomposerT>
  static _CCCL_HOST_DEVICE digit_extractor_t<FundamentalExtractorT, DecomposerT>
  digit_extractor(int begin_bit, int num_bits, DecomposerT decomposer)
  {
    return custom_digit_extractor_t<DecomposerT>(decomposer, begin_bit, num_bits);
  }
};

} // namespace radix

} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED

//! Twiddling keys for radix sort
template <bool IS_DESCENDING, typename KeyT>
struct RadixSortTwiddle
{
private:
  using traits                        = detail::radix::traits_t<KeyT>;
  using bit_ordered_type              = typename traits::bit_ordered_type;
  using bit_ordered_conversion_policy = typename traits::bit_ordered_conversion_policy;
  using bit_ordered_inversion_policy  = typename traits::bit_ordered_inversion_policy;

public:
  template <class DecomposerT = detail::identity_decomposer_t>
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE //
  bit_ordered_type
  In(bit_ordered_type key, DecomposerT decomposer = {})
  {
    key = bit_ordered_conversion_policy::to_bit_ordered(decomposer, key);
    if constexpr (IS_DESCENDING)
    {
      key = bit_ordered_inversion_policy::inverse(decomposer, key);
    }
    return key;
  }

  template <class DecomposerT = detail::identity_decomposer_t>
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE //
  bit_ordered_type
  Out(bit_ordered_type key, DecomposerT decomposer = {})
  {
    if constexpr (IS_DESCENDING)
    {
      key = bit_ordered_inversion_policy::inverse(decomposer, key);
    }
    key = bit_ordered_conversion_policy::from_bit_ordered(decomposer, key);
    return key;
  }

  template <class DecomposerT = detail::identity_decomposer_t>
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE //
  bit_ordered_type
  DefaultKey(DecomposerT decomposer = {})
  {
    return IS_DESCENDING ? traits::min_raw_binary_key(decomposer) : traits::max_raw_binary_key(decomposer);
  }
};

CUB_NAMESPACE_END
