/*
 *  Copyright 2024 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// Portions of this code are derived from
//
// Manjunath Kudlur's Carbon library
//
// and
//
// Based on Boost.Phoenix v1.2
// Copyright (c) 2001-2002 Joel de Guzman

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/functional.h>
#include <thrust/tuple.h>

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{
// there's no standard plus_equal functional, so roll an ad hoc one here
struct plus_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) += THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) += THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) += THRUST_FWD(t2);
  }
};

// there's no standard minus_equal functional, so roll an ad hoc one here
struct minus_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) -= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) -= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) -= THRUST_FWD(t2);
  }
};

// there's no standard multiplies_equal functional, so roll an ad hoc one here
struct multiplies_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) *= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) *= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) *= THRUST_FWD(t2);
  }
};

// there's no standard divides_equal functional, so roll an ad hoc one here
struct divides_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) /= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) /= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) /= THRUST_FWD(t2);
  }
};

// there's no standard modulus_equal functional, so roll an ad hoc one here
struct modulus_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) %= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) %= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) %= THRUST_FWD(t2);
  }
};

// there's no standard bit_and_equal functional, so roll an ad hoc one here
struct bit_and_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) &= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) &= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) &= THRUST_FWD(t2);
  }
};

// there's no standard bit_or_equal functional, so roll an ad hoc one here
struct bit_or_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) |= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) |= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) |= THRUST_FWD(t2);
  }
};

// there's no standard bit_xor_equal functional, so roll an ad hoc one here
struct bit_xor_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) ^= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) ^= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) ^= THRUST_FWD(t2);
  }
};

// there's no standard bit_lshift_equal functional, so roll an ad hoc one here
struct bit_lshift_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) <<= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) <<= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) <<= THRUST_FWD(t2);
  }
};

// there's no standard bit_rshift_equal functional, so roll an ad hoc one here
struct bit_rshift_equal
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) >>= THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) >>= THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) >>= THRUST_FWD(t2);
  }
};

// there's no standard bit_lshift functional, so roll an ad hoc one here
struct bit_lshift
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) << THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) << THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) << THRUST_FWD(t2);
  }
};

// there's no standard bit_rshift functional, so roll an ad hoc one here
struct bit_rshift
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1& t1, T2&& t2) const
    noexcept(noexcept(THRUST_FWD(t1) >> THRUST_FWD(t2))) -> decltype(THRUST_FWD(t1) >> THRUST_FWD(t2))
  {
    return THRUST_FWD(t1) >> THRUST_FWD(t2);
  }
};

#define MAKE_BINARY_COMPOSITE(op, functor)                                                                       \
  template <typename A, typename B, ::cuda::std::enable_if_t<is_actor<A>::value || is_actor<B>::value, int> = 0> \
  _CCCL_HOST_DEVICE auto operator op(const A& a, const B& b)->decltype(compose(functor{}, a, b))                 \
  {                                                                                                              \
    return compose(functor{}, a, b);                                                                             \
  }

MAKE_BINARY_COMPOSITE(==, ::cuda::std::equal_to<>)
MAKE_BINARY_COMPOSITE(!=, ::cuda::std::not_equal_to<>)
MAKE_BINARY_COMPOSITE(<, ::cuda::std::less<>)
MAKE_BINARY_COMPOSITE(<=, ::cuda::std::less_equal<>)
MAKE_BINARY_COMPOSITE(>, ::cuda::std::greater<>)
MAKE_BINARY_COMPOSITE(>=, ::cuda::std::greater_equal<>)

MAKE_BINARY_COMPOSITE(+, ::cuda::std::plus<>)
MAKE_BINARY_COMPOSITE(-, ::cuda::std::minus<>)
MAKE_BINARY_COMPOSITE(*, ::cuda::std::multiplies<>)
MAKE_BINARY_COMPOSITE(/, ::cuda::std::divides<>)
MAKE_BINARY_COMPOSITE(%, ::cuda::std::modulus<>)

MAKE_BINARY_COMPOSITE(+=, plus_equal)
MAKE_BINARY_COMPOSITE(-=, minus_equal)
MAKE_BINARY_COMPOSITE(*=, multiplies_equal)
MAKE_BINARY_COMPOSITE(/=, divides_equal)
MAKE_BINARY_COMPOSITE(%=, modulus_equal)

MAKE_BINARY_COMPOSITE(&&, ::cuda::std::logical_and<>)
MAKE_BINARY_COMPOSITE(||, ::cuda::std::logical_or<>)

MAKE_BINARY_COMPOSITE(&, ::cuda::std::bit_and<>)
MAKE_BINARY_COMPOSITE(|, ::cuda::std::bit_or<>)
MAKE_BINARY_COMPOSITE(^, ::cuda::std::bit_xor<>)
MAKE_BINARY_COMPOSITE(<<, bit_lshift)
MAKE_BINARY_COMPOSITE(>>, bit_rshift)

MAKE_BINARY_COMPOSITE(&=, bit_and_equal)
MAKE_BINARY_COMPOSITE(|=, bit_or_equal)
MAKE_BINARY_COMPOSITE(^=, bit_xor_equal)
MAKE_BINARY_COMPOSITE(<<=, bit_lshift_equal)
MAKE_BINARY_COMPOSITE(>>=, bit_rshift_equal)

#undef MAKE_BINARY_COMPOSITE

// there's no standard unary_plus functional, so roll an ad hoc one here
struct unary_plus
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1) const noexcept(noexcept(+THRUST_FWD(t1)))
    -> decltype(+THRUST_FWD(t1))
  {
    return +THRUST_FWD(t1);
  }
};

// there's no standard prefix_increment functional, so roll an ad hoc one here
struct prefix_increment
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1) const noexcept(noexcept(++THRUST_FWD(t1)))
    -> decltype(++THRUST_FWD(t1))
  {
    return ++THRUST_FWD(t1);
  }
}; // end prefix_increment

// there's no standard postfix_increment functional, so roll an ad hoc one here
struct postfix_increment
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1) const noexcept(noexcept(THRUST_FWD(t1)++))
    -> decltype(THRUST_FWD(t1)++)
  {
    return THRUST_FWD(t1)++;
  }
}; // end postfix_increment

// there's no standard prefix_decrement functional, so roll an ad hoc one here
struct prefix_decrement
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1) const noexcept(noexcept(--THRUST_FWD(t1)))
    -> decltype(--THRUST_FWD(t1))
  {
    return --THRUST_FWD(t1);
  }
}; // end prefix_decrement

// there's no standard postfix_decrement functional, so roll an ad hoc one here
struct postfix_decrement
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1) const noexcept(noexcept(THRUST_FWD(t1)--))
    -> decltype(THRUST_FWD(t1)--)
  {
    return THRUST_FWD(t1)--;
  }
}; // end prefix_increment

// there's no standard bit_not functional, so roll an ad hoc one here
struct bit_not
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1) const noexcept(noexcept(~THRUST_FWD(t1)))
    -> decltype(~THRUST_FWD(t1))
  {
    return ~THRUST_FWD(t1);
  }
}; // end prefix_increment

#define MAKE_UNARY_COMPOSITE(op, functor)                                         \
  template <typename A, ::cuda::std::enable_if_t<is_actor<A>::value, int> = 0>    \
  _CCCL_HOST_DEVICE auto operator op(const A& a)->decltype(compose(functor{}, a)) \
  {                                                                               \
    return compose(functor{}, a);                                                 \
  }

MAKE_UNARY_COMPOSITE(+, unary_plus)
MAKE_UNARY_COMPOSITE(-, ::cuda::std::negate<>)
MAKE_UNARY_COMPOSITE(++, prefix_increment)
MAKE_UNARY_COMPOSITE(--, prefix_decrement)
MAKE_UNARY_COMPOSITE(!, ::cuda::std::logical_not<>)
MAKE_UNARY_COMPOSITE(~, bit_not)

#undef MAKE_UNARY_COMPOSITE

#define MAKE_UNARY_COMPOSITE_POSTFIX(op, functor)                                      \
  template <typename A, ::cuda::std::enable_if_t<is_actor<A>::value, int> = 0>         \
  _CCCL_HOST_DEVICE auto operator op(const A& a, int)->decltype(compose(functor{}, a)) \
  {                                                                                    \
    return compose(functor{}, a);                                                      \
  }

MAKE_UNARY_COMPOSITE_POSTFIX(++, postfix_increment)
MAKE_UNARY_COMPOSITE_POSTFIX(--, postfix_decrement)

#undef MAKE_UNARY_COMPOSITE_POSTFIX

// there's no standard assign functional, so roll an ad hoc one here
struct assign
{
  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const
    THRUST_DECLTYPE_RETURNS(THRUST_FWD(t1) = THRUST_FWD(t2))
};

template <typename Eval, typename T>
_CCCL_HOST_DEVICE auto do_assign(const actor<Eval>& _1, const T& _2) -> decltype(compose(assign{}, _1, _2))
{
  return compose(assign{}, _1, _2);
}
} // namespace functional
} // namespace detail
THRUST_NAMESPACE_END
