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
#include <thrust/detail/type_deduction.h>
#include <thrust/tuple.h>

#include <cuda/std/type_traits>
#include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{
// An actor is a node in an expression template
template <typename Eval>
struct actor : Eval
{
  constexpr actor() = default;

  _CCCL_HOST_DEVICE actor(const Eval& base)
      : Eval(base)
  {}

  template <typename... Ts>
  _CCCL_HOST_DEVICE auto operator()(Ts&&... ts) const -> decltype(Eval::eval(THRUST_FWD(ts)...))
  {
    return Eval::eval(THRUST_FWD(ts)...);
  }

  template <typename T>
  _CCCL_HOST_DEVICE auto operator=(const T& _1) const -> decltype(do_assign(*this, _1))
  {
    return do_assign(*this, _1);
  }
};

template <typename T>
struct is_actor : ::cuda::std::false_type
{};

template <typename T>
struct is_actor<actor<T>> : ::cuda::std::true_type
{};

// a node selecting and returning one of the arguments to the entire expression template
template <unsigned int Pos>
struct argument
{
  template <typename... Ts>
  _CCCL_HOST_DEVICE auto eval(Ts&&... args) const
    -> decltype(thrust::get<Pos>(thrust::tuple<Ts&&...>{THRUST_FWD(args)...}))
  {
    return thrust::get<Pos>(thrust::tuple<Ts&&...>{THRUST_FWD(args)...});
  }
};

template <unsigned int Pos>
struct placeholder
{
  using type = actor<argument<Pos>>;
};

// composition of actors/nodes
template <typename...>
struct composite;

template <typename Eval, typename SubExpr>
struct composite<Eval, SubExpr>
{
  constexpr composite() = default;

  // TODO(bgruber): drop ctor and use aggregate initialization in C++17
  _CCCL_HOST_DEVICE composite(const Eval& eval, const SubExpr& subexpr)
      : m_eval(eval)
      , m_subexpr(subexpr)
  {}

  template <typename... Ts>
  _CCCL_HOST_DEVICE auto eval(Ts&&... args) const
    -> decltype(::cuda::std::declval<Eval>().eval(::cuda::std::declval<SubExpr>().eval(THRUST_FWD(args)...)))
  {
    return m_eval.eval(m_subexpr.eval(THRUST_FWD(args)...));
  }

private:
  Eval m_eval;
  SubExpr m_subexpr;
};

template <typename Eval, typename SubExpr1, typename SubExpr2>
struct composite<Eval, SubExpr1, SubExpr2>
{
  constexpr composite() = default;

  // TODO(bgruber): drop ctor and use aggregate initialization in C++17
  _CCCL_HOST_DEVICE composite(const Eval& eval, const SubExpr1& subexpr1, const SubExpr2& subexpr2)
      : m_eval(eval)
      , m_subexpr1(subexpr1)
      , m_subexpr2(subexpr2)
  {}

  template <typename... Ts>
  _CCCL_HOST_DEVICE auto eval(Ts&&... args) const
    -> decltype(::cuda::std::declval<Eval>().eval(::cuda::std::declval<SubExpr1>().eval(THRUST_FWD(args)...),
                                                  ::cuda::std::declval<SubExpr2>().eval(THRUST_FWD(args)...)))
  {
    return m_eval.eval(m_subexpr1.eval(THRUST_FWD(args)...), m_subexpr2.eval(THRUST_FWD(args)...));
  }

private:
  Eval m_eval;
  SubExpr1 m_subexpr1;
  SubExpr2 m_subexpr2;
};

template <typename Eval>
struct actor;

// Adapts a transparent unary functor from functional.h (e.g. ::cuda::std::negate<>) into the Eval interface.
template <typename F>
struct operator_adaptor : F
{
  constexpr operator_adaptor() = default;

  _CCCL_HOST_DEVICE operator_adaptor(F f)
      : F(::cuda::std::move(f))
  {}

  template <typename... Ts>
  _CCCL_HOST_DEVICE auto eval(Ts&&... args) const -> decltype(F{}(THRUST_FWD(args)...))
  {
    return static_cast<const F&>(*this)(THRUST_FWD(args)...);
  }
};

// a node returning a fixed value
template <typename T>
struct value
{
  T m_val;

  template <typename... Ts>
  _CCCL_HOST_DEVICE T eval(Ts&&...) const
  {
    return m_val;
  }
};

template <typename T>
_CCCL_HOST_DEVICE auto make_actor(T&& x) -> actor<value<::cuda::std::decay_t<T>>>
{
  return {{THRUST_FWD(x)}};
}

template <typename Eval>
_CCCL_HOST_DEVICE auto make_actor(actor<Eval> x) -> actor<Eval>
{
  return x;
}

template <typename Eval, typename SubExpr>
_CCCL_HOST_DEVICE auto compose(Eval e, const SubExpr& subexpr)
  -> decltype(actor<composite<operator_adaptor<Eval>, decltype(make_actor(subexpr))>>{
    {{::cuda::std::move(e)}, make_actor(subexpr)}})
{
  return actor<composite<operator_adaptor<Eval>, decltype(make_actor(subexpr))>>{
    {{::cuda::std::move(e)}, make_actor(subexpr)}};
}

template <typename Eval, typename SubExpr1, typename SubExpr2>
_CCCL_HOST_DEVICE auto compose(Eval e, const SubExpr1& subexpr1, const SubExpr2& subexpr2)
  -> decltype(actor<composite<operator_adaptor<Eval>, decltype(make_actor(subexpr1)), decltype(make_actor(subexpr2))>>{
    {{::cuda::std::move(e)}, make_actor(subexpr1), make_actor(subexpr2)}})
{
  return actor<composite<operator_adaptor<Eval>, decltype(make_actor(subexpr1)), decltype(make_actor(subexpr2))>>{
    {{::cuda::std::move(e)}, make_actor(subexpr1), make_actor(subexpr2)}};
}
} // namespace functional
} // namespace detail
THRUST_NAMESPACE_END
