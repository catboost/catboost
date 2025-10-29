// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ITER_MOVE_H
#define _LIBCUDACXX___ITERATOR_ITER_MOVE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wvoid-ptr-dereference")

// [iterator.cust.move]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__iter_move)

_CCCL_HOST_DEVICE void iter_move();

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __unqualified_iter_move =
  __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) { iter_move(_CUDA_VSTD::forward<_Tp>(__t)); };

template <class _Tp>
concept __move_deref = !__unqualified_iter_move<_Tp> && requires(_Tp&& __t) {
  *__t;
  requires is_lvalue_reference_v<decltype(*__t)>;
};

template <class _Tp>
concept __just_deref = !__unqualified_iter_move<_Tp> && !__move_deref<_Tp> && requires(_Tp&& __t) {
  *__t;
  requires(!is_lvalue_reference_v<decltype(*__t)>);
};

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__unqualified_iter_move_,
                       requires(_Tp&& __t)(requires(__class_or_enum<remove_cvref_t<_Tp>>),
                                           ((void) iter_move(_CUDA_VSTD::forward<_Tp>(__t)))));

template <class _Tp>
_CCCL_CONCEPT __unqualified_iter_move = _CCCL_FRAGMENT(__unqualified_iter_move_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __move_deref_,
  requires(_Tp&& __t)(requires(!__unqualified_iter_move<_Tp>), requires(is_lvalue_reference_v<decltype(*__t)>)));

template <class _Tp>
_CCCL_CONCEPT __move_deref = _CCCL_FRAGMENT(__move_deref_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__just_deref_,
                       requires(_Tp&& __t)(requires(!__unqualified_iter_move<_Tp>),
                                           requires(!__move_deref<_Tp>),
                                           requires(!is_lvalue_reference_v<decltype(*__t)>)));

template <class _Tp>
_CCCL_CONCEPT __just_deref = _CCCL_FRAGMENT(__just_deref_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

// [iterator.cust.move]

struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip)
  _CCCL_REQUIRES(__unqualified_iter_move<_Ip>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator()(_Ip&& __i) const
    noexcept(noexcept(iter_move(_CUDA_VSTD::forward<_Ip>(__i))))
  {
    return iter_move(_CUDA_VSTD::forward<_Ip>(__i));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip)
  _CCCL_REQUIRES(__move_deref<_Ip>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Ip&& __i) const
    noexcept(noexcept(_CUDA_VSTD::move(*_CUDA_VSTD::forward<_Ip>(__i))))
      -> decltype(_CUDA_VSTD::move(*_CUDA_VSTD::forward<_Ip>(__i)))
  {
    return _CUDA_VSTD::move(*_CUDA_VSTD::forward<_Ip>(__i));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip)
  _CCCL_REQUIRES(__just_deref<_Ip>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Ip&& __i) const noexcept(noexcept(*_CUDA_VSTD::forward<_Ip>(__i)))
    -> decltype(*_CUDA_VSTD::forward<_Ip>(__i))
  {
    return *_CUDA_VSTD::forward<_Ip>(__i);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO
inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto iter_move = __iter_move::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()
template <__dereferenceable _Tp>
  requires requires(_Tp& __t) {
    { _CUDA_VRANGES::iter_move(__t) } -> __can_reference;
  }
using iter_rvalue_reference_t = decltype(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<_Tp&>()));

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__can_iter_rvalue_reference_t_,
                       requires(_Tp& __t)(requires(__dereferenceable<_Tp>),
                                          requires(__can_reference<decltype(_CUDA_VRANGES::iter_move(__t))>)));

template <class _Tp>
_CCCL_CONCEPT __can_iter_rvalue_reference_t = _CCCL_FRAGMENT(__can_iter_rvalue_reference_t_, _Tp);

template <class _Tp>
using __iter_rvalue_reference_t = decltype(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<_Tp&>()));

template <class _Tp>
using iter_rvalue_reference_t = enable_if_t<__can_iter_rvalue_reference_t<_Tp>, __iter_rvalue_reference_t<_Tp>>;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_ITER_MOVE_H
