// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_DATA_H
#define _LIBCUDACXX___RANGES_DATA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/auto_cast.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

// [range.prim.data]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__data)

template <class _Tp>
_CCCL_CONCEPT __ptr_to_object = is_pointer_v<_Tp> && is_object_v<remove_pointer_t<_Tp>>;

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __member_data = __can_borrow<_Tp> && __workaround_52970<_Tp> && requires(_Tp&& __t) {
  { _LIBCUDACXX_AUTO_CAST(__t.data()) } -> __ptr_to_object;
};

template <class _Tp>
concept __ranges_begin_invocable = !__member_data<_Tp> && __can_borrow<_Tp> && requires(_Tp&& __t) {
  { _CUDA_VRANGES::begin(__t) } -> contiguous_iterator;
};
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__member_data_,
                       requires(_Tp&& __t)(requires(__can_borrow<_Tp>),
                                           requires(__workaround_52970<_Tp>),
                                           requires(__ptr_to_object<decltype(_LIBCUDACXX_AUTO_CAST(__t.data()))>)));

template <class _Tp>
_CCCL_CONCEPT __member_data = _CCCL_FRAGMENT(__member_data_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__ranges_begin_invocable_,
                       requires(_Tp&& __t)(requires(!__member_data<_Tp>),
                                           requires(__can_borrow<_Tp>),
                                           requires(contiguous_iterator<decltype(_CUDA_VRANGES::begin(__t))>)));

template <class _Tp>
_CCCL_CONCEPT __ranges_begin_invocable = _CCCL_FRAGMENT(__ranges_begin_invocable_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__member_data<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const noexcept(noexcept(__t.data()))
  {
    return __t.data();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__ranges_begin_invocable<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__t))))
  {
    return _CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__t));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto data = __data::__fn{};
} // namespace __cpo

// [range.prim.cdata]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__cdata)
struct __fn
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_lvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::data(static_cast<const remove_reference_t<_Tp>&>(__t))))
      -> decltype(_CUDA_VRANGES::data(static_cast<const remove_reference_t<_Tp>&>(__t)))
  {
    return _CUDA_VRANGES::data(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_rvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::data(static_cast<const _Tp&&>(__t))))
      -> decltype(_CUDA_VRANGES::data(static_cast<const _Tp&&>(__t)))
  {
    return _CUDA_VRANGES::data(static_cast<const _Tp&&>(__t));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto cdata = __cdata::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_DATA_H
