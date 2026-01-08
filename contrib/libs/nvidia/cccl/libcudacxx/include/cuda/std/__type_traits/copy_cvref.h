//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H
#define _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/add_rvalue_reference.h>

#if _CCCL_COMPILER(GCC, <, 7)
#  define _CCCL_ADD_LVALUE_REFERENCE_WAR(_Tp) add_lvalue_reference_t<_Tp>
#  define _CCCL_ADD_RVALUE_REFERENCE_WAR(_Tp) add_rvalue_reference_t<_Tp>
#else
#  define _CCCL_ADD_LVALUE_REFERENCE_WAR(_Tp) _Tp&
#  define _CCCL_ADD_RVALUE_REFERENCE_WAR(_Tp) _Tp&&
#endif

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __apply_cvref_
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _Tp;
};

struct __apply_cvref_c
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = const _Tp;
};

struct __apply_cvref_v
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = volatile _Tp;
};

struct __apply_cvref_cv
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = const volatile _Tp;
};

struct __apply_cvref_lr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_LVALUE_REFERENCE_WAR(_Tp);
};

struct __apply_cvref_clr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_LVALUE_REFERENCE_WAR(const _Tp);
};

struct __apply_cvref_vlr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_LVALUE_REFERENCE_WAR(volatile _Tp);
};

struct __apply_cvref_cvlr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_LVALUE_REFERENCE_WAR(const volatile _Tp);
};

struct __apply_cvref_rr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_RVALUE_REFERENCE_WAR(_Tp);
};

struct __apply_cvref_crr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_RVALUE_REFERENCE_WAR(const _Tp);
};

struct __apply_cvref_vrr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_RVALUE_REFERENCE_WAR(volatile _Tp);
};

struct __apply_cvref_cvrr
{
  template <class _Tp>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_ADD_RVALUE_REFERENCE_WAR(const volatile _Tp);
};

template <class>
extern __apply_cvref_ __apply_cvref;
template <class _Tp>
extern __apply_cvref_c __apply_cvref<const _Tp>;
template <class _Tp>
extern __apply_cvref_v __apply_cvref<volatile _Tp>;
template <class _Tp>
extern __apply_cvref_cv __apply_cvref<const volatile _Tp>;
template <class _Tp>
extern __apply_cvref_lr __apply_cvref<_Tp&>;
template <class _Tp>
extern __apply_cvref_clr __apply_cvref<const _Tp&>;
template <class _Tp>
extern __apply_cvref_vlr __apply_cvref<volatile _Tp&>;
template <class _Tp>
extern __apply_cvref_cvlr __apply_cvref<const volatile _Tp&>;
template <class _Tp>
extern __apply_cvref_rr __apply_cvref<_Tp&&>;
template <class _Tp>
extern __apply_cvref_crr __apply_cvref<const _Tp&&>;
template <class _Tp>
extern __apply_cvref_vrr __apply_cvref<volatile _Tp&&>;
template <class _Tp>
extern __apply_cvref_cvrr __apply_cvref<const volatile _Tp&&>;

template <class _Tp>
using __apply_cvref_fn = decltype(__apply_cvref<_Tp>);

template <class _From, class _To>
using __copy_cvref_t = typename __apply_cvref_fn<_From>::template __call<_To>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#undef _CCCL_ADD_RVALUE_REFERENCE_WAR
#undef _CCCL_ADD_LVALUE_REFERENCE_WAR

#endif // _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H
