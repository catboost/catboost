// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BINARY_FUNCTION_H
#define _LIBCUDACXX___FUNCTIONAL_BINARY_FUNCTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

template <class _Arg1, class _Arg2, class _Result>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED binary_function
{
  using first_argument_type  = _Arg1;
  using second_argument_type = _Arg2;
  using result_type          = _Result;
};

#endif // defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

template <class _Arg1, class _Arg2, class _Result>
struct __binary_function_keep_layout_base
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using first_argument_type _LIBCUDACXX_DEPRECATED  = _Arg1;
  using second_argument_type _LIBCUDACXX_DEPRECATED = _Arg2;
  using result_type _LIBCUDACXX_DEPRECATED          = _Result;
#endif // _LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS
};

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Arg1, class _Arg2, class _Result>
using __binary_function = binary_function<_Arg1, _Arg2, _Result>;
_CCCL_SUPPRESS_DEPRECATED_POP
#else
template <class _Arg1, class _Arg2, class _Result>
using __binary_function = __binary_function_keep_layout_base<_Arg1, _Arg2, _Result>;
#endif // !_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_BINARY_FUNCTION_H
