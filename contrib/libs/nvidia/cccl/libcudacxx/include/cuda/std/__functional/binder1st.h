// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BINDER1ST_H
#define _LIBCUDACXX___FUNCTIONAL_BINDER1ST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/unary_function.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class __Operation>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED
binder1st : public __unary_function<typename __Operation::second_argument_type, typename __Operation::result_type>
{
protected:
  __Operation op;
  typename __Operation::first_argument_type value;

public:
  _CCCL_API inline binder1st(const __Operation& __x, const typename __Operation::first_argument_type __y)
      : op(__x)
      , value(__y)
  {}
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline typename __Operation::result_type operator()(typename __Operation::second_argument_type& __x) const
  {
    return op(value, __x);
  }
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline typename __Operation::result_type
  operator()(const typename __Operation::second_argument_type& __x) const
  {
    return op(value, __x);
  }
};

template <class __Operation, class _Tp>
_LIBCUDACXX_DEPRECATED _CCCL_API inline binder1st<__Operation> bind1st(const __Operation& __op, const _Tp& __x)
{
  return binder1st<__Operation>(__op, __x);
}

_CCCL_SUPPRESS_DEPRECATED_POP

#endif // defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_BINDER1ST_H
