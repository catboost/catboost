//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_COPY_N_H
#define _LIBCUDACXX___ALGORITHM_COPY_N_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__utility/convert_to_integral.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator,
          class _Size,
          class _OutputIterator,
          enable_if_t<__has_input_traversal<_InputIterator>, int>          = 0,
          enable_if_t<!__has_random_access_traversal<_InputIterator>, int> = 0>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _OutputIterator
copy_n(_InputIterator __first, _Size __orig_n, _OutputIterator __result)
{
  using _IntegralSize = decltype(__convert_to_integral(__orig_n));
  _IntegralSize __n   = static_cast<_IntegralSize>(__orig_n);
  if (__n > 0)
  {
    *__result = *__first;
    ++__result;
    for (--__n; __n > 0; --__n)
    {
      ++__first;
      *__result = *__first;
      ++__result;
    }
  }
  return __result;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator,
          class _Size,
          class _OutputIterator,
          enable_if_t<__has_random_access_traversal<_InputIterator>, int> = 0>
_CCCL_API constexpr _OutputIterator copy_n(_InputIterator __first, _Size __orig_n, _OutputIterator __result)
{
  using _IntegralSize = decltype(__convert_to_integral(__orig_n));
  _IntegralSize __n   = static_cast<_IntegralSize>(__orig_n);
  return _CUDA_VSTD::copy(__first, __first + __n, __result);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_COPY_N_H
