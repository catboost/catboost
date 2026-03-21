//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_IS_HEAP_H
#define _LIBCUDACXX___ALGORITHM_IS_HEAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/comp_ref_type.h>
#include <cuda/std/__algorithm/is_heap_until.h>
#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _RandomAccessIterator, class _Compare>
[[nodiscard]] _CCCL_API constexpr bool
is_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
  return _CUDA_VSTD::__is_heap_until(__first, __last, static_cast<__comp_ref_type<_Compare>>(__comp)) == __last;
}

template <class _RandomAccessIterator>
[[nodiscard]] _CCCL_API constexpr bool is_heap(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
  return _CUDA_VSTD::is_heap(__first, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_IS_HEAP_H
