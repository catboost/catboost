//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_UNWRAP_ITER_H
#define _LIBCUDACXX___ALGORITHM_UNWRAP_ITER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// TODO: Change the name of __unwrap_iter_impl to something more appropriate
// The job of __unwrap_iter is to remove iterator wrappers (like reverse_iterator or __wrap_iter),
// to reduce the number of template instantiations and to enable pointer-based optimizations e.g. in _CUDA_VSTD::copy.
// In debug mode, we don't do this.
//
// Some algorithms (e.g. _CUDA_VSTD::copy, but not _CUDA_VSTD::sort) need to convert an
// "unwrapped" result back into the original iterator type. Doing that is the job of __rewrap_iter.

// Default case - we can't unwrap anything
template <class _Iter, bool = __has_contiguous_traversal<_Iter>>
struct __unwrap_iter_impl
{
  _CCCL_EXEC_CHECK_DISABLE
  static _CCCL_API constexpr _Iter __rewrap(_Iter, _Iter __iter)
  {
    return __iter;
  }
  _CCCL_EXEC_CHECK_DISABLE
  static _CCCL_API constexpr _Iter __unwrap(_Iter __i) noexcept
  {
    return __i;
  }
};

// It's a contiguous iterator, so we can use a raw pointer instead
template <class _Iter>
struct __unwrap_iter_impl<_Iter, true>
{
  using _ToAddressT = decltype(_CUDA_VSTD::__to_address(_CUDA_VSTD::declval<_Iter>()));

  _CCCL_EXEC_CHECK_DISABLE
  static _CCCL_API constexpr _Iter __rewrap(_Iter __orig_iter, _ToAddressT __unwrapped_iter)
  {
    return __orig_iter + (__unwrapped_iter - _CUDA_VSTD::__to_address(__orig_iter));
  }

  _CCCL_EXEC_CHECK_DISABLE
  static _CCCL_API constexpr _ToAddressT __unwrap(_Iter __i) noexcept
  {
    return _CUDA_VSTD::__to_address(__i);
  }
};

template <class _Iter, class _Impl = __unwrap_iter_impl<_Iter>, enable_if_t<is_copy_constructible<_Iter>::value, int> = 0>
_CCCL_API constexpr decltype(_Impl::__unwrap(_CUDA_VSTD::declval<_Iter>())) __unwrap_iter(_Iter __i) noexcept
{
  return _Impl::__unwrap(__i);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _OrigIter, class _Iter, class _Impl = __unwrap_iter_impl<_OrigIter>>
_CCCL_API constexpr _OrigIter __rewrap_iter(_OrigIter __orig_iter, _Iter __iter) noexcept
{
  return _Impl::__rewrap(_CUDA_VSTD::move(__orig_iter), _CUDA_VSTD::move(__iter));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_UNWRAP_ITER_H
