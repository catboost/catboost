//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_UNWRAP_RANGE_H
#define _LIBCUDACXX___ALGORITHM_UNWRAP_RANGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/unwrap_iter.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __unwrap_range and __rewrap_range are used to unwrap ranges which may have different iterator and sentinel types.
// __unwrap_iter and __rewrap_iter don't work for this, because they assume that the iterator and sentinel have
// the same type. __unwrap_range tries to get two iterators and then forward to __unwrap_iter.

#if _CCCL_STD_VER >= 2020
template <class _Iter, class _Sent>
struct __unwrap_range_impl
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr auto __unwrap(_Iter __first, _Sent __sent)
    requires random_access_iterator<_Iter> && sized_sentinel_for<_Sent, _Iter>
  {
    auto __last = ranges::next(__first, __sent);
    return pair{_CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::move(__first)),
                _CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::move(__last))};
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr auto __unwrap(_Iter __first, _Sent __last)
  {
    return pair{_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last)};
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr auto
  __rewrap(_Iter __orig_iter, decltype(_CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::move(__orig_iter))) __iter)
    requires random_access_iterator<_Iter> && sized_sentinel_for<_Sent, _Iter>
  {
    return _CUDA_VSTD::__rewrap_iter(_CUDA_VSTD::move(__orig_iter), _CUDA_VSTD::move(__iter));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr auto __rewrap(const _Iter&, _Iter __iter)
    requires(!(random_access_iterator<_Iter> && sized_sentinel_for<_Sent, _Iter>) )
  {
    return __iter;
  }
};

template <class _Iter>
struct __unwrap_range_impl<_Iter, _Iter>
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr auto __unwrap(_Iter __first, _Iter __last)
  {
    return pair{_CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::move(__first)),
                _CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::move(__last))};
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr auto __rewrap(_Iter __orig_iter, decltype(_CUDA_VSTD::__unwrap_iter(__orig_iter)) __iter)
  {
    return _CUDA_VSTD::__rewrap_iter(_CUDA_VSTD::move(__orig_iter), _CUDA_VSTD::move(__iter));
  }
};

_CCCL_EXEC_CHECK_DISABLE
template <class _Iter, class _Sent>
_CCCL_API constexpr auto __unwrap_range(_Iter __first, _Sent __last)
{
  return __unwrap_range_impl<_Iter, _Sent>::__unwrap(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Sent, class _Iter, class _Unwrapped>
_CCCL_API constexpr _Iter __rewrap_range(_Iter __orig_iter, _Unwrapped __iter)
{
  return __unwrap_range_impl<_Iter, _Sent>::__rewrap(_CUDA_VSTD::move(__orig_iter), _CUDA_VSTD::move(__iter));
}
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
_CCCL_EXEC_CHECK_DISABLE
template <class _Iter, class _Unwrapped = decltype(_CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::declval<_Iter>()))>
_CCCL_API constexpr pair<_Unwrapped, _Unwrapped> __unwrap_range(_Iter __first, _Iter __last)
{
  return _CUDA_VSTD::make_pair(
    _CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::move(__first)), _CUDA_VSTD::__unwrap_iter(_CUDA_VSTD::move(__last)));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Iter, class _Unwrapped>
_CCCL_API constexpr _Iter __rewrap_range(_Iter __orig_iter, _Unwrapped __iter)
{
  return _CUDA_VSTD::__rewrap_iter(_CUDA_VSTD::move(__orig_iter), _CUDA_VSTD::move(__iter));
}
#endif // _CCCL_STD_VER <= 2017

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_UNWRAP_RANGE_H
