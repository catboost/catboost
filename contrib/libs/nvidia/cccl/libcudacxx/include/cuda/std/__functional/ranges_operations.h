// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_RANGES_OPERATIONS_H
#define _LIBCUDACXX___FUNCTIONAL_RANGES_OPERATIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

struct equal_to
{
  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(equality_comparable_with<_Tp, _Up>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t, _Up&& __u) const
    noexcept(noexcept(bool(_CUDA_VSTD::forward<_Tp>(__t) == _CUDA_VSTD::forward<_Up>(__u))))
  {
    return _CUDA_VSTD::forward<_Tp>(__t) == _CUDA_VSTD::forward<_Up>(__u);
  }

  using is_transparent = void;
};

struct not_equal_to
{
  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(equality_comparable_with<_Tp, _Up>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t, _Up&& __u) const
    noexcept(noexcept(bool(!(_CUDA_VSTD::forward<_Tp>(__t) == _CUDA_VSTD::forward<_Up>(__u)))))
  {
    return !(_CUDA_VSTD::forward<_Tp>(__t) == _CUDA_VSTD::forward<_Up>(__u));
  }

  using is_transparent = void;
};

struct less
{
  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(totally_ordered_with<_Tp, _Up>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t, _Up&& __u) const
    noexcept(noexcept(bool(_CUDA_VSTD::forward<_Tp>(__t) < _CUDA_VSTD::forward<_Up>(__u))))
  {
    return _CUDA_VSTD::forward<_Tp>(__t) < _CUDA_VSTD::forward<_Up>(__u);
  }

  using is_transparent = void;
};

struct less_equal
{
  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(totally_ordered_with<_Tp, _Up>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t, _Up&& __u) const
    noexcept(noexcept(bool(!(_CUDA_VSTD::forward<_Up>(__u) < _CUDA_VSTD::forward<_Tp>(__t)))))
  {
    return !(_CUDA_VSTD::forward<_Up>(__u) < _CUDA_VSTD::forward<_Tp>(__t));
  }

  using is_transparent = void;
};

struct greater
{
  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(totally_ordered_with<_Tp, _Up>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t, _Up&& __u) const
    noexcept(noexcept(bool(_CUDA_VSTD::forward<_Up>(__u) < _CUDA_VSTD::forward<_Tp>(__t))))
  {
    return _CUDA_VSTD::forward<_Up>(__u) < _CUDA_VSTD::forward<_Tp>(__t);
  }

  using is_transparent = void;
};

struct greater_equal
{
  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(totally_ordered_with<_Tp, _Up>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t, _Up&& __u) const
    noexcept(noexcept(bool(!(_CUDA_VSTD::forward<_Tp>(__t) < _CUDA_VSTD::forward<_Up>(__u)))))
  {
    return !(_CUDA_VSTD::forward<_Tp>(__t) < _CUDA_VSTD::forward<_Up>(__u));
  }

  using is_transparent = void;
};

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_RANGES_OPERATIONS_H
