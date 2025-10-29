//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_TUPLE_H
#define _LIBCUDACXX___COMPLEX_TUPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/complex.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct tuple_size<complex<_Tp>> : _CUDA_VSTD::integral_constant<size_t, 2>
{};

template <size_t _Index, class _Tp>
  struct tuple_element<_Index, complex<_Tp>> : _CUDA_VSTD::enable_if < _Index<2, _Tp>
{};

template <class _Tp>
struct __get_complex_impl
{
  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr _Tp& get(complex<_Tp>& __z) noexcept
  {
    return (_Index == 0) ? __z.__re_ : __z.__im_;
  }

  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr _Tp&& get(complex<_Tp>&& __z) noexcept
  {
    return _CUDA_VSTD::move((_Index == 0) ? __z.__re_ : __z.__im_);
  }

  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr const _Tp& get(const complex<_Tp>& __z) noexcept
  {
    return (_Index == 0) ? __z.__re_ : __z.__im_;
  }

  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr const _Tp&& get(const complex<_Tp>&& __z) noexcept
  {
    return _CUDA_VSTD::move((_Index == 0) ? __z.__re_ : __z.__im_);
  }
};

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp& get(complex<_Tp>& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");

  return __get_complex_impl<_Tp>::template get<_Index>(__z);
}

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp&& get(complex<_Tp>&& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");

  return __get_complex_impl<_Tp>::template get<_Index>(_CUDA_VSTD::move(__z));
}

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp& get(const complex<_Tp>& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");

  return __get_complex_impl<_Tp>::template get<_Index>(__z);
}

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp&& get(const complex<_Tp>&& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");

  return __get_complex_impl<_Tp>::template get<_Index>(_CUDA_VSTD::move(__z));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_TUPLE_H
