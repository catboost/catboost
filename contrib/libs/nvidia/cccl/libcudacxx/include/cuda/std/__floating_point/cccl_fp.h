//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_CCCL_FP_H
#define _LIBCUDACXX___FLOATING_POINT_CCCL_FP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/conversion_rank_order.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <__fp_format _Fmt>
class __cccl_fp
{
  static_assert(_Fmt != __fp_format::__invalid);

  using __storage_type = __fp_storage_t<_Fmt>;

  __storage_type __storage_;

public:
  _CCCL_HIDE_FROM_ABI constexpr __cccl_fp() noexcept = default;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_fp_v<_Tp> _CCCL_AND __fp_is_implicit_conversion_v<_Tp, __cccl_fp>)
  _CCCL_API constexpr __cccl_fp(const _Tp&) noexcept
      : __cccl_fp{}
  {
    // todo: implement construction from a floating-point type using __fp_cast
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_fp_v<_Tp> _CCCL_AND(!__fp_is_implicit_conversion_v<_Tp, __cccl_fp>))
  _CCCL_API explicit constexpr __cccl_fp(const _Tp&) noexcept
      : __cccl_fp{}
  {
    // todo: implement construction from a floating-point type using __fp_cast
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_integral_v<_Tp>)
  _CCCL_API constexpr __cccl_fp(const _Tp&) noexcept
      : __cccl_fp{}
  {
    // todo: implement construction from an integral type using __fp_cast
  }

  _CCCL_HIDE_FROM_ABI constexpr __cccl_fp(const __cccl_fp&) noexcept = default;

  _CCCL_HIDE_FROM_ABI constexpr __cccl_fp& operator=(const __cccl_fp&) noexcept = default;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_fp_v<_Tp> _CCCL_AND(!__is_ext_cccl_fp_v<_Tp>)
                   _CCCL_AND __fp_is_implicit_conversion_v<__cccl_fp, _Tp>)
  _CCCL_API constexpr operator _Tp() const noexcept
  {
    // todo: implement conversion to a floating-point type using __fp_cast
    return _Tp{};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_fp_v<_Tp> _CCCL_AND(!__is_ext_cccl_fp_v<_Tp>)
                   _CCCL_AND(!__fp_is_implicit_conversion_v<__cccl_fp, _Tp>))
  _CCCL_API explicit constexpr operator _Tp() const noexcept
  {
    // todo: implement conversion to a floating-point type using __fp_cast
    return _Tp{};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_integral_v<_Tp>)
  _CCCL_API constexpr operator _Tp() const noexcept
  {
    // todo: implement conversion to an integral type using __fp_cast
    return _Tp{};
  }

  template <class _Tp>
  friend _CCCL_API constexpr _Tp __fp_from_storage(__fp_storage_of_t<_Tp> __v) noexcept;
  template <class _Tp>
  friend _CCCL_API constexpr __fp_storage_of_t<_Tp> __fp_get_storage(_Tp __v) noexcept;
};

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __cccl_fp<_Fmt> operator+(__cccl_fp<_Fmt> __v) noexcept
{
  return __v;
}

_CCCL_TEMPLATE(__fp_format _Fmt)
_CCCL_REQUIRES(__fp_is_signed_v<_Fmt>)
[[nodiscard]] _CCCL_API constexpr __cccl_fp<_Fmt> operator-(__cccl_fp<_Fmt> __v) noexcept
{
  return _CUDA_VSTD::__fp_neg(__v);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_CCCL_FP_H
