//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_MASK_H
#define _LIBCUDACXX___FLOATING_POINT_MASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <__fp_format _Fmt>
inline constexpr auto __fp_sign_mask_v =
  static_cast<__fp_storage_t<_Fmt>>(__fp_storage_t<_Fmt>(1) << (__fp_exp_nbits_v<_Fmt> + __fp_mant_nbits_v<_Fmt>) );

template <class _Tp>
inline constexpr auto __fp_sign_mask_of_v = __fp_sign_mask_v<__fp_format_of_v<_Tp>>;

template <__fp_format _Fmt>
inline constexpr auto __fp_exp_mask_v = static_cast<__fp_storage_t<_Fmt>>(
  ((__fp_storage_t<_Fmt>(1) << __fp_exp_nbits_v<_Fmt>) -1) << __fp_mant_nbits_v<_Fmt>);

template <class _Tp>
inline constexpr auto __fp_exp_mask_of_v = __fp_exp_mask_v<__fp_format_of_v<_Tp>>;

template <__fp_format _Fmt>
inline constexpr auto __fp_mant_mask_v =
  static_cast<__fp_storage_t<_Fmt>>((__fp_storage_t<_Fmt>(1) << __fp_mant_nbits_v<_Fmt>) -1);

template <class _Tp>
inline constexpr auto __fp_mant_mask_of_v = __fp_mant_mask_v<__fp_format_of_v<_Tp>>;

template <__fp_format _Fmt>
inline constexpr auto __fp_exp_mant_mask_v =
  static_cast<__fp_storage_t<_Fmt>>(__fp_exp_mask_v<_Fmt> | __fp_mant_mask_v<_Fmt>);

template <class _Tp>
inline constexpr auto __fp_exp_mant_mask_of_v = __fp_exp_mant_mask_v<__fp_format_of_v<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_MASK_H
