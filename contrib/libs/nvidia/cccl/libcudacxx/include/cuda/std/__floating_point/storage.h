//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_STORAGE_H
#define _LIBCUDACXX___FLOATING_POINT_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr auto __fp_storage_type_impl() noexcept
{
  if constexpr (_Fmt == __fp_format::__fp8_nv_e4m3 || _Fmt == __fp_format::__fp8_nv_e5m2
                || _Fmt == __fp_format::__fp8_nv_e8m0 || _Fmt == __fp_format::__fp6_nv_e2m3
                || _Fmt == __fp_format::__fp6_nv_e3m2 || _Fmt == __fp_format::__fp4_nv_e2m1)
  {
    return uint8_t{};
  }
  else if constexpr (_Fmt == __fp_format::__binary16 || _Fmt == __fp_format::__bfloat16)
  {
    return uint16_t{};
  }
  else if constexpr (_Fmt == __fp_format::__binary32)
  {
    return uint32_t{};
  }
  else if constexpr (_Fmt == __fp_format::__binary64)
  {
    return uint64_t{};
  }
#if _CCCL_HAS_INT128()
  else if constexpr (_Fmt == __fp_format::__fp80_x86 || _Fmt == __fp_format::__binary128)
  {
    return __uint128_t{};
  }
#endif // _CCCL_HAS_INT128()
  else
  {
    static_assert(__always_false_v<decltype(_Fmt)>, "Unsupported floating point format");
  }
}

template <__fp_format _Fmt>
using __fp_storage_t = decltype(__fp_storage_type_impl<_Fmt>());

template <class _Tp>
using __fp_storage_of_t = __fp_storage_t<__fp_format_of_v<_Tp>>;

#if _CCCL_HAS_NVFP16()
struct __cccl_nvfp16_manip_helper : __half
{
  using __half::__x;
};
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
struct __cccl_nvbf16_manip_helper : __nv_bfloat16
{
  using __nv_bfloat16::__x;
};
#endif // _CCCL_HAS_NVBF16()

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_from_storage(__fp_storage_of_t<_Tp> __v) noexcept
{
  if constexpr (_CCCL_TRAIT(__is_std_fp, _Tp) || _CCCL_TRAIT(__is_ext_compiler_fp, _Tp))
  {
    return _CUDA_VSTD::bit_cast<_Tp>(__v);
  }
  else if constexpr (_CCCL_TRAIT(__is_ext_cccl_fp, _Tp))
  {
    _Tp __ret{};
    __ret.__storage_ = __v;
    return __ret;
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __half))
  {
    __cccl_nvfp16_manip_helper __helper{};
    __helper.__x = __v;
    return __helper;
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_bfloat16))
  {
    __cccl_nvbf16_manip_helper __helper{};
    __helper.__x = __v;
    return __helper;
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e4m3))
  {
    __nv_fp8_e4m3 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e5m2))
  {
    __nv_fp8_e5m2 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e8m0))
  {
    __nv_fp8_e8m0 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e2m3))
  {
    _CCCL_ASSERT((__v & 0xc0u) == 0u, "Invalid __nv_fp6_e2m3 storage value");
    __nv_fp6_e2m3 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e3m2))
  {
    _CCCL_ASSERT((__v & 0xc0u) == 0u, "Invalid __nv_fp6_e3m2 storage value");
    __nv_fp6_e3m2 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp4_e2m1))
  {
    _CCCL_ASSERT((__v & 0xf0u) == 0u, "Invalid __nv_fp4_e2m1 storage value");
    __nv_fp4_e2m1 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating point format");
  }
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES((!_CCCL_TRAIT(is_same, _Up, __fp_storage_of_t<_Tp>)))
_CCCL_API constexpr _Tp __fp_from_storage(const _Up& __v) noexcept = delete;

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __fp_storage_of_t<_Tp> __fp_get_storage(_Tp __v) noexcept
{
  if constexpr (_CCCL_TRAIT(__is_std_fp, _Tp) || _CCCL_TRAIT(__is_ext_compiler_fp, _Tp))
  {
    return _CUDA_VSTD::bit_cast<__fp_storage_of_t<_Tp>>(__v);
  }
  else if constexpr (_CCCL_TRAIT(__is_ext_cccl_fp, _Tp))
  {
    return __v.__storage_;
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __half))
  {
    return __cccl_nvfp16_manip_helper{__v}.__x;
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_bfloat16))
  {
    return __cccl_nvbf16_manip_helper{__v}.__x;
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e4m3))
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e5m2))
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e8m0))
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e2m3))
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e3m2))
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp4_e2m1))
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating point format");
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_STORAGE_H
