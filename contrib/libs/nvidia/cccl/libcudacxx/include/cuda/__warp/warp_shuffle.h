//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPO__RATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_WARP_SHUFFLE_H
#define _CUDA___WARP_WARP_SHUFFLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  if __cccl_ptx_isa >= 600

#    include <cuda/__cmath/ceil_div.h>
#    include <cuda/__ptx/instructions/get_sreg.h>
#    include <cuda/__ptx/instructions/shfl_sync.h>
#    include <cuda/std/__bit/has_single_bit.h>
#    include <cuda/std/__concepts/concept_macros.h>
#    include <cuda/std/__memory/addressof.h>
#    include <cuda/std/__type_traits/enable_if.h>
#    include <cuda/std/__type_traits/integral_constant.h>
#    include <cuda/std/__type_traits/is_pointer.h>
#    include <cuda/std/__type_traits/is_void.h>
#    include <cuda/std/__type_traits/remove_cvref.h>
#    include <cuda/std/cstdint>

#    include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

template <typename _Tp>
struct warp_shuffle_result
{
  _Tp data;
  bool pred;

  template <typename _Up = _Tp>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE
  operator cuda::std::enable_if_t<!cuda::std::is_array_v<_Up>, _Up>() const
  {
    return data;
  }
};

template <int _Width = 32, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Up> warp_shuffle_idx(
  const _Tp& __data, int __src_lane, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = _CUDA_VSTD::is_same_v<_Up, void*> || _CUDA_VSTD::is_same_v<_Up, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(_CUDA_VSTD::has_single_bit(uint32_t{_Width}) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  if constexpr (_Width == 1)
  {
    return warp_shuffle_result<_Up>{__data, true};
  }
  else
  {
    constexpr int __ratio = ::cuda::ceil_div(sizeof(_Up), sizeof(uint32_t));
    auto __clamp_segmask  = (_Width - 1u) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    _CUDA_VSTD::memcpy(
      static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Up));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_idx(__array[i], __pred, __src_lane, __clamp_segmask, __lane_mask);
    }
    warp_shuffle_result<_Up> __result;
    __result.pred = __pred;
    _CUDA_VSTD::memcpy(
      static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
    return __result;
  }
}

template <int _Width, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Up>
warp_shuffle_idx(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return _CUDA_DEVICE::warp_shuffle_idx(__data, __src_lane, 0xFFFFFFFF, __width);
}

template <int _Width = 32, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Tp> warp_shuffle_up(
  const _Tp& __data, int __delta, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = _CUDA_VSTD::is_same_v<_Up, void*> || _CUDA_VSTD::is_same_v<_Up, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(_CUDA_VSTD::has_single_bit(uint32_t{_Width}) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __delta, &__pred1),
                                                           "all active lanes must have the same delta");))
  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__delta == 0, "delta must be 0 when Width == 1");
    return warp_shuffle_result<_Up>{__data, true};
  }
  else
  {
    _CCCL_ASSERT(__delta >= 1 && __delta < _Width, "delta must be in the range [1, _Width)");
    constexpr int __ratio = ::cuda::ceil_div(sizeof(_Up), sizeof(uint32_t));
    auto __clamp_segmask  = (__warp_size - _Width) << 8;
    bool __pred;
    uint32_t __array[__ratio];
    _CUDA_VSTD::memcpy(
      static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Up));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_up(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    warp_shuffle_result<_Up> __result;
    __result.pred = __pred;
    _CUDA_VSTD::memcpy(
      static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
    return __result;
  }
}

template <int _Width, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Up>
warp_shuffle_up(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return _CUDA_DEVICE::warp_shuffle_up(__data, __src_lane, 0xFFFFFFFF, __width);
}

template <int _Width = 32, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Up> warp_shuffle_down(
  const _Tp& __data, int __delta, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = _CUDA_VSTD::is_same_v<_Up, void*> || _CUDA_VSTD::is_same_v<_Up, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(_CUDA_VSTD::has_single_bit(uint32_t{_Width}) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __delta, &__pred1),
                                                           "all active lanes must have the same delta");))
  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__delta == 0, "delta must be 0 when Width == 1");
    return warp_shuffle_result<_Up>{__data, true};
  }
  else
  {
    _CCCL_ASSERT(__delta >= 1 && __delta < _Width, "delta must be in the range [1, _Width)");
    constexpr int __ratio = ::cuda::ceil_div(sizeof(_Up), sizeof(uint32_t));
    auto __clamp_segmask  = (_Width - 1u) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    _CUDA_VSTD::memcpy(
      static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Up));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_down(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    warp_shuffle_result<_Up> __result;
    __result.pred = __pred;
    _CUDA_VSTD::memcpy(
      static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
    return __result;
  }
}

template <int _Width, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Tp>
warp_shuffle_down(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return _CUDA_DEVICE::warp_shuffle_down(__data, __src_lane, 0xFFFFFFFF, __width);
}

template <int _Width = 32, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Up> warp_shuffle_xor(
  const _Tp& __data, int __xor_mask, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = _CUDA_VSTD::is_same_v<_Up, void*> || _CUDA_VSTD::is_same_v<_Up, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(_CUDA_VSTD::has_single_bit(uint32_t{_Width}) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __xor_mask, &__pred1),
                                                           "all active lanes must have the same delta");))
  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__xor_mask == 0, "delta must be 0 when Width == 1");
    return warp_shuffle_result<_Up>{__data, true};
  }
  else
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "delta must be in the range [1, _Width)");
    constexpr int __ratio = ::cuda::ceil_div(sizeof(_Up), sizeof(uint32_t));
    auto __clamp_segmask  = (_Width - 1u) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    _CUDA_VSTD::memcpy(
      static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Up));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_bfly(__array[i], __pred, __xor_mask, __clamp_segmask, __lane_mask);
    }
    warp_shuffle_result<_Up> __result;
    __result.pred = __pred;
    _CUDA_VSTD::memcpy(
      static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
    return __result;
  }
}

template <int _Width, typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE warp_shuffle_result<_Up>
warp_shuffle_xor(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return _CUDA_DEVICE::warp_shuffle_xor(__data, __src_lane, 0xFFFFFFFF, __width);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#    include <cuda/std/__cccl/epilogue.h>

#  endif // __cccl_ptx_isa >= 600
#endif // _CCCL_CUDA_COMPILATION()
#endif // _CUDA___WARP_WARP_SHUFFLE_H
