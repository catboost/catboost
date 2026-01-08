//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPO__RATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_WARP_MATCH_H
#define _CUDA___WARP_WARP_MATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__warp/lane_mask.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__type_traits/remove_cv.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/cstring>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

extern "C" _CCCL_DEVICE void __cuda__match_all_sync_is_not_supported_before_SM_70__();

template <typename _Tp, typename _Up = _CUDA_VSTD::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE bool
warp_match_all(const _Tp& __data, lane_mask __lane_mask = lane_mask::all())
{
  _CCCL_ASSERT(__lane_mask != lane_mask::none(), "lane_mask must be non-zero");
  constexpr int __ratio = ::cuda::ceil_div(sizeof(_Up), sizeof(uint32_t));
  uint32_t __array[__ratio];
  _CUDA_VSTD::memcpy(__array, _CUDA_VSTD::addressof(__data), sizeof(_Up));
  bool __ret = true;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < __ratio; ++i)
  {
    int __pred = false;
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,
                      (::__match_all_sync(__lane_mask.value(), __array[i], &__pred);),
                      (_CUDA_DEVICE::__cuda__match_all_sync_is_not_supported_before_SM_70__();));
    __ret = __ret && __pred;
  }
  return __ret;
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()
#endif // _CUDA___WARP_WARP_MATCH_H
