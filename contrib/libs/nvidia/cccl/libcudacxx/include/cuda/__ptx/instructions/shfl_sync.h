//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_SHFL_SYNC_H
#define _CUDA_PTX_SHFL_SYNC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/cstdint>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

#if __cccl_ptx_isa >= 600

enum class __dot_shfl_mode
{
  __up,
  __down,
  __bfly,
  __idx
};

[[maybe_unused]]
_CCCL_DEVICE static inline uint32_t
__shfl_sync_dst_lane(__dot_shfl_mode __shfl_mode, uint32_t __lane_idx_offset, uint32_t __clamp_segmask)
{
  auto __lane     = _CUDA_VPTX::get_sreg_laneid();
  auto __clamp    = __clamp_segmask & 0b11111;
  auto __segmask  = __clamp_segmask >> 8;
  auto __max_lane = (__lane & __segmask) | (__clamp & ~__segmask);
  uint32_t __j    = 0;
  if (__shfl_mode == __dot_shfl_mode::__idx)
  {
    auto __min_lane = __lane & __segmask;
    __j             = __min_lane | (__lane_idx_offset & ~__segmask);
  }
  else if (__shfl_mode == __dot_shfl_mode::__up)
  {
    __j = __lane_idx_offset >= __lane ? 0 : __lane - __lane_idx_offset;
  }
  else if (__shfl_mode == __dot_shfl_mode::__down)
  {
    __j = __lane + __lane_idx_offset;
  }
  else
  {
    __j = __lane ^ __lane_idx_offset;
  }
  auto __dst = __shfl_mode == __dot_shfl_mode::__up
               ? (__j >= __max_lane ? __j : __lane) //
               : (__j <= __max_lane ? __j : __lane);
  return (1u << __dst);
}

template <typename _Tp>
_CCCL_DEVICE static inline void __shfl_sync_checks(
  __dot_shfl_mode __shfl_mode,
  _Tp,
  [[maybe_unused]] uint32_t __lane_idx_offset,
  [[maybe_unused]] uint32_t __clamp_segmask,
  [[maybe_unused]] uint32_t __lane_mask)
{
  static_assert(sizeof(_Tp) == 4, "shfl.sync only accepts 4-byte data types");
  if (__shfl_mode != __dot_shfl_mode::__idx)
  {
    _CCCL_ASSERT(__lane_idx_offset < 32, "the lane index or offset must be less than the warp size");
  }
  _CCCL_ASSERT(__lane_mask != 0, "lane_mask must be non-zero");
  _CCCL_ASSERT((__clamp_segmask | 0b1111100011111) == 0b1111100011111,
               "clamp value + segmentation mask must use the bit positions [0:4] and [8:12]");
  _CCCL_ASSERT(_CUDA_VPTX::__shfl_sync_dst_lane(__shfl_mode, __lane_idx_offset, __clamp_segmask) & __lane_mask,
               "the destination lane must be a member of the lane mask");
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_idx(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__idx, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.idx.b32 %0|p, %2, %3, %4, %5;                \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_idx(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__idx, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile("{                                                      \n\t\t"
               "shfl.sync.idx.b32 %0, %1, %2, %3, %4;                  \n\t\t"
               "}"
               : "=r"(__ret)
               : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_up(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__up, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.up.b32 %0|p, %2, %3, %4, %5;                 \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_up(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__up, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile("{                                                      \n\t\t"
               "shfl.sync.up.b32 %0, %1, %2, %3, %4;                   \n\t\t"
               "}"
               : "=r"(__ret)
               : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_down(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__down, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.down.b32 %0|p, %2, %3, %4, %5;               \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_down(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__down, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile("{                                                      \n\t\t"
               "shfl.sync.down.b32 %0, %1, %2, %3, %4;                 \n\t\t"
               "}"
               : "=r"(__ret)
               : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_bfly(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__bfly, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.bfly.b32 %0|p, %2, %3, %4, %5;               \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_bfly(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  _CUDA_VPTX::__shfl_sync_checks(__dot_shfl_mode::__bfly, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = _CUDA_VSTD::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile( //
    "{                                                      \n\t\t"
    "shfl.sync.bfly.b32 %0, %1, %2, %3, %4;                 \n\t\t"
    "}"
    : "=r"(__ret)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return _CUDA_VSTD::bit_cast<uint32_t>(__ret);
}

#endif // __cccl_ptx_isa >= 600

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_SHFL_SYNC_H
