//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_LANE_MASK_H
#define _CUDA___WARP_LANE_MASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

//! @brief A class representing a lane mask in a warp.
class lane_mask
{
  _CUDA_VSTD::uint32_t __value_;

public:
  //! @brief Constructs a lane mask object from a 32-bit unsigned integer.
  //!
  //! @param __v The value to initialize the lane mask with. Defaults to 0.
  //!
  //! @post `value() == __v`
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI explicit constexpr lane_mask(_CUDA_VSTD::uint32_t __v = 0) noexcept
      : __value_{__v}
  {}

  //! @brief Returns the value of the lane mask as a 32-bit unsigned integer.
  //!
  //! @return The value of the lane mask.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr _CUDA_VSTD::uint32_t value() const noexcept
  {
    return __value_;
  }

  //! @brief Converts the lane mask to a 32-bit unsigned integer.
  //!
  //! This operator allows explicit conversion of the lane mask to a 32-bit unsigned integer.
  //!
  //! @return The value of the lane mask as a 32-bit unsigned integer.
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI explicit constexpr operator _CUDA_VSTD::uint32_t() const noexcept
  {
    return __value_;
  }

  //! @brief Returns a lane mask object with no lane bits set.
  //!
  //! @return A lane mask with no lane bits set.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static constexpr lane_mask none() noexcept
  {
    return lane_mask{};
  }

  //! @brief Returns a lane mask object with all lane bits set.
  //!
  //! @return A lane mask with all lane bits set.
  //!
  //! @note This function may return a mask with 1s set even on inactive lane bits,
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static constexpr lane_mask all() noexcept
  {
    return lane_mask{0xffffffff};
  }

  //! @brief Returns a lane mask object with all currently active lane bits set.
  //!
  //! This function returns a lane_mask object equivalent to calling `lane_mask{::__activemask()}`.
  //!
  //! @return A lane mask with all active lane bits set.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_active() noexcept
  {
    return lane_mask{::__activemask()};
  }

  //! @brief Returns a lane mask object with the current lane bit set.
  //!
  //! This function is equivalent to constructing a lane_mask object with value of %%lanemask_eq PTX special register.
  //!
  //! @return A lane mask with the current lane bit set.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask this_lane() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_eq()};
  }

  //! @brief Returns a lane mask object with all lanes less than the current lane set.
  //!
  //! This function is equivalent to constructing a lane_mask object with value of %%lanemask_lt PTX special register.
  //!
  //! @return A lane mask with all lanes less than the current lane set.
  //!
  //! @note This function may return a mask with 1s set even on inactive lane bits,
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_less() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_lt()};
  }

  //! @brief Returns a lane mask object with all lanes equal to or less than the current lane set.
  //!
  //! This function is equivalent to constructing a lane_mask object with value of %%lanemask_le PTX special register.
  //!
  //! @return A lane mask with all lanes equal to or less than the current lane set.
  //!
  //! @note This function may return a mask with 1s set even on inactive lane bits,
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_less_equal() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_le()};
  }

  //! @brief Returns a lane mask object with all lanes greater than the current lane set.
  //!
  //! This function is equivalent to constructing a lane_mask object with value of %%lanemask_gt PTX special register.
  //!
  //! @return A lane mask with all lanes greater than the current lane set.
  //!
  //! @note This function may return a mask with 1s set even on inactive lane bits,
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_greater() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_gt()};
  }

  //! @brief Returns a lane mask object with all lanes greater than or equal to the current lane set.
  //!
  //! This function is equivalent to constructing a lane_mask object with value of %%lanemask_ge PTX special register.
  //!
  //! @return A lane mask with all lanes greater than or equal to the current lane set.
  //!
  //! @note This function may return a mask with 1s set even on inactive lane bits,
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_greater_equal() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_ge()};
  }

  //! @brief Returns a lane mask object with all lanes not equal to the current lane set.
  //!
  //! This function is equivalent to constructing a lane_mask object with a negated value of %%lanemask_eq PTX special
  //! register.
  //!
  //! @return A lane mask with all lanes not equal to the current lane set.
  //!
  //! @note This function may return a mask with 1s set even on inactive lane bits.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_not_equal() noexcept
  {
    return lane_mask{~_CUDA_VPTX::get_sreg_lanemask_eq()};
  }

  //! @brief Bitwise AND operator for lane_mask.
  //!
  //! @param __lhs The left-hand side lane_mask.
  //! @param __rhs The right-hand side lane_mask.
  //!
  //! @return A new lane_mask object representing the bitwise AND of the two lane_masks.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator&(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return lane_mask{__lhs.__value_ & __rhs.__value_};
  }

  //! @brief Bitwise AND assignment operator for lane_mask.
  //!
  //! @param __v The lane_mask to AND with the current lane_mask.
  //!
  //! @return A reference to the current lane_mask after the AND operation.
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator&=(lane_mask __v) noexcept
  {
    return *this = *this & __v;
  }

  //! @brief Bitwise OR operator for lane_mask.
  //!
  //! @param __lhs The left-hand side lane_mask.
  //! @param __rhs The right-hand side lane_mask.
  //!
  //! @return A new lane_mask object representing the bitwise OR of the two lane_masks.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator|(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return lane_mask{__lhs.__value_ | __rhs.__value_};
  }

  //! @brief Bitwise OR assignment operator for lane_mask.
  //!
  //! @param __v The lane_mask to OR with the current lane_mask.
  //!
  //! @return A reference to the current lane_mask after the OR operation.
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator|=(lane_mask __v) noexcept
  {
    return *this = *this | __v;
  }

  //! @brief Bitwise XOR operator for lane_mask.
  //!
  //! @param __lhs The left-hand side lane_mask.
  //! @param __rhs The right-hand side lane_mask.
  //!
  //! @return A new lane_mask object representing the bitwise XOR of the two lane_masks.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator^(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return lane_mask{__lhs.__value_ ^ __rhs.__value_};
  }

  //! @brief Bitwise XOR assignment operator for lane_mask.
  //!
  //! @param __v The lane_mask to XOR with the current lane_mask.
  //!
  //! @return A reference to the current lane_mask after the XOR operation.
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator^=(lane_mask __v) noexcept
  {
    return *this = *this ^ __v;
  }

  //! @brief Left shift operator for lane_mask.
  //!
  //! @param __mask The lane_mask to shift.
  //! @param __shift The number of bits to shift left.
  //!
  //! @return A new lane_mask object representing the left-shifted lane_mask.
  //!
  //! @pre `__shift` must be in the range [0, 32).
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator<<(lane_mask __mask, int __shift) noexcept
  {
    _CCCL_ASSERT(__shift >= 0 && __shift < 32, "shift must be in range [0, 32)");
    return lane_mask{__mask.__value_ << __shift};
  }

  //! @brief Left shift assignment operator for lane_mask.
  //!
  //! @param __shift The number of bits to shift left.
  //!
  //! @return A reference to the current lane_mask after the left shift operation.
  //!
  //! @pre `__shift` must be in the range [0, 32).
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator<<=(int __shift) noexcept
  {
    _CCCL_ASSERT(__shift >= 0 && __shift < 32, "shift must be in range [0, 32)");
    return *this = *this << __shift;
  }

  //! @brief Right shift operator for lane_mask.
  //!
  //! @param __mask The lane_mask to shift.
  //! @param __shift The number of bits to shift right.
  //!
  //! @return A new lane_mask object representing the right-shifted lane_mask.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator>>(lane_mask __mask, int __shift) noexcept
  {
    _CCCL_ASSERT(__shift >= 0 && __shift < 32, "shift must be in range [0, 32)");
    return lane_mask{__mask.__value_ >> __shift};
  }

  //! @brief Right shift assignment operator for lane_mask.
  //!
  //! @param __shift The number of bits to shift right.
  //!
  //! @return A reference to the current lane_mask after the right shift operation.
  //!
  //! @pre `__shift` must be in the range [0, 32).
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator>>=(int __shift) noexcept
  {
    _CCCL_ASSERT(__shift >= 0 && __shift < 32, "shift must be in range [0, 32)");
    return *this = *this >> __shift;
  }

  //! @brief Bitwise NOT operator for lane_mask.
  //!
  //! @param __mask The lane_mask to negate.
  //!
  //! @return A new lane_mask object representing the negated lane_mask.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask operator~(lane_mask __mask) noexcept
  {
    return lane_mask{~__mask.__value_};
  }

  //! @brief Equality operator for lane_mask.
  //!
  //! @param __lhs The left-hand side lane_mask.
  //! @param __rhs The right-hand side lane_mask.
  //!
  //! @return `true` if the two lane_masks are equal, `false` otherwise.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator==(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return __lhs.__value_ == __rhs.__value_;
  }

  //! @brief Inequality operator for lane_mask.
  //!
  //! @param __lhs The left-hand side lane_mask.
  //! @param __rhs The right-hand side lane_mask.
  //!
  //! @return `true` if the two lane_masks are not equal, `false` otherwise.
  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator!=(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___WARP_LANE_MASK_H
