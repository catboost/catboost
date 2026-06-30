//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EVENT_EVENT_REF_H
#define _CUDA___EVENT_EVENT_REF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/std/cassert>
#  include <cuda/std/cstddef>
#  include <cuda/std/utility>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

class event;
class timed_event;
class stream_ref;

//! @brief An non-owning wrapper for an untimed `cudaEvent_t`.
class event_ref
{
private:
  friend class event;
  friend class timed_event;

  ::cudaEvent_t __event_{};

public:
  using value_type = ::cudaEvent_t;

  //! @brief Construct a new `event_ref` object from a `cudaEvent_t`
  //!
  //! This constructor provides an implicit conversion from `cudaEvent_t`
  //!
  //! @post `get() == __evnt`
  //!
  //! @note: It is the callers responsibility to ensure the `event_ref` does not
  //! outlive the event denoted by the `cudaEvent_t` handle.
  _CCCL_HOST_API constexpr event_ref(::cudaEvent_t __evnt) noexcept
      : __event_(__evnt)
  {}

  /// Disallow construction from an `int`, e.g., `0`.
  event_ref(int) = delete;

  /// Disallow construction from `nullptr`.
  event_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Records an event on the specified stream
  //!
  //! @param __stream
  //!
  //! @throws cuda_error if the event record fails
  _CCCL_HOST_API void record(stream_ref __stream) const;

  //! @brief Synchronizes the event
  //!
  //! @throws cuda_error if waiting for the event fails
  _CCCL_HOST_API void sync() const
  {
    _CCCL_ASSERT(__event_ != nullptr, "cuda::event_ref::sync no event set");
    ::cuda::__driver::__eventSynchronize(__event_);
  }

  //! @brief Checks if all the work in the stream prior to the record of the event has completed.
  //!
  //! If is_done returns true, calling sync() on this event will return immediately
  //!
  //! @throws cuda_error if the event query fails
  [[nodiscard]] _CCCL_HOST_API bool is_done() const
  {
    _CCCL_ASSERT(__event_ != nullptr, "cuda::event_ref::sync no event set");
    ::cudaError_t __status = ::cuda::__driver::__eventQueryNoThrow(__event_);
    if (__status == ::cudaSuccess)
    {
      return true;
    }
    else if (__status == ::cudaErrorNotReady)
    {
      return false;
    }
    else
    {
      ::cuda::__throw_cuda_error(__status, "Failed to query CUDA event");
    }
  }

  //! @brief Retrieve the native `cudaEvent_t` handle.
  //!
  //! @return cudaEvent_t The native handle being held by the event_ref object.
  [[nodiscard]] _CCCL_HOST_API constexpr ::cudaEvent_t get() const noexcept
  {
    return __event_;
  }

  //! @brief Checks if the `event_ref` is valid
  //!
  //! @return true if the `event_ref` is valid, false otherwise.
  [[nodiscard]] _CCCL_HOST_API explicit constexpr operator bool() const noexcept
  {
    return __event_ != nullptr;
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  //! @brief Compares two `event_ref`s for equality
  //!
  //! @note Allows comparison with `cudaEvent_t` due to implicit conversion to
  //! `event_ref`.
  //!
  //! @param __lhs The first `event_ref` to compare
  //! @param __rhs The second `event_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same `cudaEvent_t` object.
  [[nodiscard]] friend _CCCL_HOST_API constexpr bool operator==(event_ref __lhs, event_ref __rhs) noexcept
  {
    return __lhs.__event_ == __rhs.__event_;
  }

  //! @brief Compares two `event_ref`s for inequality
  //!
  //! @note Allows comparison with `cudaEvent_t` due to implicit conversion to
  //! `event_ref`.
  //!
  //! @param __lhs The first `event_ref` to compare
  //! @param __rhs The second `event_ref` to compare
  //! @return true if `lhs` and `rhs` refer to different `cudaEvent_t` objects.
  [[nodiscard]] friend _CCCL_HOST_API constexpr bool operator!=(event_ref __lhs, event_ref __rhs) noexcept
  {
    return __lhs.__event_ != __rhs.__event_;
  }
#  endif // _CCCL_DOXYGEN_INVOKED
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___EVENT_EVENT_REF_H
