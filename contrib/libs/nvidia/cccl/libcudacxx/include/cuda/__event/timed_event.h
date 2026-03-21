//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EVENT_TIMED_EVENT_H
#define _CUDA___EVENT_TIMED_EVENT_H

#include <cuda_runtime_api.h>
// cuda_runtime_api needs to come first

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__device/device_ref.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__event/event.h>
#  include <cuda/__utility/no_init.h>
#  include <cuda/std/__utility/to_underlying.h>
#  include <cuda/std/chrono>
#  include <cuda/std/cstddef>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief An owning wrapper for a `cudaEvent_t` with timing enabled.
class timed_event : public event
{
public:
  //! @brief Construct a new `timed_event` object with the specified flags
  //!        and record the event on the specified stream.
  //!
  //! @throws cuda_error if the event creation fails.
  _CCCL_HOST_API explicit timed_event(stream_ref __stream, event_flags __flags = event_flags::none);

  //! @brief Construct a new `timed_event` object with the specified flags. The event can only be recorded on streams
  //! from the specified device.
  //!
  //! @throws cuda_error if the event creation fails.
  _CCCL_HOST_API explicit timed_event(device_ref __device, event_flags __flags = event_flags::none)
      : event(__device, ::cuda::std::to_underlying(__flags))
  {}

  //! @brief Construct a new `timed_event` object into the moved-from state.
  //!
  //! @post `get()` returns `cudaEvent_t()`.
  _CCCL_HOST_API explicit constexpr timed_event(no_init_t) noexcept
      : event(no_init)
  {}

  timed_event(timed_event&&) noexcept            = default;
  timed_event(const timed_event&)                = delete;
  timed_event& operator=(timed_event&&) noexcept = default;
  timed_event& operator=(const timed_event&)     = delete;

  //! @brief Construct a `timed_event` object from a native `cudaEvent_t` handle.
  //!
  //! @param __evnt The native handle
  //!
  //! @return timed_event The constructed `timed_event` object
  //!
  //! @note The constructed `timed_event` object takes ownership of the native handle.
  [[nodiscard]] static _CCCL_HOST_API timed_event from_native_handle(::cudaEvent_t __evnt) noexcept
  {
    return timed_event(__evnt);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static timed_event from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static timed_event from_native_handle(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Compute the time elapsed between two `timed_event` objects.
  //!
  //! @throws cuda_error if the query for the elapsed time fails.
  //!
  //! @param __end The `timed_event` object representing the end time.
  //! @param __start The `timed_event` object representing the start time.
  //!
  //! @return cuda::std::chrono::nanoseconds The elapsed time in nanoseconds.
  //!
  //! @note The elapsed time has a resolution of approximately 0.5 microseconds.
  [[nodiscard]] friend _CCCL_HOST_API ::cuda::std::chrono::nanoseconds
  operator-(const timed_event& __end, const timed_event& __start)
  {
    const auto __ms = ::cuda::__driver::__eventElapsedTime(__start.get(), __end.get());
    return ::cuda::std::chrono::nanoseconds(static_cast<::cuda::std::chrono::nanoseconds::rep>(__ms * 1'000'000.0));
  }

private:
  // Use `timed_event::from_native_handle(e)` to construct an owning `timed_event`
  // object from a `cudaEvent_t` handle.
  _CCCL_HOST_API explicit constexpr timed_event(::cudaEvent_t __evnt) noexcept
      : event(__evnt)
  {}
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___EVENT_TIMED_EVENT_H
