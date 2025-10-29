//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EVENT_EVENT_H
#define _CUDA___EVENT_EVENT_H

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
#  include <cuda/__event/event_ref.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__utility/no_init.h>
#  include <cuda/std/__utility/to_underlying.h>
#  include <cuda/std/cstddef>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

class timed_event;

//! @brief Flags to use when creating the event.
enum class event_flags : unsigned
{
  none          = cudaEventDefault,
  blocking_sync = cudaEventBlockingSync,
  interprocess  = cudaEventInterprocess,
};

[[nodiscard]] _CCCL_HOST_API constexpr event_flags operator|(event_flags __lhs, event_flags __rhs) noexcept
{
  return static_cast<event_flags>(::cuda::std::to_underlying(__lhs) | ::cuda::std::to_underlying(__rhs));
}

//! @brief An owning wrapper for an untimed `cudaEvent_t`.
class event : public event_ref
{
  friend class timed_event;

public:
  //! @brief Construct a new `event` object with timing disabled, and record
  //!        the event in the specified stream.
  //!
  //! @throws cuda_error if the event creation fails.
  _CCCL_HOST_API explicit event(stream_ref __stream, event_flags __flags = event_flags::none);

  //! @brief Construct a new `event` object with timing disabled. The event can only be recorded on streams from the
  //! specified device.
  //!
  //! @throws cuda_error if the event creation fails.
  _CCCL_HOST_API explicit event(device_ref __device, event_flags __flags = event_flags::none)
      : event(__device, ::cuda::std::to_underlying(__flags) | cudaEventDisableTiming)
  {}

  //! @brief Construct a new `event` object into the moved-from state.
  //!
  //! @post `get()` returns `cudaEvent_t()`.
  _CCCL_HOST_API explicit constexpr event(no_init_t) noexcept
      : event_ref(::cudaEvent_t{})
  {}

  //! @brief Move-construct a new `event` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  _CCCL_HOST_API constexpr event(event&& __other) noexcept
      : event_ref(::cuda::std::exchange(__other.__event_, {}))
  {}

  // Disallow copy construction.
  event(const event&) = delete;

  //! @brief Destroy the `event` object
  //!
  //! @note If the event fails to be destroyed, the error is silently ignored.
  _CCCL_HOST_API ~event()
  {
    if (__event_ != nullptr)
    {
      // Needs to call driver API in case current device is not set, runtime version would set dev 0 current
      // Alternative would be to store the device and push/pop here
      [[maybe_unused]] auto __status = _CUDA_DRIVER::__eventDestroyNoThrow(__event_);
    }
  }

  //! @brief Move-assign an `event` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  _CCCL_HOST_API event& operator=(event&& __other) noexcept
  {
    event __tmp(_CUDA_VSTD::move(__other));
    _CUDA_VSTD::swap(__event_, __tmp.__event_);
    return *this;
  }

  // Disallow copy assignment.
  event& operator=(const event&) = delete;

  //! @brief Construct an `event` object from a native `cudaEvent_t` handle.
  //!
  //! @param __evnt The native handle
  //!
  //! @return event The constructed `event` object
  //!
  //! @note The constructed `event` object takes ownership of the native handle.
  [[nodiscard]] static _CCCL_HOST_API event from_native_handle(::cudaEvent_t __evnt) noexcept
  {
    return event(__evnt);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static event from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static event from_native_handle(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Retrieve the native `cudaEvent_t` handle and give up ownership.
  //!
  //! @return cudaEvent_t The native handle being held by the `event` object.
  //!
  //! @post The event object is in a moved-from state.
  [[nodiscard]] _CCCL_HOST_API constexpr ::cudaEvent_t release() noexcept
  {
    return _CUDA_VSTD::exchange(__event_, {});
  }

private:
  // Use `event::from_native_handle(e)` to construct an owning `event`
  // object from a `cudaEvent_t` handle.
  _CCCL_HOST_API explicit constexpr event(::cudaEvent_t __evnt) noexcept
      : event_ref(__evnt)
  {}

  _CCCL_HOST_API explicit event(stream_ref __stream, unsigned __flags);

  _CCCL_HOST_API explicit event(device_ref __device, unsigned __flags)
      : event_ref(::cudaEvent_t{})
  {
    [[maybe_unused]] __ensure_current_context __ctx_setter(__device);
    __event_ = ::cuda::__driver::__eventCreate(static_cast<unsigned>(__flags));
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___EVENT_EVENT_H
