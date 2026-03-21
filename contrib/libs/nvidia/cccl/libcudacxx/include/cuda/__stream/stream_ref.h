//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___STREAM_STREAM_REF
#define _CUDA___STREAM_STREAM_REF

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
#  include <cuda/__event/timed_event.h>
#  include <cuda/__fwd/get_stream.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__utility/no_init.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__utility/to_underlying.h>
#  include <cuda/std/cstddef>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

namespace __detail
{
// 0 is a valid stream in CUDA, so we need some other invalid stream representation
// Can't make it constexpr, because cudaStream_t is a pointer type
static const ::cudaStream_t __invalid_stream = reinterpret_cast<::cudaStream_t>(~0ULL);
} // namespace __detail

//! @brief A type representing a stream ID.
enum class stream_id : unsigned long long
{
};

//! @brief A non-owning wrapper for a `cudaStream_t`.
class stream_ref
{
protected:
  ::cudaStream_t __stream{0};

public:
  using value_type = ::cudaStream_t;

  //! @brief Constructs a `stream_ref` of the "default" CUDA stream.
  //!
  //! For behavior of the default stream,
  //! @see //! https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
  CCCL_DEPRECATED_BECAUSE("Using the default/null stream is generally discouraged. If you need to use it, please "
                          "construct a "
                          "stream_ref from cudaStream_t{nullptr}")
  _CCCL_HIDE_FROM_ABI stream_ref() = default;

  //! @brief Constructs a `stream_ref` from a `cudaStream_t` handle.
  //!
  //! This constructor provides implicit conversion from `cudaStream_t`.
  //!
  //! @note: It is the callers responsibility to ensure the `stream_ref` does not
  //! outlive the stream identified by the `cudaStream_t` handle.
  _CCCL_API constexpr stream_ref(value_type __stream_) noexcept
      : __stream{__stream_}
  {}

  //! Disallow construction from an `int`, e.g., `0`.
  stream_ref(int) = delete;

  //! Disallow construction from `nullptr`.
  stream_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Compares two `stream_ref`s for equality
  //!
  //! @note Allows comparison with `cudaStream_t` due to implicit conversion to
  //! `stream_ref`.
  //!
  //! @param lhs The first `stream_ref` to compare
  //! @param rhs The second `stream_ref` to compare
  //! @return true if equal, false if unequal
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const stream_ref& __lhs, const stream_ref& __rhs) noexcept
  {
    return __lhs.__stream == __rhs.__stream;
  }

  //! @brief Compares two `stream_ref`s for inequality
  //!
  //! @note Allows comparison with `cudaStream_t` due to implicit conversion to
  //! `stream_ref`.
  //!
  //! @param lhs The first `stream_ref` to compare
  //! @param rhs The second `stream_ref` to compare
  //! @return true if unequal, false if equal
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const stream_ref& __lhs, const stream_ref& __rhs) noexcept
  {
    return __lhs.__stream != __rhs.__stream;
  }

  //! Returns the wrapped `cudaStream_t` handle.
  [[nodiscard]] _CCCL_API constexpr value_type get() const noexcept
  {
    return __stream;
  }

  //! @brief Synchronizes the wrapped stream.
  //!
  //! @throws cuda::cuda_error if synchronization fails.
  _CCCL_HOST_API void sync() const
  {
    _CUDA_DRIVER::__streamSynchronize(__stream);
  }

  //! @brief Deprecated. Use sync() instead.
  //!
  //! @deprecated Use sync() instead.
  CCCL_DEPRECATED_BECAUSE("Use sync() instead.") _CCCL_HOST_API void wait() const
  {
    sync();
  }

  //! @brief Make all future work submitted into this stream depend on completion of the specified event
  //!
  //! @param __ev Event that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  _CCCL_HOST_API void wait(event_ref __ev) const
  {
    _CCCL_ASSERT(__ev.get() != nullptr, "cuda::stream_ref::wait invalid event passed");
    // Need to use driver API, cudaStreamWaitEvent would push dev 0 if stack was empty
    _CUDA_DRIVER::__streamWaitEvent(get(), __ev.get());
  }

  //! @brief Make all future work submitted into this stream depend on completion of all work from the specified
  //! stream
  //!
  //! @param __other Stream that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  _CCCL_HOST_API void wait(stream_ref __other) const
  {
    // TODO consider an optimization to not create an event every time and instead have one persistent event or one
    // per stream
    _CCCL_ASSERT(__stream != __detail::__invalid_stream, "cuda::stream_ref::wait invalid stream passed");
    if (*this != __other)
    {
      event __tmp(__other);
      wait(__tmp);
    }
  }

  //! \brief Queries if all operations on the stream have completed.
  //!
  //! \throws cuda::cuda_error if the query fails.
  //!
  //! \return `true` if all operations have completed, or `false` if not.
  [[nodiscard]] _CCCL_HOST_API bool is_done() const
  {
    const auto __result = _CUDA_DRIVER::__streamQueryNoThrow(__stream);
    switch (__result)
    {
      case ::cudaErrorNotReady:
        return false;
      case ::cudaSuccess:
        return true;
      default:
        ::cuda::__throw_cuda_error(__result, "Failed to query stream.");
    }
  }

  //! @brief Queries if all operations on the wrapped stream have completed.
  //!
  //! @throws cuda::cuda_error if the query fails.
  //!
  //! @return `true` if all operations have completed, or `false` if not.
  [[nodiscard]] CCCL_DEPRECATED_BECAUSE("Use is_done() instead.") _CCCL_HOST_API bool ready() const
  {
    return is_done();
  }

  //! @brief Queries the priority of the wrapped stream.
  //!
  //! @throws cuda::cuda_error if the query fails.
  //!
  //! @return value representing the priority of the wrapped stream.
  [[nodiscard]] _CCCL_HOST_API int priority() const
  {
    return _CUDA_DRIVER::__streamGetPriority(__stream);
  }

  //! @brief Get the unique ID of the stream
  //!
  //! Stream handles are sometimes reused, but ID is guaranteed to be unique.
  //!
  //! @return The unique ID of the stream
  //!
  //! @throws cuda_error if the ID query fails
  [[nodiscard]] _CCCL_HOST_API stream_id id() const
  {
    return stream_id{_CUDA_DRIVER::__streamGetId(__stream)};
  }

  //! @brief Create a new event and record it into this stream
  //!
  //! @return A new event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  [[nodiscard]] _CCCL_HOST_API event record_event(event_flags __flags = event_flags::none) const
  {
    return event(*this, __flags);
  }

  //! @brief Create a new timed event and record it into this stream
  //!
  //! @return A new timed event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  [[nodiscard]] _CCCL_HOST_API timed_event record_timed_event(event_flags __flags = event_flags::none) const
  {
    return timed_event(*this, __flags);
  }

  //! @brief Get device under which this stream was created.
  //!
  //! Note: In case of a stream created under a `green_context` the device on which that `green_context` was created is
  //! returned
  //!
  //! @throws cuda_error if device check fails
  [[nodiscard]] _CCCL_HOST_API device_ref device() const
  {
    ::CUdevice __device{};
#  if _CCCL_CTK_AT_LEAST(13, 0)
    __device = ::cuda::__driver::__streamGetDevice(__stream);
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
    {
      ::CUcontext __stream_ctx = ::cuda::__driver::__streamGetCtx(__stream);
      __ensure_current_context __setter(__stream_ctx);
      __device = ::cuda::__driver::__ctxGetDevice();
    }
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
    return device_ref{::cuda::__driver::__cudevice_to_ordinal(__device)};
  }

  //! @brief Queries the \c stream_ref for itself. This makes \c stream_ref usable in places where we expect an
  //! environment with a \c get_stream_t query
  [[nodiscard]] _CCCL_API constexpr stream_ref query(const ::cuda::get_stream_t&) const noexcept
  {
    return *this;
  }
};

_CCCL_HOST_API inline void event_ref::record(stream_ref __stream) const
{
  _CCCL_ASSERT(__event_ != nullptr, "cuda::event_ref::record no event set");
  _CCCL_ASSERT(__stream.get() != nullptr, "cuda::event_ref::record invalid stream passed");
  // Need to use driver API, cudaEventRecord will push dev 0 if stack is empty
  _CUDA_DRIVER::__eventRecord(__event_, __stream.get());
}

_CCCL_HOST_API inline event::event(stream_ref __stream, event_flags __flags)
    : event(__stream, ::cuda::std::to_underlying(__flags) | cudaEventDisableTiming)
{
  record(__stream);
}

_CCCL_HOST_API inline event::event(stream_ref __stream, unsigned __flags)
    : event_ref(::cudaEvent_t{})
{
  [[maybe_unused]] __ensure_current_context __ctx_setter(__stream);
  __event_ = ::cuda::__driver::__eventCreate(static_cast<unsigned>(__flags));
}

_CCCL_HOST_API inline timed_event::timed_event(stream_ref __stream, event_flags __flags)
    : event(__stream, ::cuda::std::to_underlying(__flags))
{
  record(__stream);
}

_CCCL_HOST_API inline __ensure_current_context::__ensure_current_context(stream_ref __stream)
{
  auto __ctx = __driver::__streamGetCtx(__stream.get());
  _CUDA_DRIVER::__ctxPush(__ctx);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif //_CUDA___STREAM_STREAM_REF
