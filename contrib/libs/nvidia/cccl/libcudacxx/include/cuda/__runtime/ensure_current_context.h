//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___RUNTIME_ENSURE_CURRENT_CONTEXT_H
#define _CUDA___RUNTIME_ENSURE_CURRENT_CONTEXT_H

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
#  include <cuda/__device/physical_device.h>
#  include <cuda/__driver/driver_api.h>

#  include <cuda/std/__cccl/prologue.h>

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

class stream_ref;

//! @brief RAII helper which on construction sets the current context to the specified one.
//! It sets the state back on destruction.
//!
struct [[maybe_unused]] __ensure_current_context
{
  //! @brief Construct a new `__ensure_current_context` object and switch to the primary context of the specified
  //!        device.
  //!
  //! @param new_device The device to switch the context to
  //!
  //! @throws cuda_error if the context switch fails
  _CCCL_HOST_API explicit __ensure_current_context(device_ref __new_device)
  {
    auto __ctx = ::cuda::__physical_devices()[__new_device.get()].__primary_context();
    ::cuda::__driver::__ctxPush(__ctx);
  }

  //! @brief Construct a new `__ensure_current_context` object and switch to the specified
  //!        context.
  //!
  //! @param ctx The context to switch to
  //!
  //! @throws cuda_error if the context switch fails
  _CCCL_HOST_API explicit __ensure_current_context(::CUcontext __ctx)
  {
    _CUDA_DRIVER::__ctxPush(__ctx);
  }

  //! @brief Construct a new `__ensure_current_context` object and switch to the context
  //!        under which the specified stream was created.
  //!
  //! @param stream Stream indicating the context to switch to
  //!
  //! @throws cuda_error if the context switch fails
  _CCCL_HOST_API explicit __ensure_current_context(stream_ref __stream);

  __ensure_current_context(__ensure_current_context&&)                 = delete;
  __ensure_current_context(__ensure_current_context const&)            = delete;
  __ensure_current_context& operator=(__ensure_current_context&&)      = delete;
  __ensure_current_context& operator=(__ensure_current_context const&) = delete;

  //! @brief Destroy the `__ensure_current_context` object and switch back to the original
  //!        context.
  //!
  //! @throws cuda_error if the device switch fails. If the destructor is called
  //!         during stack unwinding, the program is automatically terminated.
  _CCCL_HOST_API ~__ensure_current_context() noexcept(false)
  {
    // TODO would it make sense to assert here that we pushed and popped the same thing?
    _CUDA_DRIVER::__ctxPop();
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  endif // _CCCL_DOXYGEN_INVOKED

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___RUNTIME_ENSURE_CURRENT_CONTEXT_H
