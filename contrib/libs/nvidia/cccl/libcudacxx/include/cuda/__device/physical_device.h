//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_PHYSICAL_DEVICE_H
#define _CUDA___DEVICE_PHYSICAL_DEVICE_H

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
#  include <cuda/__fwd/devices.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/span>
#  include <cuda/std/string_view>

#  include <cassert>
#  include <memory>
#  include <mutex>
#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

[[nodiscard]] inline ::cuda::std::span<__physical_device> __physical_devices();

// This is the element type of the the global `devices` array. In the future, we
// can cache device properties here.
//
//! @brief An immovable "owning" representation of a CUDA device.
class __physical_device
{
  friend _CCCL_HOST_API inline ::std::unique_ptr<__physical_device[]>
  __make_physical_devices(::cuda::std::size_t __device_count);

  ::CUdevice __device_{};

  ::std::once_flag __primary_ctx_once_flag_{};
  ::CUcontext __primary_ctx_{};

  static constexpr ::cuda::std::size_t __max_name_length{256};
  ::std::once_flag __name_once_flag_{};
  char __name_[__max_name_length]{};
  ::cuda::std::size_t __name_length_{};

  ::std::once_flag __peers_once_flag_{};
  ::std::vector<device_ref> __peers_{};

public:
  _CCCL_HIDE_FROM_ABI __physical_device() = default;

  _CCCL_HOST_API ~__physical_device()
  {
    if (__primary_ctx_ != nullptr)
    {
      [[maybe_unused]] const auto __ignore = ::cuda::__driver::__primaryCtxReleaseNoThrow(__device_);
    }
  }

  //! @brief Retrieve the primary context for this device.
  //!
  //! @return A reference to the primary context for this device.
  [[nodiscard]] _CCCL_HOST_API ::CUcontext __primary_context()
  {
    ::std::call_once(__primary_ctx_once_flag_, [this]() {
      __primary_ctx_ = ::cuda::__driver::__primaryCtxRetain(__device_);
    });
    return __primary_ctx_;
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::string_view __name()
  {
    ::std::call_once(__name_once_flag_, [this]() {
      const auto __id = ::cuda::__driver::__cudevice_to_ordinal(__device_);
      ::cuda::__driver::__deviceGetName(__name_, __max_name_length, __id);
      __name_length_ = ::cuda::std::char_traits<char>::length(__name_);
    });
    return ::cuda::std::string_view{__name_, __name_length_};
  }

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<const device_ref> __peers()
  {
    ::std::call_once(__peers_once_flag_, [this]() {
      const auto __count = static_cast<int>(::cuda::__physical_devices().size());
      const auto __id    = ::cuda::__driver::__cudevice_to_ordinal(__device_);
      __peers_.reserve(__count);
      for (int __other_id = 0; __other_id < __count; ++__other_id)
      {
        // Exclude the device this API is called on. The main use case for this API
        // is enable/disable peer access. While enable peer access can be called on
        // device on which memory resides, disable peer access will error-out.
        // Usage of the peer access control is smoother when *this is excluded,
        // while it can be easily added with .push_back() on the vector if a full
        // group of peers is needed (for cases other than peer access control)
        if (__other_id != __id)
        {
          device_ref __dev{__id};
          device_ref __other_dev{__other_id};

          // While in almost all practical applications peer access should be symmetrical,
          // it is possible to build a system with one directional peer access, check
          // both ways here just to be safe
          if (__dev.has_peer_access_to(__other_dev) && __other_dev.has_peer_access_to(__dev))
          {
            __peers_.push_back(__other_dev);
          }
        }
      }
    });
    return ::cuda::std::span<const device_ref>{__peers_};
  }
};

[[nodiscard]] _CCCL_HOST_API inline ::std::unique_ptr<__physical_device[]>
__make_physical_devices(::cuda::std::size_t __device_count)
{
  ::std::unique_ptr<__physical_device[]> __devices{::new __physical_device[__device_count]};
  for (::cuda::std::size_t __i = 0; __i < __device_count; ++__i)
  {
    __devices[__i].__device_ = static_cast<int>(__i);
  }
  return __devices;
}

[[nodiscard]] inline ::cuda::std::span<__physical_device> __physical_devices()
{
  static const auto __device_count = static_cast<::cuda::std::size_t>(::cuda::__driver::__deviceGetCount());
  static const auto __devices      = ::cuda::__make_physical_devices(__device_count);
  return ::cuda::std::span<__physical_device>{__devices.get(), __device_count};
}

// device_ref methods dependent on __physical_device

_CCCL_HOST_API inline void device_ref::init() const
{
  (void) ::cuda::__physical_devices()[__id_].__primary_context();
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::string_view device_ref::name() const
{
  return ::cuda::__physical_devices()[__id_].__name();
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::span<const device_ref> device_ref::peers() const
{
  return ::cuda::__physical_devices()[__id_].__peers();
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_PHYSICAL_DEVICE_H
