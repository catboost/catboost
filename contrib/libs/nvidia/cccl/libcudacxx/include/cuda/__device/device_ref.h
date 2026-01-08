//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_DEVICE_REF_H
#define _CUDA___DEVICE_DEVICE_REF_H

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
#  include <cuda/__fwd/devices.h>
#  include <cuda/std/span>
#  include <cuda/std/string_view>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief A non-owning representation of a CUDA device
class device_ref
{
  int __id_ = 0;

public:
  //! @brief Create a `device_ref` object from a native device ordinal.
  /*implicit*/ _CCCL_HOST_API constexpr device_ref(int __id) noexcept
      : __id_(__id)
  {}

  //! @brief Retrieve the native ordinal of the `device_ref`
  //!
  //! @return int The native device ordinal held by the `device_ref` object
  [[nodiscard]] _CCCL_HOST_API constexpr int get() const noexcept
  {
    return __id_;
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  //! @brief Compares two `device_ref`s for equality
  //!
  //! @note Allows comparison with `int` due to implicit conversion to
  //! `device_ref`.
  //!
  //! @param __lhs The first `device_ref` to compare
  //! @param __rhs The second `device_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same device ordinal
  [[nodiscard]] friend _CCCL_HOST_API constexpr bool operator==(device_ref __lhs, device_ref __rhs) noexcept
  {
    return __lhs.__id_ == __rhs.__id_;
  }

#    if _CCCL_STD_VER <= 2017
  //! @brief Compares two `device_ref`s for inequality
  //!
  //! @note Allows comparison with `int` due to implicit conversion to
  //! `device_ref`.
  //!
  //! @param __lhs The first `device_ref` to compare
  //! @param __rhs The second `device_ref` to compare
  //! @return true if `lhs` and `rhs` refer to different device ordinal
  [[nodiscard]] friend _CCCL_HOST_API constexpr bool operator!=(device_ref __lhs, device_ref __rhs) noexcept
  {
    return __lhs.__id_ != __rhs.__id_;
  }
#    endif // _CCCL_STD_VER <= 2017
#  endif // _CCCL_DOXYGEN_INVOKED

  //! @brief Retrieve the specified attribute for the device
  //!
  //! @param __attr The attribute to query. See `device::attrs` for the available
  //!        attributes.
  //!
  //! @throws cuda_error if the attribute query fails
  //!
  //! @sa device::attrs
  template <typename _Attr>
  [[nodiscard]] _CCCL_HOST_API auto attribute(_Attr __attr) const
  {
    return __attr(*this);
  }

  //! @overload
  template <::cudaDeviceAttr _Attr>
  [[nodiscard]] _CCCL_HOST_API auto attribute() const
  {
    return attribute(__dev_attr<_Attr>());
  }

  //! @brief Initializes the primary context of the device.
  _CCCL_HOST_API void init() const; // implemented in <cuda/__device/physical_device.h> to avoid circular dependency

  //! @brief Retrieve the name of this device.
  //!
  //! @return String view containing the name of this device.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::string_view name() const; // implemented in
                                                                      // <cuda/__device/physical_device.h> to avoid
                                                                      // circular dependency

  //! @brief Queries if its possible for this device to directly access specified device's memory.
  //!
  //! If this function returns true, device supplied to this call can be passed into enable_peer_access
  //! on memory resource or pool that manages memory on this device. It will make allocations from that
  //! pool accessible by this device.
  //!
  //! @param __other_dev Device to query the peer access
  //! @return true if its possible for this device to access the specified device's memory
  [[nodiscard]] _CCCL_HOST_API bool has_peer_access_to(device_ref __other_dev) const
  {
    return ::cuda::__driver::__deviceCanAccessPeer(
      ::cuda::__driver::__deviceGet(get()), ::cuda::__driver::__deviceGet(__other_dev.get()));
  }

  // TODO this might return some more complex type in the future
  // TODO we might want to include the calling device, depends on what we decide
  // peer access APIs

  //! @brief Retrieve `device_ref`s that are peers of this device
  //!
  //! The device on which this API is called is not included in the vector.
  //!
  //! @throws cuda_error if any peer access query fails
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<const device_ref> peers() const; // implemented in
                                                                                  // <cuda/__device/physical_device.h>
                                                                                  // to avoid circular dependency
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_DEVICE_REF_H
