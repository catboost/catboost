/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights meserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/config.cuh>

#include <cub/util_device.cuh>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/cuda/detail/execution_policy.h>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <thrust/system/cuda/error.h>
#  include <thrust/system_error.h>

#  include <cstdio>
#endif // !_CCCL_COMPILER(NVRTC)

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

inline _CCCL_HOST_DEVICE cudaStream_t default_stream()
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return cudaStreamPerThread;
#else
  return cudaStreamLegacy;
#endif
}

// Fallback implementation of the customization point.
template <class Derived>
_CCCL_HOST_DEVICE cudaStream_t get_stream(execution_policy<Derived>&)
{
  return default_stream();
}

// Entry point/interface.
template <class Derived>
_CCCL_HOST_DEVICE cudaStream_t stream(execution_policy<Derived>& policy)
{
  return get_stream(derived_cast(policy));
}

// Fallback implementation of the customization point.
template <class Derived>
_CCCL_HOST_DEVICE bool must_perform_optional_stream_synchronization(execution_policy<Derived>&)
{
  return true;
}

// Entry point/interface.
template <class Derived>
_CCCL_HOST_DEVICE bool must_perform_optional_synchronization(execution_policy<Derived>& policy)
{
  return must_perform_optional_stream_synchronization(derived_cast(policy));
}

// Fallback implementation of the customization point.
_CCCL_EXEC_CHECK_DISABLE
template <class Derived>
_CCCL_HOST_DEVICE cudaError_t synchronize_stream(execution_policy<Derived>& policy)
{
  return cub::SyncStream(stream(policy));
}

// Entry point/interface.
template <class Policy>
_CCCL_HOST_DEVICE cudaError_t synchronize(Policy& policy)
{
  return synchronize_stream(derived_cast(policy));
}

// Fallback implementation of the customization point.
_CCCL_EXEC_CHECK_DISABLE
template <class Derived>
_CCCL_HOST_DEVICE cudaError_t synchronize_stream_optional(execution_policy<Derived>& policy)
{
  cudaError_t result;

  if (must_perform_optional_synchronization(policy))
  {
    result = synchronize_stream(policy);
  }
  else
  {
    result = cudaSuccess;
  }

  return result;
}

// Entry point/interface.
template <class Policy>
_CCCL_HOST_DEVICE cudaError_t synchronize_optional(Policy& policy)
{
  return synchronize_stream_optional(derived_cast(policy));
}

#if !_CCCL_COMPILER(NVRTC)
template <class Type>
THRUST_HOST_FUNCTION cudaError_t trivial_copy_from_device(Type* dst, Type const* src, size_t count, cudaStream_t stream)
{
  cudaError status = cudaSuccess;
  if (count == 0)
  {
    return status;
  }

  status = ::cudaMemcpyAsync(dst, src, sizeof(Type) * count, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return status;
}

template <class Type>
THRUST_HOST_FUNCTION cudaError_t trivial_copy_to_device(Type* dst, Type const* src, size_t count, cudaStream_t stream)
{
  cudaError status = cudaSuccess;
  if (count == 0)
  {
    return status;
  }

  status = ::cudaMemcpyAsync(dst, src, sizeof(Type) * count, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  return status;
}

template <class Policy, class Type>
THRUST_RUNTIME_FUNCTION cudaError_t
trivial_copy_device_to_device(Policy& policy, Type* dst, Type const* src, size_t count)
{
  cudaError_t status = cudaSuccess;
  if (count == 0)
  {
    return status;
  }

  cudaStream_t stream = cuda_cub::stream(policy);
  //
  status = ::cudaMemcpyAsync(dst, src, sizeof(Type) * count, cudaMemcpyDeviceToDevice, stream);
  cuda_cub::synchronize_optional(policy);
  return status;
}
#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_HOST_DEVICE inline void throw_on_error(cudaError_t status)
{
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
#ifdef THRUST_RDC_ENABLED
  cudaGetLastError();
#else
  NV_IF_TARGET(NV_IS_HOST, (cudaGetLastError();));
#endif

  if (cudaSuccess != status)
  {
    // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
    // instructions out of the target logic.
#ifdef THRUST_RDC_ENABLED

#  define THRUST_TEMP_DEVICE_CODE \
    printf("Thrust CUDA backend error: %s: %s\n", cudaGetErrorName(status), cudaGetErrorString(status))

#else

#  define THRUST_TEMP_DEVICE_CODE printf("Thrust CUDA backend error: %d\n", static_cast<int>(status))

#endif

    NV_IF_TARGET(NV_IS_HOST,
                 (throw thrust::system_error(status, thrust::cuda_category());),
                 (THRUST_TEMP_DEVICE_CODE; ::cuda::std::terminate();));

#undef THRUST_TEMP_DEVICE_CODE
  }
}

_CCCL_HOST_DEVICE inline void throw_on_error(cudaError_t status, char const* msg)
{
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
#ifdef THRUST_RDC_ENABLED
  cudaGetLastError();
#else
  NV_IF_TARGET(NV_IS_HOST, (cudaGetLastError();));
#endif

  if (cudaSuccess != status)
  {
    // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
    // instructions out of the target logic.
#ifdef THRUST_RDC_ENABLED

#  define THRUST_TEMP_DEVICE_CODE \
    printf("Thrust CUDA backend error: %s: %s: %s\n", cudaGetErrorName(status), cudaGetErrorString(status), msg)

#else

#  define THRUST_TEMP_DEVICE_CODE printf("Thrust CUDA backend error: %d: %s\n", static_cast<int>(status), msg)

#endif

    NV_IF_TARGET(NV_IS_HOST,
                 (throw thrust::system_error(status, thrust::cuda_category(), msg);),
                 (THRUST_TEMP_DEVICE_CODE; ::cuda::std::terminate();));

#undef THRUST_TEMP_DEVICE_CODE
  }
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
