// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

namespace detail
{

struct TripleChevronFactory
{
  CUB_RUNTIME_FUNCTION THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron operator()(
    dim3 grid, dim3 block, _CUDA_VSTD::size_t shared_mem, cudaStream_t stream, bool dependent_launch = false) const
  {
    return THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid, block, shared_mem, stream, dependent_launch);
  }

  CUB_RUNTIME_FUNCTION cudaError_t PtxVersion(int& version)
  {
    return cub::PtxVersion(version);
  }

  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION cudaError_t MultiProcessorCount(int& sm_count) const
  {
    int device_ordinal;
    cudaError_t error = CubDebug(cudaGetDevice(&device_ordinal));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get SM count
    return cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal);
  }

  template <typename Kernel>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION cudaError_t
  MaxSmOccupancy(int& sm_occupancy, Kernel kernel_ptr, int block_size, int dynamic_smem_bytes = 0)
  {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, kernel_ptr, block_size, dynamic_smem_bytes);
  }

  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION cudaError_t MaxGridDimX(int& max_grid_dim_x) const
  {
    int device_ordinal;
    cudaError_t error = CubDebug(cudaGetDevice(&device_ordinal));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get max grid dimension
    return cudaDeviceGetAttribute(&max_grid_dim_x, cudaDevAttrMaxGridDimX, device_ordinal);
  }

  // TODO(bgruber): this is very similar to thrust::cuda_cub::core::get_max_shared_memory_per_block. We should unify
  // this.
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION cudaError_t MaxSharedMemory(int& max_shared_memory) const
  {
    int device = 0;
    auto error = CubDebug(cudaGetDevice(&device));
    if (error != cudaSuccess)
    {
      return error;
    }

    return cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, device);
  }
};

} // namespace detail

CUB_NAMESPACE_END
