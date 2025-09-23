/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
#include <thrust/system/cuda/detail/core/alignment.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <cassert>


THRUST_NAMESPACE_BEGIN

namespace cuda_cub {
namespace launcher {

  struct triple_chevron
  {
    typedef size_t Size;
    dim3 const grid;
    dim3 const block;
    Size const shared_mem;
    cudaStream_t const stream;

    THRUST_RUNTIME_FUNCTION
    triple_chevron(dim3         grid_,
                   dim3         block_,
                   Size         shared_mem_ = 0,
                   cudaStream_t stream_     = 0)
        : grid(grid_),
          block(block_),
          shared_mem(shared_mem_),
          stream(stream_) {}

    template<class K, class... Args>
    cudaError_t __host__
    doit_host(K k, Args const&... args) const
    {
      k<<<grid, block, shared_mem, stream>>>(args...);
      return cudaPeekAtLastError();
    }

    template<class T>
    size_t __device__
    align_up(size_t offset) const
    {
      size_t alignment = alignment_of<T>::value;
      return alignment * ((offset + (alignment - 1))/ alignment);
    }

    size_t __device__ argument_pack_size(size_t size) const { return size; }
    template <class Arg, class... Args>
    size_t __device__
    argument_pack_size(size_t size, Arg const& arg, Args const&... args) const
    {
      size = align_up<Arg>(size);
      return argument_pack_size(size + sizeof(Arg), args...);
    }

    template <class Arg>
    size_t __device__ copy_arg(char* buffer, size_t offset, Arg arg) const
    {
      offset = align_up<Arg>(offset);
      for (int i = 0; i != sizeof(Arg); ++i)
        buffer[offset+i] = *((char*)&arg + i);
      return offset + sizeof(Arg);
    }

    __device__
    void fill_arguments(char*, size_t) const
    {}

    template<class Arg, class... Args>
    __device__
    void fill_arguments(char* buffer,
                     size_t offset,
                     Arg const& arg,
                     Args const& ... args) const
    {
      fill_arguments(buffer, copy_arg(buffer, offset, arg), args...);
    }

    template<class K, class... Args>
    cudaError_t __device__
    doit_device(K k, Args const&... args) const
    {
      const size_t size = argument_pack_size(0,args...);
      void *param_buffer = cudaGetParameterBuffer(64,size);
      fill_arguments((char*)param_buffer, 0, args...);
      return launch_device(k, param_buffer);
    }

    template <class K>
    cudaError_t __device__
    launch_device(K k, void* buffer) const
    {
      return cudaLaunchDevice((void*)k,
                              buffer,
                              dim3(grid),
                              dim3(block),
                              shared_mem,
                              stream);
    }

    __thrust_exec_check_disable__
    template <class K, class... Args>
    THRUST_FUNCTION
    cudaError_t doit(K k, Args const&... args) const
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return doit_host(k, args...);),
                   (return doit_device(k, args...);));
    }

  }; // struct triple_chevron

}    // namespace launcher
}    // namespace cuda_

THRUST_NAMESPACE_END
