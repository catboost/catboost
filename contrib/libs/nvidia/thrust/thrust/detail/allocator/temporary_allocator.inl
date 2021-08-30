/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/detail/config.h>
#include <thrust/detail/allocator/temporary_allocator.h>
#include <thrust/detail/temporary_buffer.h>
#include <thrust/system/detail/bad_alloc.h>
#include <cassert>

#if (defined(__NVCOMPILER_CUDA__) || defined(__CUDA_ARCH__)) && \
    THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <thrust/system/cuda/detail/terminate.h>
#endif

namespace thrust
{
namespace detail
{


template<typename T, typename System>
__host__ __device__
  typename temporary_allocator<T,System>::pointer
    temporary_allocator<T,System>
      ::allocate(typename temporary_allocator<T,System>::size_type cnt)
{
  pointer_and_size result = thrust::get_temporary_buffer<T>(system(), cnt);

  // handle failure
  if(result.second < cnt)
  {
    // deallocate and throw
    // note that we pass cnt to deallocate, not a value derived from result.second
    deallocate(result.first, cnt);

    if (THRUST_IS_HOST_CODE) {
      #if THRUST_INCLUDE_HOST_CODE
        throw thrust::system::detail::bad_alloc("temporary_buffer::allocate: get_temporary_buffer failed");
      #endif
    } else {
      #if THRUST_INCLUDE_DEVICE_CODE && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        thrust::system::cuda::detail::terminate_with_message("temporary_buffer::allocate: get_temporary_buffer failed");
      #endif
    }
  } // end if

  return result.first;
} // end temporary_allocator::allocate()


template<typename T, typename System>
__host__ __device__
  void temporary_allocator<T,System>
    ::deallocate(typename temporary_allocator<T,System>::pointer p, typename temporary_allocator<T,System>::size_type n)
{
  return thrust::return_temporary_buffer(system(), p, n);
} // end temporary_allocator


} // end detail
} // end thrust

