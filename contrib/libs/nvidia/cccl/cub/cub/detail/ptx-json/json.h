/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/detail/ptx-json/array.h>
#include <cub/detail/ptx-json/object.h>
#include <cub/detail/ptx-json/string.h>
#include <cub/detail/ptx-json/value.h>

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace ptx_json
{
template <auto T, typename = value_traits<T>::type>
struct tagged_json;

template <int N, string<N> T, cuda::std::size_t... Is>
struct tagged_json<T, cuda::std::index_sequence<Is...>>
{
  template <typename V, typename = cuda::std::enable_if_t<is_object<V>::value || is_array<V>::value>>
  __noinline__ __device__ void operator=(V)
  {
    asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
    V::emit();
    asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
  }
};

template <auto T>
__forceinline__ __device__ tagged_json<T> id()
{
  return {};
}
} // namespace ptx_json
