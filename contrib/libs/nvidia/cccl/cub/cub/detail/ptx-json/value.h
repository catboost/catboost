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

#include <cub/detail/ptx-json/string.h>

#include <cuda/std/type_traits>

namespace ptx_json
{
template <auto V>
struct value_traits
{
  using type = void;
};

template <auto V, typename = value_traits<V>::type>
struct value;

#pragma nv_diag_suppress 177
template <int N, string<N> V>
struct value_traits<V>
{
  using type = cuda::std::make_index_sequence<N>;
};
#pragma nv_diag_default 177

template <typename T>
struct is_value : cuda::std::false_type
{};

template <auto V>
struct is_value<value<V>> : cuda::std::true_type
{};

template <typename T>
concept a_value = is_value<T>::value;

template <a_value auto Nested>
struct value<Nested, void>
{
  __forceinline__ __device__ static void emit()
  {
    value<Nested>::emit();
  }
};

template <int V>
struct value<V, void>
{
  __forceinline__ __device__ static void emit()
  {
    asm volatile("%0" ::"n"(V) : "memory");
  }
};

#pragma nv_diag_suppress 842
template <int N, string<N> V, cuda::std::size_t... Is>
struct value<V, cuda::std::index_sequence<Is...>>
{
#pragma nv_diag_default 842
  __forceinline__ __device__ static void emit()
  {
    // See the definition of storage_helper for why laundering the string through it is necessary.
    asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
  }
};
}; // namespace ptx_json
