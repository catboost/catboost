/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh> // static_size_v
#include <cub/util_namespace.cuh>

#include <cuda/std/array> // array
#include <cuda/std/cstddef> // size_t
#include <cuda/std/iterator> // _CUDA_VSTD::iter_value_t
#include <cuda/std/type_traits> // _If
#include <cuda/std/utility> // index_sequence

CUB_NAMESPACE_BEGIN
namespace detail
{

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/***********************************************************************************************************************
 * Generic Array-like to Array Conversion
 **********************************************************************************************************************/

template <typename CastType, typename Input, _CUDA_VSTD::size_t... i>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE _CUDA_VSTD::array<CastType, static_size_v<Input>>
to_array_impl(const Input& input, _CUDA_VSTD::index_sequence<i...>)
{
  using ArrayType = _CUDA_VSTD::array<CastType, static_size_v<Input>>;
  return ArrayType{static_cast<CastType>(input[i])...};
}

template <typename CastType = void, typename Input>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE _CUDA_VSTD::array<CastType, static_size_v<Input>>
to_array(const Input& input)
{
  using InputType = _CUDA_VSTD::iter_value_t<Input>;
  using CastType1 = _CUDA_VSTD::_If<_CUDA_VSTD::is_same_v<CastType, void>, InputType, CastType>;
  return to_array_impl<CastType1>(input, _CUDA_VSTD::make_index_sequence<static_size_v<Input>>{});
}

#endif // !_CCCL_DOXYGEN_INVOKED

} // namespace detail
CUB_NAMESPACE_END
