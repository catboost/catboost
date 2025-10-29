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

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/is_metafunction_defined.h>
#include <thrust/detail/type_traits/minimum_type.h>

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename... Ts>
struct unrelated_systems
{};

template <typename System>
inline constexpr bool is_unrelated_systems = false;

template <typename... Ts>
inline constexpr bool is_unrelated_systems<unrelated_systems<Ts...>> = true;

template <typename... Ts>
_CCCL_HOST_DEVICE auto minimum_system_impl(int) -> minimum_type<Ts...>;
template <typename... Ts>
_CCCL_HOST_DEVICE auto minimum_system_impl(long) -> unrelated_systems<Ts...>;

// if a minimum system exists for these arguments, return it
// otherwise, collect the arguments and report them as unrelated
template <typename... Ts>
using minimum_system_t = decltype(minimum_system_impl<Ts...>(0));

} // namespace detail
THRUST_NAMESPACE_END
