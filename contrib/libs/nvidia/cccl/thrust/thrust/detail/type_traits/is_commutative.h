/*
 *  Copyright 2024 NVIDIA Corporation
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
#include <thrust/functional.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename BinaryFunction>
struct is_commutative : public ::cuda::std::false_type
{};

template <typename T>
struct is_commutative<typename ::cuda::std::plus<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::std::multiplies<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::minimum<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::maximum<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::std::logical_or<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::std::logical_and<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::std::bit_or<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::std::bit_and<T>> : public ::cuda::std::is_arithmetic<T>
{};
template <typename T>
struct is_commutative<typename ::cuda::std::bit_xor<T>> : public ::cuda::std::is_arithmetic<T>
{};

} // end namespace detail
THRUST_NAMESPACE_END
