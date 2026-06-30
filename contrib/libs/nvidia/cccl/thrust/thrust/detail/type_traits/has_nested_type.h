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

#define __THRUST_DEFINE_HAS_NESTED_TYPE(trait_name, nested_type_name)         \
  template <typename T>                                                       \
  struct trait_name                                                           \
  {                                                                           \
    using yes_type = char;                                                    \
    using no_type  = int;                                                     \
    template <typename S>                                                     \
    _CCCL_HOST_DEVICE static yes_type test(typename S::nested_type_name*);    \
    template <typename S>                                                     \
    _CCCL_HOST_DEVICE static no_type test(...);                               \
    static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);         \
    using type              = thrust::detail::integral_constant<bool, value>; \
  };
