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

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// reserve 0 for undefined
#define THRUST_HOST_SYSTEM_CPP 1
#define THRUST_HOST_SYSTEM_OMP 2
#define THRUST_HOST_SYSTEM_TBB 3

#ifndef THRUST_HOST_SYSTEM
#  define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#endif // THRUST_HOST_SYSTEM

#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP
#  define __THRUST_HOST_SYSTEM_NAMESPACE cpp
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
#  define __THRUST_HOST_SYSTEM_NAMESPACE omp
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_TBB
#  define __THRUST_HOST_SYSTEM_NAMESPACE tbb
#endif

// clang-format off
#define __THRUST_HOST_SYSTEM_ROOT thrust/system/cpp
// clang-format on
