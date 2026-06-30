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

/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config> // IWYU pragma: export

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE: The order of these #includes matters.

#include <thrust/detail/config/compiler.h> // IWYU pragma: export
#include <thrust/detail/config/cpp_dialect.h> // IWYU pragma: export
#include <thrust/detail/config/simple_defines.h> // IWYU pragma: export
// host_system.h & device_system.h must be #included as early as possible because other config headers depend on it
#include <thrust/detail/config/host_system.h> // IWYU pragma: export

#include <thrust/detail/config/device_system.h> // IWYU pragma: export
#include <thrust/detail/config/namespace.h> // IWYU pragma: export
