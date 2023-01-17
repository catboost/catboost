/*
 *  Copyright 2018 NVIDIA Corporation
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

/*! \file void_t.h
 *  \brief C++17's `void_t`. 
 */

#pragma once

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2017
#  include <type_traits>
#endif

THRUST_NAMESPACE_BEGIN

#if THRUST_CPP_DIALECT >= 2011

template <typename...> struct voider { using type = void; };

#if THRUST_CPP_DIALECT >= 2017
using std::void_t;
#else
template <typename... Ts> using void_t = typename voider<Ts...>::type;
#endif

#else // Older than C++11.

template <
  typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
, typename = void
>
struct voider
{
  typedef void type;
};

#endif

THRUST_NAMESPACE_END

