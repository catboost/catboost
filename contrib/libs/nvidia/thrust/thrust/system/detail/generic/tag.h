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


/*! \file generic/tag.h
 *  \brief Implementation of the generic backend's tag.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{

// tag exists only to make the generic entry points the least priority match
// during ADL. tag should not be derived from and is constructible from anything
struct tag
{
  template<typename T>
  __host__ __device__ inline
  tag(const T &) {}
};

} // end generic
} // end detail
} // end system
} // end thrust

