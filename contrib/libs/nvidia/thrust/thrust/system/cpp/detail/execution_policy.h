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
#include <thrust/system/detail/sequential/execution_policy.h>

namespace thrust
{
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace cpp
{
namespace detail
{

// this awkward sequence of definitions arise
// from the desire both for tag to derive
// from execution_policy and for execution_policy
// to convert to tag (when execution_policy is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of execution_policy
template<typename> struct execution_policy;

// specialize execution_policy for tag
template<>
  struct execution_policy<tag>
    : thrust::system::detail::sequential::execution_policy<tag>
{};

// tag's definition comes before the
// generic definition of execution_policy
struct tag : execution_policy<tag> {};

// allow conversion to tag when it is not a successor
template<typename Derived>
  struct execution_policy
    : thrust::system::detail::sequential::execution_policy<Derived>
{
  typedef tag tag_type; 
  operator tag() const { return tag(); }
};

} // end detail

// alias execution_policy and tag here
using thrust::system::cpp::detail::execution_policy;
using thrust::system::cpp::detail::tag;

} // end cpp
} // end system

// alias items at top-level
namespace cpp
{

using thrust::system::cpp::execution_policy;
using thrust::system::cpp::tag;

} // end cpp
} // end thrust

