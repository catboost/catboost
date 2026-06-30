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

#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/host_system_tag.h>
#include <thrust/iterator/iterator_categories.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
_CCCL_HOST_DEVICE auto cat_to_system_impl(...) -> void;

_CCCL_HOST_DEVICE auto cat_to_system_impl(const input_host_iterator_tag&) -> host_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const output_host_iterator_tag&) -> host_system_tag;

_CCCL_HOST_DEVICE auto cat_to_system_impl(const input_device_iterator_tag&) -> device_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const output_device_iterator_tag&) -> device_system_tag;

template <typename Category>
struct iterator_category_to_system
{
  using type = decltype(cat_to_system_impl(Category{}));
};
} // namespace detail
THRUST_NAMESPACE_END
