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
namespace detail
{
namespace sequential
{


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
void stable_primitive_sort(sequential::execution_policy<DerivedPolicy> &exec,
                           RandomAccessIterator first,
                           RandomAccessIterator last);


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_primitive_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                                  RandomAccessIterator1 keys_first,
                                  RandomAccessIterator1 keys_last,
                                  RandomAccessIterator2 values_first);


} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace thrust

#include <thrust/system/detail/sequential/stable_primitive_sort.inl>

