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
#include <thrust/count.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/count.h>
#include <thrust/system/detail/adl/count.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename EqualityComparable>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::difference_type
    count(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last, const EqualityComparable& value)
{
  using thrust::system::detail::generic::count;
  return count(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, value);
} // end count()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::difference_type
    count_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::count_if;
  return count_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end count_if()


template <typename InputIterator, typename EqualityComparable>
typename thrust::iterator_traits<InputIterator>::difference_type
count(InputIterator first, InputIterator last, const EqualityComparable& value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::count(select_system(system), first, last, value);
} // end count()


template <typename InputIterator, typename Predicate>
typename thrust::iterator_traits<InputIterator>::difference_type
count_if(InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::count_if(select_system(system), first, last, pred);
} // end count_if()

THRUST_NAMESPACE_END
