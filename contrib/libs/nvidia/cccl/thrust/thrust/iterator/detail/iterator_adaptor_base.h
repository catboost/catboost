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
#include <thrust/detail/use_default.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

// forward declaration of iterator_adaptor for make_iterator_adaptor_base below
template <typename Derived,
          typename Base,
          typename Value,
          typename System,
          typename Traversal,
          typename Reference,
          typename Difference>
class iterator_adaptor;

namespace detail
{
// If T is use_default, return the result of invoking DefaultNullaryFn, otherwise return T.
template <class T, class DefaultNullaryFn>
using replace_if_use_default = typename ::cuda::std::
  _If<::cuda::std::is_same_v<T, use_default>, DefaultNullaryFn, ::cuda::std::type_identity<T>>::type;

// A metafunction which computes an iterator_adaptor's base class, a specialization of iterator_facade.
template <typename Derived,
          typename Base,
          typename Value,
          typename System,
          typename Traversal,
          typename Reference,
          typename Difference>
struct make_iterator_adaptor_base
{
private:
  using value     = replace_if_use_default<Value, lazy_trait<it_value_t, Base>>;
  using system    = replace_if_use_default<System, iterator_system<Base>>;
  using traversal = replace_if_use_default<Traversal, iterator_traversal<Base>>;
  using reference =
    replace_if_use_default<Reference,
                           ::cuda::std::_If<::cuda::std::is_same_v<Value, use_default>,
                                            lazy_trait<it_reference_t, Base>,
                                            ::cuda::std::add_lvalue_reference<Value>>>;
  using difference = replace_if_use_default<Difference, lazy_trait<it_difference_t, Base>>;

public:
  using type = iterator_facade<Derived, value, system, traversal, reference, difference>;
};

} // namespace detail
THRUST_NAMESPACE_END
