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

/*! \file thrust/iterator/counting_iterator.h
 *  \brief An iterator which returns an increasing incrementable value
 *         when dereferenced
 */

/*
 * Copyright David Abrahams 2003.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/strided_iterator.h>

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/type_traits>

THRUST_NAMESPACE_BEGIN

template <typename Incrementable, typename System, typename Traversal, typename Difference, typename StrideHolder>
class counting_iterator;

namespace detail
{
template <typename Number>
using counting_iterator_difference_type =
  ::cuda::std::_If<::cuda::std::is_integral_v<Number> && sizeof(Number) < sizeof(int), int, ::cuda::std::ptrdiff_t>;

template <typename Incrementable, typename System, typename Traversal, typename Difference, typename StrideHolder>
struct make_counting_iterator_base
{
  using system = typename eval_if<::cuda::std::is_same<System, use_default>::value,
                                  ::cuda::std::type_identity<any_system_tag>,
                                  ::cuda::std::type_identity<System>>::type;

  using traversal = replace_if_use_default<Traversal, ::cuda::std::type_identity<random_access_traversal_tag>>;
  using difference =
    replace_if_use_default<Difference, ::cuda::std::type_identity<counting_iterator_difference_type<Incrementable>>>;

  // our implementation departs from Boost's in that counting_iterator::dereference
  // returns a copy of its counter, rather than a reference to it. returning a reference
  // to the internal state of an iterator causes subtle bugs (consider the temporary
  // iterator created in the expression *(iter + i)) and has no compelling use case
  using type =
    iterator_adaptor<counting_iterator<Incrementable, System, Traversal, Difference, StrideHolder>,
                     Incrementable,
                     Incrementable,
                     system,
                     traversal,
                     Incrementable,
                     difference>;
};

using unit_stride = compile_time_value<1>;
} // namespace detail

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p counting_iterator is an iterator which represents a pointer into a range of sequentially changing values. This
//! iterator is useful for creating a range filled with a sequence without explicitly storing it in memory. Using \p
//! counting_iterator saves memory capacity and bandwidth.
//!
//! The following code snippet demonstrates how to create a \p counting_iterator whose \c value_type is \c int and which
//! sequentially increments by \c 1.
//!
//! \code
//! #include <thrust/iterator/counting_iterator.h>
//! ...
//! // create iterators
//! thrust::counting_iterator<int> first(10);
//! thrust::counting_iterator<int> last = first + 3;
//!
//! first[0]   // returns 10
//! first[1]   // returns 11
//! first[100] // returns 110
//!
//! // sum of [first, last)
//! thrust::reduce(first, last);   // returns 33 (i.e. 10 + 11 + 12)
//!
//! // initialize vector to [0,1,2,..]
//! thrust::counting_iterator<int> iter(0);
//! thrust::device_vector<int> vec(500);
//! thrust::copy(iter, iter + vec.size(), vec.begin());
//! \endcode
//!
//! This next example demonstrates how to use a \p counting_iterator with the \p thrust::copy_if function to compute the
//! indices of the non-zero elements of a \p device_vector. In this example, we use the \p make_counting_iterator
//! function to avoid specifying the type of the \p counting_iterator.
//!
//! \code
//! #include <thrust/iterator/counting_iterator.h>
//! #include <thrust/copy.h>
//! #include <thrust/functional.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!  // this example computes indices for all the nonzero values in a sequence
//!
//!  // sequence of zero and nonzero values
//!  thrust::device_vector<int> stencil{0, 1, 1, 0, 0, 1, 0, 1};
//!
//!  // storage for the nonzero indices
//!  thrust::device_vector<int> indices(8);
//!
//!  // compute indices of nonzero elements
//!  using IndexIterator = thrust::device_vector<int>::iterator;
//!
//!  // use make_counting_iterator to define the sequence [0, 8)
//!  IndexIterator indices_end = thrust::copy_if(thrust::make_counting_iterator(0),
//!                                              thrust::make_counting_iterator(8),
//!                                              stencil.begin(),
//!                                              indices.begin(),
//!                                              ::cuda::std::identity{});
//!  // indices now contains [1,2,5,7]
//!
//!  return 0;
//! }
//! \endcode
//!
//! \see make_counting_iterator
template <typename Incrementable,
          typename System       = use_default,
          typename Traversal    = use_default,
          typename Difference   = use_default,
          typename StrideHolder = detail::unit_stride>
class _CCCL_DECLSPEC_EMPTY_BASES counting_iterator
    : public detail::make_counting_iterator_base<Incrementable, System, Traversal, Difference, StrideHolder>::type
    , StrideHolder
{
  //! \cond
  using super_t =
    typename detail::make_counting_iterator_base<Incrementable, System, Traversal, Difference, StrideHolder>::type;
  friend class iterator_core_access;

public:
  using reference       = typename super_t::reference;
  using difference_type = typename super_t::difference_type;
  //! \endcond

  //! Default constructor initializes this \p counting_iterator's counter to `Incrementable{}`.
  _CCCL_HOST_DEVICE counting_iterator()
      : super_t(Incrementable{})
  {}

  //! Copy constructor copies the value of another counting_iterator with related System type.
  //!
  //! \param rhs The \p counting_iterator to copy.
  template <class OtherSystem,
            detail::enable_if_convertible_t<
              iterator_system_t<counting_iterator<Incrementable, OtherSystem, Traversal, Difference, StrideHolder>>,
              iterator_system_t<super_t>,
              int> = 0>
  _CCCL_HOST_DEVICE
  counting_iterator(counting_iterator<Incrementable, OtherSystem, Traversal, Difference, StrideHolder> const& rhs)
      : super_t(rhs.base())
  {}

  //! This \c explicit constructor copies the value of an \c Incrementable into a new \p counting_iterator's \c
  //! Incrementable counter.
  //!
  //! \param x The initial value of the new \p counting_iterator's \c Incrementable counter.
  _CCCL_HOST_DEVICE explicit counting_iterator(Incrementable x)
      : super_t(x)
  {}

  _CCCL_HOST_DEVICE explicit counting_iterator(Incrementable x, StrideHolder stride)
      : super_t(x)
      , StrideHolder(stride)
  {}

  //! \cond

private:
  template <typename S = StrideHolder>
  _CCCL_HOST_DEVICE auto stride() const
  {
    return static_cast<const S&>(*this).value;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void advance(difference_type n)
  {
    if constexpr (::cuda::std::is_same_v<StrideHolder, detail::unit_stride>)
    {
      this->base_reference() = static_cast<Incrementable>(this->base_reference() + n);
    }
    else
    {
      this->base_reference() += n * stride();
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void increment()
  {
    if constexpr (::cuda::std::is_same_v<StrideHolder, detail::unit_stride>)
    {
      ++this->base_reference();
    }
    else
    {
      this->base_reference() += stride();
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void decrement()
  {
    if constexpr (::cuda::std::is_same_v<StrideHolder, detail::unit_stride>)
    {
      --this->base_reference();
    }
    else
    {
      this->base_reference() -= stride();
    }
  }

  _CCCL_HOST_DEVICE reference dereference() const
  {
    return this->base_reference();
  }

  // note that we implement equal specially for floating point counting_iterator
  template <typename OtherSystem, typename OtherTraversal, typename OtherDifference, typename OtherStrideHolder>
  _CCCL_HOST_DEVICE bool equal(
    counting_iterator<Incrementable, OtherSystem, OtherTraversal, OtherDifference, OtherStrideHolder> const& y) const
  {
    if constexpr (::cuda::is_floating_point_v<Incrementable>)
    {
      return distance_to(y) == 0;
    }
    else
    {
      return this->base() == y.base();
    }
  }

  template <typename OtherSystem, typename OtherTraversal, typename OtherDifference>
  _CCCL_HOST_DEVICE difference_type distance_to(
    counting_iterator<Incrementable, OtherSystem, OtherTraversal, OtherDifference, StrideHolder> const& y) const
  {
    if constexpr (::cuda::std::is_integral<Incrementable>::value)
    {
      return static_cast<difference_type>(y.base()) - static_cast<difference_type>(this->base());
    }
    else
    {
      return y.base() - this->base();
    }
  }

  //! \endcond
};

//! \p make_counting_iterator creates a \p counting_iterator
//! using an initial value for its \c Incrementable counter.
//!
//! \param x The initial value of the new \p counting_iterator's counter.
//! \return A new \p counting_iterator whose counter has been initialized to \p x.
template <typename Incrementable>
inline _CCCL_HOST_DEVICE counting_iterator<Incrementable> make_counting_iterator(Incrementable x)
{
  return counting_iterator<Incrementable>(x);
}

// FIXME(bgruber): Sphinx fails to document the fancyiterators group if make_counting_iterator has overloads, so we
// exclude them for now
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

//! Constructs a counting_iterator with a runtime stride
template <typename Incrementable, typename Stride>
_CCCL_HOST_DEVICE auto make_counting_iterator(Incrementable x, Stride stride)
{
  return counting_iterator<Incrementable, use_default, random_access_traversal_tag, use_default, runtime_value<Stride>>(
    x, {stride});
}

//! Constructs a counting_iterator with a compile-time stride
template <auto Stride, typename Incrementable>
_CCCL_HOST_DEVICE auto make_counting_iterator(Incrementable x)
{
  return counting_iterator<Incrementable,
                           use_default,
                           random_access_traversal_tag,
                           use_default,
                           compile_time_value<Stride>>(x, {});
}
#endif // _CCCL_DOXYGEN_INVOKED

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
