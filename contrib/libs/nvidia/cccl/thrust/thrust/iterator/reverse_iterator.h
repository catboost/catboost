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

//! \file thrust/iterator/reverse_iterator.h
//! \brief An iterator adaptor which adapts another iterator to traverse backwards

/*
 * (C) Copyright David Abrahams 2002.
 * (C) Copyright Jeremy Siek    2002.
 * (C) Copyright Thomas Witt    2002.
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
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename>
class reverse_iterator;

namespace detail
{
template <typename BidirectionalIterator>
struct make_reverse_iterator_base
{
  using type = iterator_adaptor<reverse_iterator<BidirectionalIterator>, BidirectionalIterator>;
};
} // namespace detail

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p reverse_iterator is an iterator which represents a pointer into a reversed view of a given range. In this way, \p
//! reverse_iterator allows backwards iteration through a bidirectional input range.
//!
//! It is important to note that although \p reverse_iterator is constructed from a given iterator, it points to the
//! element preceding it. In this way, the past-the-end \p reverse_iterator of a given range points to the element
//! preceding the first element of the input range. By the same token, the first \p reverse_iterator of a given range is
//! constructed from a past-the-end iterator of the original range yet points to the last element of the input.
//!
//! The following code snippet demonstrates how to create a \p reverse_iterator which represents a reversed view of the
//! contents of a \p device_vector.
//!
//! \code
//! #include <thrust/iterator/reverse_iterator.h>
//! #include <thrust/device_vector.h>
//! ...
//! thrust::device_vector<float> v{0.0f, 1.0f, 2.0f, 3.0f};
//!
//! using Iterator = thrust::device_vector<float>::iterator;
//!
//! // note that we point the iterator to the *end* of the device_vector
//! thrust::reverse_iterator<Iterator> iter(values.end());
//!
//! *iter;   // returns 3.0f;
//! iter[0]; // returns 3.0f;
//! iter[1]; // returns 2.0f;
//! iter[2]; // returns 1.0f;
//! iter[3]; // returns 0.0f;
//!
//! // iter[4] is an out-of-bounds error
//! \endcode
//!
//! Since reversing a range is a common operation, containers like \p device_vector have nested aliases for declaration
//! shorthand and methods for constructing reverse_iterators. The following code snippet is equivalent to the previous:
//!
//! \code
//! #include <thrust/device_vector.h>
//! ...
//! thrust::device_vector<float> v{0.0f, 1.0f, 2.0f, 3.0f};
//!
//! // we use the nested type reverse_iterator to refer to a reversed view of a device_vector and the method rbegin() to
//! // create a reverse_iterator pointing to the beginning of the reversed device_vector
//! thrust::device_iterator<float>::reverse_iterator iter = values.rbegin();
//!
//! *iter;   // returns 3.0f;
//! iter[0]; // returns 3.0f;
//! iter[1]; // returns 2.0f;
//! iter[2]; // returns 1.0f;
//! iter[3]; // returns 0.0f;
//!
//! // iter[4] is an out-of-bounds error
//!
//! // similarly, rend() points to the end of the reversed sequence:
//! assert(values.rend() == (iter + 4));
//! \endcode
//!
//! Finally, the following code snippet demonstrates how to use reverse_iterator to perform a reversed prefix sum
//! operation on the contents of a device_vector:
//!
//! \code
//! #include <thrust/device_vector.h>
//! #include <thrust/scan.h>
//! ...
//! thrust::device_vector<int> v{0, 1, 2, 3, 4};
//!
//! thrust::device_vector<int> result(5);
//!
//! // exclusive scan v into result in reverse
//! thrust::exclusive_scan(v.rbegin(), v.rend(), result.begin());
//!
//! // result is now {0, 4, 7, 9, 10}
//! \endcode
//!
//! \see make_reverse_iterator
template <typename BidirectionalIterator>
class reverse_iterator : public detail::make_reverse_iterator_base<BidirectionalIterator>::type
{
  //! \cond

private:
  using super_t = typename detail::make_reverse_iterator_base<BidirectionalIterator>::type;

  friend class iterator_core_access;
  //! \endcond

public:
  reverse_iterator() = default;

  //! \p Constructor accepts a \c BidirectionalIterator pointing to a range
  //! for this \p reverse_iterator to reverse.
  //!
  //! \param x A \c BidirectionalIterator pointing to a range to reverse.
  _CCCL_HOST_DEVICE explicit reverse_iterator(BidirectionalIterator x)
      : super_t(x)
  {}

  //!  \p Copy constructor allows construction from a related compatible
  //!  \p reverse_iterator.
  //!
  //!  \param r A \p reverse_iterator to copy from.
  template <typename OtherBidirectionalIterator,
            detail::enable_if_convertible_t<OtherBidirectionalIterator, BidirectionalIterator, int> = 0>
  _CCCL_HOST_DEVICE reverse_iterator(reverse_iterator<OtherBidirectionalIterator> const& rhs)
      : super_t(rhs.base())
  {}

  //! \cond

private:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE typename super_t::reference dereference() const
  {
    auto b = this->base();
    return *--b;
  }

  _CCCL_HOST_DEVICE void increment()
  {
    --this->base_reference();
  }

  _CCCL_HOST_DEVICE void decrement()
  {
    ++this->base_reference();
  }

  _CCCL_HOST_DEVICE void advance(typename super_t::difference_type n)
  {
    this->base_reference() += -n;
  }

  template <typename OtherBidirectionalIterator>
  _CCCL_HOST_DEVICE typename super_t::difference_type
  distance_to(reverse_iterator<OtherBidirectionalIterator> const& y) const
  {
    return this->base_reference() - y.base();
  }
  //! \endcond
};

//! \p make_reverse_iterator creates a \p reverse_iterator
//! from a \c BidirectionalIterator pointing to a range of elements to reverse.
//!
//! \param x A \c BidirectionalIterator pointing to a range to reverse.
//! \return A new \p reverse_iterator which reverses the range \p x.
template <typename BidirectionalIterator>
_CCCL_HOST_DEVICE reverse_iterator<BidirectionalIterator> make_reverse_iterator(BidirectionalIterator x)
{
  return reverse_iterator<BidirectionalIterator>(x);
}

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
