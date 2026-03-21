// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_facade.h>

#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p offset_iterator wraps another iterator and an integral offset, applies the offset to the iterator when
//! dereferencing, comparing, or computing the distance between two offset_iterators. This is useful, when the
//! underlying iterator cannot be incremented, decremented, or advanced (e.g., because those operations are only
//! supported in device code).
//!
//!
//! The following code snippet demonstrates how to create an \p offset_iterator:
//!
//! \code
//! #include <thrust/iterator/offset_iterator.h>
//! #include <thrust/fill.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   thrust::device_vector<int> data{1, 2, 3, 4};
//!   auto b = offset_iterator{data.begin(), 1};
//!   auto e = offset_iterator{data.end(), -1};
//!   thrust::fill(b, e, 42);
//!   // data is now [1, 42, 42, 4]
//!   ++b; // does not call ++ on the underlying iterator
//!   assert(b == e - 1);
//!
//!   return 0;
//! }
//! \endcode
//!
//! Alternatively, an \p offset_iterator can also use an iterator to retrieve the offset from an iterator. However, such
//! an \p offset_iterator cannot be moved anymore by changing the offset, so it will move the base iterator instead.
//!
//! \code
//! #include <thrust/iterator/offset_iterator.h>
//! #include <thrust/fill.h>
//! #include <thrust/functional.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   using thrust::placeholders::_1;
//!   thrust::device_vector<int> data{1, 2, 3, 4};
//!
//!   thrust::device_vector<ptrdiff> offsets{1}; // offset is only available on device
//!   auto offset = thrust::make_transform_iterator(offsets.begin(), _1 * 2);
//!   thrust::offset_iterator iter(data.begin(), offset); // load and transform offset upon access
//!   // iter is at position 2 (= 1 * 2) in data, and would return 3 in device code
//!
//!   return 0;
//! }
//! \endcode
//!
//! In the above example, the offset is loaded from a device vector, transformed by a \p transform_iterator, and then
//! applied to the underlying iterator, when the \p offset_iterator is accessed.
template <typename Iterator, typename Offset = typename ::cuda::std::iterator_traits<Iterator>::difference_type>
class offset_iterator : public iterator_adaptor<offset_iterator<Iterator, Offset>, Iterator>
{
  //! \cond
  friend class iterator_core_access;
  using super_t = iterator_adaptor<offset_iterator<Iterator, Offset>, Iterator>;

public:
  using reference       = typename super_t::reference;
  using difference_type = typename super_t::difference_type;
  //! \endcond

  _CCCL_HOST_DEVICE offset_iterator(Iterator it = {}, Offset offset = {})
      : super_t(::cuda::std::move(it))
      , m_offset(offset)
  {}

  _CCCL_HOST_DEVICE const Offset& offset() const
  {
    return m_offset;
  }

  _CCCL_HOST_DEVICE Offset& offset()
  {
    return m_offset;
  }

  //! \cond

private:
  static constexpr bool indirect_offset = ::cuda::std::indirectly_readable<Offset>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE auto offset_value() const
  {
    if constexpr (indirect_offset)
    {
      return static_cast<difference_type>(*m_offset);
    }
    else
    {
      return static_cast<difference_type>(m_offset);
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE reference dereference() const
  {
    return *(this->base() + offset_value());
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE bool equal(const offset_iterator& other) const
  {
    return this->base() + offset_value() == other.base() + other.offset_value();
  }

  _CCCL_HOST_DEVICE void advance(difference_type n)
  {
    if constexpr (indirect_offset)
    {
      this->base_reference() += n;
    }
    else
    {
      m_offset += n;
    }
  }

  _CCCL_HOST_DEVICE void increment()
  {
    if constexpr (indirect_offset)
    {
      ++this->base_reference();
    }
    else
    {
      ++m_offset;
    }
  }

  _CCCL_HOST_DEVICE void decrement()
  {
    if constexpr (indirect_offset)
    {
      --this->base_reference();
    }
    else
    {
      --m_offset;
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE difference_type distance_to(const offset_iterator& other) const
  {
    return (other.base() + other.offset_value()) - (this->base() + offset_value());
  }

  Offset m_offset;
  //! \endcond
};

template <typename Iterator>
_CCCL_HOST_DEVICE offset_iterator(Iterator) -> offset_iterator<Iterator>;

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
