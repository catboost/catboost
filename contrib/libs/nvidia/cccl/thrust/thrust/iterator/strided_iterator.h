// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! Holds a runtime value
template <typename T>
struct runtime_value
{
  T value;
};

//! Holds a compile-time value
// we cannot use ::cuda::std::integral_constant, because it has a conversion operator to T that causes an ambiguity
// with operator+(counting_iterator, counting_iterator::difference_type) in any expression `counting_iterator +
// integral`.
template <auto Value>
struct compile_time_value
{
  static constexpr decltype(Value) value = Value;
};

namespace detail
{
template <typename T>
inline constexpr bool is_compile_time_value = false;

template <auto Value>
inline constexpr bool is_compile_time_value<compile_time_value<Value>> = true;
} // namespace detail

//! A \p strided_iterator wraps another iterator and moves it by a specified stride each time it is incremented or
//! decremented.
//!
//! \param RandomAccessIterator A random access iterator
//! \param StrideHolder Either a \ref runtime_value or a \ref compile_time_value specifying the stride
template <typename RandomAccessIterator, typename StrideHolder>
class _CCCL_DECLSPEC_EMPTY_BASES strided_iterator
    : public iterator_adaptor<strided_iterator<RandomAccessIterator, StrideHolder>, RandomAccessIterator>
    , StrideHolder
{
  //! \cond
  using super_t = iterator_adaptor<strided_iterator, RandomAccessIterator>;
  friend class iterator_core_access;

public:
  using difference_type = typename super_t::difference_type;
  //! \endcond

  static_assert(::cuda::std::random_access_iterator<RandomAccessIterator>,
                "The iterator underlying a strided_iterator must be a random access iterator.");
  static_assert(::cuda::std::is_same_v<iterator_traversal_t<RandomAccessIterator>, random_access_traversal_tag>);
  static_assert(::cuda::std::is_convertible_v<decltype(StrideHolder::value), difference_type>,
                "The stride must be convertible to the iterator's difference_type");

  strided_iterator() = default;

  //! Creates a strided_iterator from an existing iterator and a stride.
  _CCCL_HOST_DEVICE strided_iterator(RandomAccessIterator it, StrideHolder stride = {})
      : super_t(it)
      , StrideHolder(stride)
  {}

  static constexpr bool has_static_stride = detail::is_compile_time_value<StrideHolder>;

  //! Returns either the \ref runtime_value or the \ref compile_time_value holding the stride's value
  _CCCL_HOST_DEVICE const auto& stride_holder() const
  {
    return static_cast<const StrideHolder&>(*this);
  }

  //! Returns the stride's value
  _CCCL_HOST_DEVICE auto stride() const -> difference_type
  {
    return static_cast<detail::it_difference_t<RandomAccessIterator>>(stride_holder().value);
  }

private:
  //! \cond
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void advance(difference_type n)
  {
    this->base_reference() += n * stride();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void increment()
  {
    this->base_reference() += stride();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void decrement()
  {
    this->base_reference() -= stride();
  }

  template <typename OtherStrideHolder>
  _CCCL_HOST_DEVICE bool equal(strided_iterator<RandomAccessIterator, OtherStrideHolder> const& other) const
  {
    return this->base() == other.base();
  }

  _CCCL_HOST_DEVICE difference_type distance_to(strided_iterator const& other) const
  {
    const difference_type dist = other.base() - this->base();
    _CCCL_ASSERT(dist % stride() == 0, "Underlying iterator difference must be divisible by the stride");
    return dist / stride();
  }
  //! \endcond
};

//! Constructs a strided_iterator with a runtime stride
template <typename Iterator, typename Stride>
_CCCL_HOST_DEVICE auto make_strided_iterator(Iterator it, Stride stride)
{
  return strided_iterator<Iterator, runtime_value<Stride>>(it, {stride});
}

//! Constructs a strided_iterator with a compile-time stride
template <auto Stride, typename Iterator>
_CCCL_HOST_DEVICE auto make_strided_iterator(Iterator it)
{
  return strided_iterator<Iterator, compile_time_value<Stride>>(it, {});
}

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
