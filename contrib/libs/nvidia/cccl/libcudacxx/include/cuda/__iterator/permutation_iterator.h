//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_PERMUTATION_ITERATOR_H
#define _CUDA___ITERATOR_PERMUTATION_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/detail/libcxx/include/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @addtogroup iterators
//! @{

//! @brief @c permutation_iterator is an iterator which represents a pointer into a reordered view of a given range.
//! @c permutation_iterator is an imprecise name; the reordered view need not be a strict permutation. This iterator is
//! useful for fusing a scatter or gather operation with other algorithms.
//!
//! This iterator takes two arguments:
//!
//!   - an iterator to the range @c V on which the "permutation" will be applied, referred to as @c iter below
//!   - an iterator to a range of indices defining the reindexing scheme that determines how the elements of @c V will
//!   be permuted, referred to as @c index below
//!
//! Note that @c permutation_iterator is not limited to strict permutations of the given range @c V. The distance
//! between begin and end of the reindexing iterators is allowed to be smaller compared to the size of the range @c V,
//! in which case the @c permutation_iterator only provides a "permutation" of a subset of @c V. The indices do not
//! need to be unique. In this same context, it must be noted that the past-the-end @c permutation_iterator is
//! completely defined by means of the past-the-end iterator to the indices.
//!
//! The following code snippet demonstrates how to create a @c permutation_iterator which represents a reordering of the
//! contents of a @c device_vector.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//! ...
//! thrust::device_vector<float> values{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
//! thrust::device_vector<int> indices{2, 6, 1, 3};
//!
//! using ElementIterator = thrust::device_vector<float>::iterator;
//! using IndexIterator = thrust::device_vector<int>::iterator;
//!
//! cuda::permutation_iterator<ElementIterator,IndexIterator> iter(values.begin(), indices.begin());
//!
//! *iter;   // returns 30.0f;
//! iter[0]; // returns 30.0f;
//! iter[1]; // returns 70.0f;
//! iter[2]; // returns 20.0f;
//! iter[3]; // returns 40.0f;
//!
//! // iter[4] is an out-of-bounds error
//!
//! *iter   = -1.0f; // sets values[2] to -1.0f;
//! iter[0] = -1.0f; // sets values[2] to -1.0f;
//! iter[1] = -1.0f; // sets values[6] to -1.0f;
//! iter[2] = -1.0f; // sets values[1] to -1.0f;
//! iter[3] = -1.0f; // sets values[3] to -1.0f;
//!
//! // values is now {10, -1, -1, -1, 50, 60, -1, 80}
//! @endcode
template <class _Iter, class _Index = _Iter>
class permutation_iterator
{
private:
  _Iter __iter_   = {};
  _Index __index_ = {};

  // We need to factor these out because old gcc chokes with using arguments in friend functions
  template <class _Iter1>
  static constexpr bool __nothrow_difference = noexcept(_CUDA_VSTD::declval<_Iter1>() - _CUDA_VSTD::declval<_Iter1>());

  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_equality = noexcept(_CUDA_VSTD::declval<_Iter1>() == _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_less_than = noexcept(_CUDA_VSTD::declval<_Iter1>() < _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_less_equal = noexcept(_CUDA_VSTD::declval<_Iter1>() <= _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_greater_than =
    noexcept(_CUDA_VSTD::declval<_Iter1>() > _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_greater_equal =
    noexcept(_CUDA_VSTD::declval<_Iter1>() >= _CUDA_VSTD::declval<_Iter2>());

public:
  using iterator_type       = _Iter;
  using iterator_concept    = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category   = _CUDA_VSTD::random_access_iterator_tag;
  using value_type          = _CUDA_VSTD::iter_value_t<_Iter>;
  using __iter_difference_t = _CUDA_VSTD::iter_difference_t<_Iter>;
  using difference_type     = _CUDA_VSTD::iter_difference_t<_Index>;
  using __index_value_t     = _CUDA_VSTD::iter_value_t<_Index>;

  //! Ensure that the user passes an iterator to something interger_like
  static_assert(_CUDA_VSTD::__integer_like<__index_value_t>,
                "cuda::permutation_iterator: _Index must be an iterator to integer_like");

  //! Ensure that the index value_type is convertible to difference_type
  static_assert(_CUDA_VSTD::is_convertible_v<__index_value_t, difference_type>,
                "cuda::permutation_iterator: _Indexs value type must be convertible to iter_difference<Iter>");

  //! To actually use operator+ we need the index iterator to be random access
  static_assert(_CUDA_VSTD::__has_random_access_traversal<_Index>,
                "cuda::permutation_iterator: _Index must be a random access iterator!");

  //! To actually use operator+ we need the base iterator to be random access
  static_assert(_CUDA_VSTD::__has_random_access_traversal<_Iter>,
                "cuda::permutation_iterator: _Iter must be a random access iterator!");

  //! @brief Default constructs an @c permutation_iterator with a value initialized iterator and index
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI constexpr permutation_iterator() = default;

  //! @brief Constructs an @c permutation_iterator from an iterator and an optional index
  //! @param __iter The iterator to to index from
  //! @param __index The iterator with the permutations
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator(_Iter __iter, _Index __index) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Index>)
      : __iter_(__iter)
      , __index_(__index)
  {}

  //! @brief Returns a const reference to the stored base iterator @c iter
  [[nodiscard]] _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __iter_;
  }

  //! @brief Extracts the stored base iterator @c iter
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Iter base() && noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Iter>)
  {
    return _CUDA_VSTD::move(__iter_);
  }

  //! @cond
  //! @brief Returns a const reference to the stored index iterator @c index
  [[nodiscard]] _CCCL_API constexpr const _Index& __index() const noexcept
  {
    return __index_;
  }
  //! @endcond

  //! @brief Returns the current index
  //! @return Equivalent to ``*index``
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr difference_type index() const noexcept
  {
    return static_cast<difference_type>(*__index_);
  }

  //! @brief Dereferences the @c permutation_iterator
  //! @return Equivalent to ``iter[*index]``
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  operator*() noexcept(noexcept(__iter_[static_cast<__iter_difference_t>(*__index_)]))
  {
    return __iter_[static_cast<__iter_difference_t>(*__index_)];
  }

  //! @brief Dereferences the @c permutation_iterator
  //! @return Equivalent to ``iter[*index]``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _Iter2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() const
    noexcept(noexcept(__iter_[static_cast<__iter_difference_t>(*__index_)]))
  {
    return __iter_[static_cast<__iter_difference_t>(*__index_)];
  }

  //! @brief Subscripts the @c permutation_iterator by an offset
  //! @param __n The additional offset
  //! @return Equivalent to ``iter[index[__n]]``
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  operator[](difference_type __n) noexcept(noexcept(__iter_[static_cast<__iter_difference_t>(__index_[__n])]))
  {
    return __iter_[static_cast<__iter_difference_t>(__index_[__n])];
  }

  //! @brief Subscripts the @c permutation_iterator by an offset
  //! @param __n The additional offset
  //! @return Equivalent to ``iter[index[__n]]``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _Iter2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator[](difference_type __n) const
    noexcept(noexcept(__iter_[static_cast<__iter_difference_t>(__index_[__n])]))
  {
    return __iter_[static_cast<__iter_difference_t>(__index_[__n])];
  }

  //! @brief Increments the @c permutation_iterator
  //! @return Equivalent to ``++index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator& operator++() noexcept(noexcept(++__index_))
  {
    ++__index_;
    return *this;
  }

  //! @brief Increments the @c permutation_iterator
  //! @return Equivalent to ``index++``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator operator++(int) noexcept(
    noexcept(++__index_)
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Index>)
  {
    permutation_iterator __tmp = *this;
    ++__index_;
    return __tmp;
  }

  //! @brief Increments the @c permutation_iterator
  //! @return Equivalent to ``--index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator& operator--() noexcept(noexcept(--__index_))
  {
    --__index_;
    return *this;
  }

  //! @brief Increments the @c permutation_iterator
  //! @return Equivalent to ``index++``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator operator--(int) noexcept(
    noexcept(--__index_)
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Index>)
  {
    permutation_iterator __tmp = *this;
    --__index_;
    return __tmp;
  }

  //! @brief Advances a @c permutation_iterator by a given number of elements
  //! @param __iter The original @c permutation_iterator
  //! @param __n The number of elements to advance
  //! @return Equivalent to ``permutation_iterator{iter, index + __n}``
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr permutation_iterator
  operator+(const permutation_iterator& __iter, difference_type __n) noexcept( //
    noexcept(__iter.__index_ + __n)
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Index>)
  {
    return permutation_iterator{__iter.__iter_, __iter.__index_ + __n};
  }

  //! @brief Advances a @c permutation_iterator by a given number of elements
  //! @param __n The number of elements to advance
  //! @param __iter The original @c permutation_iterator
  //! @return Equivalent to ``permutation_iterator{iter, index + __n}``
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr permutation_iterator
  operator+(difference_type __n, const permutation_iterator& __iter) noexcept(
    noexcept(__iter.__index_ + __n)
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Index>)
  {
    return permutation_iterator{__iter.__iter_, __iter.__index_ + __n};
  }

  //! @brief Advances the @c permutation_iterator by a given number of elements
  //! @param __n The number of elements to advance
  //! @return Equivalent to ``index + __n``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator& operator+=(difference_type __n) noexcept(noexcept(__index_ += __n))
  {
    __index_ += __n;
    return *this;
  }

  //! @brief Decrements a @c permutation_iterator by a given number of elements
  //! @param __iter The original @c permutation_iterator
  //! @param __n The number of elements to decrement
  //! @return Equivalent to ``permutation_iterator{iter, index - __n}``
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr permutation_iterator
  operator-(const permutation_iterator& __iter, difference_type __n) noexcept( //
    noexcept(__iter.__index_ - __n)
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Index>)
  {
    return permutation_iterator{__iter.__iter_, __iter.__index_ - __n};
  }

  //! @brief Decrements the @c permutation_iterator by a given number of elements
  //! @param __n The number of elements to decrement
  //! @return Equivalent to ``index - __n``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator& operator-=(difference_type __n) noexcept(noexcept(__index_ -= __n))
  {
    __index_ -= __n;
    return *this;
  }

  //! @brief Returns the distance between two @c permutation_iterators.
  //! @return Equivalent to ``__lhs.index - __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const permutation_iterator& __lhs, const permutation_iterator& __rhs) noexcept(__nothrow_difference<_Index>)
  {
    return __lhs.__index_ - __rhs.__index();
  }

  //! @brief Compares two @c permutation_iterator for equality by comparing @c index
  //! @return Equivalent to ``__lhs.index == __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_equality<_Index, _OtherOffset>)
  {
    return __lhs.__index_ == __rhs.__index();
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c permutation_iterator for inequality by comparing @c index
  //! @return Equivalent to ``__lhs.index != __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_equality<_Index, _OtherOffset>)
  {
    return !(__lhs.__index_ == __rhs.__index());
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_three_way = noexcept(_CUDA_VSTD::declval<_Iter1>() <=> _CUDA_VSTD::declval<_Iter2>());

  //! @brief Three-way-compares two @c permutation_iterator for inequality by comparing @c index
  //! they point at
  //! @return Equivalent to ``__lhs.index <=> __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::three_way_comparable_with<_Index, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering operator<=>(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_three_way<_Index, _OtherOffset>)
  {
    return __lhs.__index_ <=> __rhs.__index();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  //! @brief Compares two @c permutation_iterator for less than by comparing @c index
  //! @return Equivalent to ``__lhs.index < __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Index, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_less_than<_Index, _OtherOffset>)
  {
    return __lhs.__index_ < __rhs.__index();
  }

  //! @brief Compares two @c permutation_iterator for less equal by comparing @c index
  //! @return Equivalent to ``__lhs.index <= __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Index, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_less_equal<_Index, _OtherOffset>)
  {
    return __lhs.__index_ <= __rhs.__index();
  }

  //! @brief Compares two @c permutation_iterator for greater than by comparing @c index
  //! @return Equivalent to ``__lhs.index > __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Index, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator>(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_greater_than<_Index, _OtherOffset>)
  {
    return __lhs.__index_ > __rhs.__index();
  }

  //! @brief Compares two @c permutation_iterator for greater equal by comparing @c index
  //! @return Equivalent to ``__lhs.index >= __rhs.index``
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Index, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_greater_equal<_Index, _OtherOffset>)
  {
    return __lhs.__index_ >= __rhs.__index();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

_CCCL_TEMPLATE(class _Iter, class _Index)
_CCCL_REQUIRES(
  _CUDA_VSTD::__has_random_access_traversal<_Iter> _CCCL_AND _CUDA_VSTD::__has_random_access_traversal<_Index>)
_CCCL_HOST_DEVICE permutation_iterator(_Iter, _Index) -> permutation_iterator<_Iter, _Index>;

//! @brief Creates an @c permutation_iterator from a base iterator and an iterator to an integral index
//! @param __iter The iterator
//! @param __index The iterator to an integral index
//! @relates permutation_iterator
_CCCL_TEMPLATE(class _Iter, class _Index)
_CCCL_REQUIRES(
  _CUDA_VSTD::__has_random_access_traversal<_Iter> _CCCL_AND _CUDA_VSTD::__has_random_access_traversal<_Index>)
[[nodiscard]] _CCCL_API constexpr permutation_iterator<_Iter, _Index>
make_permutation_iterator(_Iter __iter, _Index __index) noexcept(
  _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Index>)
{
  return permutation_iterator<_Iter, _Index>{__iter, __index};
}

//! @}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_PERMUTATION_ITERATOR_H
