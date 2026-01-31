//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_STRIDED_ITERATOR_H
#define _CUDA___ITERATOR_STRIDED_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__mdspan/submdspan_helper.h>
#include <cuda/std/__ranges/compressed_movable_box.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @addtogroup iterators
//! @{

//! @brief A @c strided_iterator wraps another iterator and advances it by a specified stride each time it is
//! incremented or decremented.
//!
//! @tparam _Iter A random access iterator
//! @tparam _Stride Either an <a href="https://eel.is/c++draft/iterator.concept.winc#4">integer-like</a> or an
//! <a href="https://eel.is/c++draft/views.contiguous#concept:integral-constant-like">integral-constant-like</a>
//! specifying the stride
template <class _Iter, class _Stride = _CUDA_VSTD::iter_difference_t<_Iter>>
class strided_iterator
{
private:
  static_assert(_CUDA_VSTD::__has_random_access_traversal<_Iter>,
                "The iterator underlying a strided_iterator must be a random access iterator.");
  static_assert(_CUDA_VSTD::__integer_like<_Stride> || _CUDA_VSTD::__integral_constant_like<_Stride>,
                "The stride of a strided_iterator must either be an integer-like or integral-constant-like.");

  template <class, class>
  friend class strided_iterator;

  // Not a base because then the friend operators would be ambiguous
  ::cuda::std::__compressed_movable_box<_Iter, _Stride> __store_;

  [[nodiscard]] _CCCL_API constexpr _Iter& __iter() noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Iter& __iter() const noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr _Stride& __stride() noexcept
  {
    return __store_.template __get<1>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Stride& __stride() const noexcept
  {
    return __store_.template __get<1>();
  }

public:
  using iterator_concept  = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _CUDA_VSTD::iter_value_t<_Iter>;
  using difference_type   = _CUDA_VSTD::iter_difference_t<_Iter>;

  // Those are technically not to spec, but pre-ranges iterator_traits do not work properly with iterators that do not
  // define all 5 aliases, see https://en.cppreference.com/w/cpp/iterator/iterator_traits.html
  using reference = ::cuda::std::iter_reference_t<_Iter>;
  using pointer   = void;

  //! @brief value-initializes both the base iterator and stride
  //! @note _Iter must be default initializable because it is a random_access_iterator and thereby semiregular
  //!       _Stride must be integer-like or integral_constant_like which requires default constructability
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter, class _Stride2 = _Stride)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Iter2> _CCCL_AND ::cuda::std::default_initializable<_Stride2>)
  _CCCL_API constexpr strided_iterator() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Iter2>
                                                  && ::cuda::std::is_nothrow_default_constructible_v<_Stride2>)
      : __store_()
  {}

  //! @brief Constructs a @c strided_iterator from a base iterator
  //! @param __iter The base iterator
  //! @note We cannot construct a @c strided_iterator with an
  //! <a href="https://eel.is/c++draft/iterator.concept.winc#4">integer-like</a> stride, because that would value
  //! construct to 0 and incrementing the iterator would do nothing.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Stride2 = _Stride)
  _CCCL_REQUIRES(_CUDA_VSTD::__integral_constant_like<_Stride2>)
  _CCCL_API constexpr explicit strided_iterator(_Iter __iter) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Iter> && ::cuda::std::is_nothrow_default_constructible_v<_Stride2>)
      : __store_(::cuda::std::move(__iter))
  {}

  //! @brief Constructs a @c strided_iterator from a base iterator and a stride
  //! @param __iter The base iterator
  //! @param __stride The new stride
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr explicit strided_iterator(_Iter __iter, _Stride __stride) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Iter> && ::cuda::std::is_nothrow_move_constructible_v<_Stride>)
      : __store_(::cuda::std::move(__iter), ::cuda::std::move(__stride))
  {}

  //! @brief Returns a const reference to the stored iterator
  [[nodiscard]] _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __iter();
  }

  //! @brief Extracts the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Iter base() && noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Iter>)
  {
    return ::cuda::std::move(__iter());
  }

  static constexpr bool __noexcept_stride =
    noexcept(static_cast<difference_type>(_CUDA_VSTD::__de_ice(_CUDA_VSTD::declval<const _Stride&>())));

  //! @brief Returns the current stride as an integral value
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr difference_type stride() const noexcept(__noexcept_stride)
  {
    return static_cast<difference_type>(::cuda::std::__de_ice(__stride()));
  }

  //! @brief Dereferences the stored base iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() noexcept(noexcept(*::cuda::std::declval<_Iter&>()))
  {
    return *__iter();
  }

  //! @brief Dereferences the stored base iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__dereferenceable<const _Iter2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() const
    noexcept(noexcept(*::cuda::std::declval<const _Iter2&>()))
  {
    return *__iter();
  }

  //! @brief Subscripts the stored base iterator with a given offset times the stride
  //! @param __n The offset
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  operator[](difference_type __n) noexcept(__noexcept_stride && noexcept(::cuda::std::declval<_Iter&>()[__n]))
  {
    return __iter()[__n * stride()];
  }

  //! @brief Subscripts the stored base iterator with a given offset times the stride
  //! @param __n The offset
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _Iter2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator[](difference_type __n) const
    noexcept(__noexcept_stride && noexcept(::cuda::std::declval<const _Iter2&>()[__n]))
  {
    return __iter()[__n * stride()];
  }

  //! @brief Increments the stored base iterator by the stride
  // Note: we cannot use __iter() += stride() in the noexcept clause because that breaks gcc < 9
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr strided_iterator&
  operator++() noexcept(__noexcept_stride && noexcept(::cuda::std::declval<_Iter&>() += 1))
  {
    __iter() += stride();
    return *this;
  }

  //! @brief Increments the stored base iterator by the stride
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr auto operator++(int) noexcept(
    noexcept(__noexcept_stride && noexcept(::cuda::std::declval<_Iter&>() += 1))
    && ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && ::cuda::std::is_nothrow_copy_constructible_v<_Stride>)
  {
    auto __tmp = *this;
    __iter() += stride();
    return __tmp;
  }

  //! @brief Decrements the stored base iterator by the stride
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr strided_iterator&
  operator--() noexcept(__noexcept_stride && noexcept(::cuda::std::declval<_Iter&>() -= 1))
  {
    __iter() -= stride();
    return *this;
  }

  //! @brief Decrements the stored base iterator by the stride
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr strided_iterator operator--(int) noexcept(
    noexcept(__noexcept_stride && noexcept(::cuda::std::declval<_Iter&>() -= 1))
    && ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && ::cuda::std::is_nothrow_copy_constructible_v<_Stride>)
  {
    auto __tmp = *this;
    __iter() -= stride();
    return __tmp;
  }

  //! @brief Advances a @c strided_iterator by a given number of steps
  //! @param __n The number of steps to increment
  //! @note Increments the base iterator by @c __n times the stride
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr strided_iterator&
  operator+=(difference_type __n) noexcept(__noexcept_stride && noexcept(::cuda::std::declval<_Iter&>() += 1))
  {
    __iter() += stride() * __n;
    return *this;
  }

  //! @brief Returns a copy of a @c strided_iterator incremented by a given number of steps
  //! @param __iter The @c strided_iterator to advance
  //! @param __n The number of steps to increment
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr strided_iterator
  operator+(const strided_iterator& __iter, difference_type __n) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && noexcept(::cuda::std::declval<const _Iter&>() + __n))
  {
    return strided_iterator{__iter.__iter() + __iter.stride() * __n, __iter.__stride()};
  }

  //! @brief Returns a copy of a @c strided_iterator incremented by a given number of steps
  //! @param __n The number of steps to increment
  //! @param __iter The @c strided_iterator to advance
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr strided_iterator
  operator+(difference_type __n, const strided_iterator& __iter) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && noexcept(::cuda::std::declval<const _Iter&>() + __n))
  {
    return strided_iterator{__iter.__iter() + __iter.stride() * __n, __iter.__stride()};
  }

  //! @brief Decrements a @c strided_iterator by a given number of steps
  //! @param __n The number of steps to decrement
  //! @note Decrements the base iterator by @c __n times the stride
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr strided_iterator&
  operator-=(difference_type __n) noexcept(__noexcept_stride && noexcept(::cuda::std::declval<_Iter&>() -= 1))
  {
    __iter() -= stride() * __n;
    return *this;
  }

  //! @brief Returns a copy of a @c strided_iterator decremented by a given number of steps
  //! @param __n The number of steps to decrement
  //! @param __iter The @c strided_iterator to decrement
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr strided_iterator
  operator-(const strided_iterator& __iter, difference_type __n) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && noexcept(::cuda::std::declval<const _Iter&>() - __n))
  {
    return strided_iterator{__iter.__iter() - __iter.stride() * __n, __iter.__stride()};
  }

  template <class _Iter2, class _OtherIter>
  static constexpr bool __noexcept_difference =
    noexcept(::cuda::std::declval<const _Iter2&>() - ::cuda::std::declval<const _OtherIter&>());

  //! @brief Returns distance between two @c strided_iterator's in units of the stride
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::sized_sentinel_for<_OtherIter, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) //
    noexcept(__noexcept_difference<_Iter, _OtherIter>)
  {
    const difference_type __diff = __x.__iter() - __y.base();
    _CCCL_ASSERT(__x.stride() == __y.stride(), "Taking the difference of two strided_iterators with different stride");
    _CCCL_ASSERT(__diff % __x.stride() == 0, "Underlying iterator difference must be divisible by the stride");
    return __diff / __x.stride();
  }

  //! @brief Compares two @c strided_iterator's for equality by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() == _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter() == __y.base();
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c strided_iterator's for inequality by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() == _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter() != __y.base();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Threeway-compares two @c strided_iterator's by comparing the stored iterators
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  _CCCL_REQUIRES(
    _CUDA_VSTD::totally_ordered<_Iter, _OtherIter> _CCCL_AND _CUDA_VSTD::three_way_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() <=> _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter() <=> __y.base();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  //! @brief Compares two @c strided_iterator's for less than by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter() < __y.base();
  }

  //! @brief Compares two @c strided_iterator's for greater than by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __y < __x;
  }

  //! @brief Compares two @c strided_iterator's for less equal by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return !(__y < __x);
  }

  //! @brief Compares two @c strided_iterator's for greater equal by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return !(__x < __y);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

template <class _Iter, typename _Stride>
_CCCL_HOST_DEVICE strided_iterator(_Iter, _Stride) -> strided_iterator<_Iter, _Stride>;

//! @brief Creates a @c strided_iterator from a random access iterator
//! @param __iter The random_access iterator
//! @relates strided_iterator
_CCCL_TEMPLATE(class _Stride, class _Iter)
_CCCL_REQUIRES(_CUDA_VSTD::__integral_constant_like<_Stride>)
[[nodiscard]] _CCCL_API constexpr auto make_strided_iterator(_Iter __iter)
{
  return strided_iterator<_Iter, _Stride>{_CUDA_VSTD::move(__iter)};
}

//! @brief Creates a @c strided_iterator from a random access iterator and a stride
//! @param __iter The random_access iterator
//! @param __stride The new stride
//! @relates strided_iterator
template <class _Iter, class _Stride>
[[nodiscard]] _CCCL_API constexpr auto make_strided_iterator(_Iter __iter, _Stride __stride)
{
  return strided_iterator<_Iter, _Stride>{_CUDA_VSTD::move(__iter), __stride};
}

//! @} // end iterators

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_STRIDED_ITERATOR_H
