//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_COUNTING_ITERATOR_H
#define _CUDA___ITERATOR_COUNTING_ITERATOR_H

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
#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/semiregular.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/unreachable_sentinel.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @addtogroup iterators
//! @{

//! @cond

template <class _Int>
struct __get_wider_signed
{
  _CCCL_API inline static auto __call() noexcept
  {
    if constexpr (sizeof(_Int) < sizeof(short))
    {
      return _CUDA_VSTD::type_identity<short>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(int))
    {
      return _CUDA_VSTD::type_identity<int>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(long))
    {
      return _CUDA_VSTD::type_identity<long>{};
    }
    else
    {
      return _CUDA_VSTD::type_identity<long long>{};
    }

    static_assert(sizeof(_Int) <= sizeof(long long),
                  "Found integer-like type that is bigger than largest integer like type.");
    _CCCL_UNREACHABLE();
  }

  using type = typename decltype(__call())::type;
};

template <class _Start>
using _IotaDiffT = typename _CUDA_VSTD::conditional_t<
  (!_CUDA_VSTD::integral<_Start> || sizeof(_CUDA_VSTD::iter_difference_t<_Start>) > sizeof(_Start)),
  _CUDA_VSTD::type_identity<_CUDA_VSTD::iter_difference_t<_Start>>,
  __get_wider_signed<_Start>>::type;

template <class _Iter>
_CCCL_CONCEPT __decrementable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __iter)(
  requires(_CUDA_VSTD::incrementable<_Iter>), _Same_as(_Iter&)(--__iter), _Same_as(_Iter)(__iter--));

template <class _Iter>
_CCCL_CONCEPT __advanceable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __iter, const _Iter __j, const _IotaDiffT<_Iter> __n)(
  requires(__decrementable<_Iter>),
  requires(_CUDA_VSTD::totally_ordered<_Iter>),
  _Same_as(_Iter&) __iter += __n,
  _Same_as(_Iter&) __iter -= __n,
  requires(_CUDA_VSTD::is_constructible_v<_Iter, decltype(__j + __n)>),
  requires(_CUDA_VSTD::is_constructible_v<_Iter, decltype(__n + __j)>),
  requires(_CUDA_VSTD::is_constructible_v<_Iter, decltype(__j - __n)>),
  requires(_CUDA_VSTD::convertible_to<decltype(__j - __j), _IotaDiffT<_Iter>>));

template <class, class = void>
struct __counting_iterator_category
{};

template <class _Tp>
struct __counting_iterator_category<_Tp, _CUDA_VSTD::enable_if_t<_CUDA_VSTD::incrementable<_Tp>>>
{
  using iterator_category = _CUDA_VSTD::input_iterator_tag;
};

//! @endcond

//! @brief A @c counting_iterator represents an iterator into a range of sequentially increasing values.
//! @tparam _Start the value type of the @c counting_iterator.
//!
//! This iterator is useful for creating a range filled with a sequence without explicitly storing it in memory. Using
//! @c counting_iterator saves memory capacity and bandwidth.
//!
//! The following code snippet demonstrates how to create a @c counting_iterator whose @c value_type is @c int
//!
//! @code{.cpp}
//! #include <cuda/iterator>
//! ...
//! // create iterators
//! cuda::counting_iterator first(10);
//! cuda::counting_iterator last = first + 3;
//!
//! first[0]   // returns 10
//! first[1]   // returns 11
//! first[100] // returns 110
//!
//! // sum of [first, last)
//! std::reduce(first, last);   // returns 33 (i.e. 10 + 11 + 12)
//!
//! // initialize vector to [0,1,2,..]
//! cuda::counting_iterator iter(0);
//! std::vector<int> vec(500);
//! std::copy(iter, iter + vec.size(), vec.begin());
//! @endcode
#if _CCCL_HAS_CONCEPTS()
template <_CUDA_VSTD::weakly_incrementable _Start>
  requires _CUDA_VSTD::copyable<_Start>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Start,
          _CUDA_VSTD::enable_if_t<_CUDA_VSTD::weakly_incrementable<_Start>, int> = 0,
          _CUDA_VSTD::enable_if_t<_CUDA_VSTD::copyable<_Start>, int>             = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class counting_iterator : public __counting_iterator_category<_Start>
{
private:
  _Start __value_ = _Start();

public:
  using iterator_concept = _CUDA_VSTD::conditional_t<
    __advanceable<_Start>,
    _CUDA_VSTD::random_access_iterator_tag,
    _CUDA_VSTD::conditional_t<__decrementable<_Start>,
                              _CUDA_VSTD::bidirectional_iterator_tag,
                              _CUDA_VSTD::conditional_t<_CUDA_VSTD::incrementable<_Start>,
                                                        _CUDA_VSTD::forward_iterator_tag,
                                                        /*Else*/ _CUDA_VSTD::input_iterator_tag>>>;

  using value_type      = _Start;
  using difference_type = _IotaDiffT<_Start>;

  // Those are technically not to spec, but pre-ranges iterator_traits do not work properly with iterators that do not
  // define all 5 aliases, see https://en.cppreference.com/w/cpp/iterator/iterator_traits.html
  using reference = _Start;
  using pointer   = void;

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI counting_iterator()
    requires _CUDA_VSTD::default_initializable<_Start>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::default_initializable<_Start2>)
  _CCCL_API constexpr counting_iterator() noexcept(_CUDA_VSTD::is_nothrow_default_constructible_v<_Start2>) {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  //! @brief Creates a @c counting_iterator from an initial value.
  //! @param __value The value to store in the @c counting_iterator
  _CCCL_API constexpr explicit counting_iterator(_Start __value) noexcept(
    _CUDA_VSTD::is_nothrow_move_constructible_v<_Start>)
      : __value_(_CUDA_VSTD::move(__value))
  {}

  //! @brief Returns the value currently stored in the @c counting_iterator
  [[nodiscard]] _CCCL_API constexpr _Start operator*() const
    noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Start>)
  {
    return __value_;
  }

  //! @brief Returns the value currently stored in the @c counting_iterator advanced by a number of steps
  //! @param __n The amount of elements to advance
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API constexpr _Start2 operator[](difference_type __n) const
    noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Start2>
             && noexcept(_CUDA_VSTD::declval<const _Start2&>() + __n))
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start>)
    {
      return _Start(__value_ + static_cast<_Start>(__n));
    }
    else
    {
      return _Start(__value_ + __n);
    }
  }

  //! @brief Increments the stored value
  _CCCL_API constexpr counting_iterator& operator++() noexcept(noexcept(++_CUDA_VSTD::declval<_Start&>()))
  {
    ++__value_;
    return *this;
  }

  //! @brief Increments the stored value
  _CCCL_API constexpr auto operator++(int) noexcept(
    noexcept(++_CUDA_VSTD::declval<_Start&>()) && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Start>)
  {
    if constexpr (_CUDA_VSTD::incrementable<_Start>)
    {
      auto __tmp = *this;
      ++__value_;
      return __tmp;
    }
    else
    {
      ++__value_;
    }
  }

  //! @brief Decrements the stored value
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__decrementable<_Start2>)
  _CCCL_API constexpr counting_iterator& operator--() noexcept(noexcept(--_CUDA_VSTD::declval<_Start2&>()))
  {
    --__value_;
    return *this;
  }

  //! @brief Decrements the stored value
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__decrementable<_Start2>)
  _CCCL_API constexpr counting_iterator operator--(int) noexcept(
    noexcept(--_CUDA_VSTD::declval<_Start2&>()) && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Start>)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Increments the stored value by a given number of elements
  //! @param __n The number of elements to increment
  _CCCL_API constexpr counting_iterator& operator+=(difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start>)
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start> && !_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      if (__n >= difference_type(0))
      {
        __value_ += static_cast<_Start>(__n);
      }
      else
      {
        __value_ -= static_cast<_Start>(-__n);
      }
    }
    else if constexpr (_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      __value_ += static_cast<_Start>(__n);
    }
    else
    {
      __value_ += __n;
    }
    return *this;
  }

  //! @brief Creates a copy of a @c counting_iterator advanced by a given number of elements
  //! @param __iter The @c counting_iterator to advance
  //! @param __n The amount of elements to advance
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr counting_iterator
  operator+(counting_iterator __iter, difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    __iter += __n;
    return __iter;
  }

  //! @brief Creates a copy of a @c counting_iterator advanced by a given number of elements
  //! @param __iter The @c counting_iterator to advance
  //! @param __n The amount of elements to advance
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr counting_iterator
  operator+(difference_type __n, counting_iterator __iter) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    return __iter + __n;
  }

  //! @brief Decrements the stored value by a given number of elements
  //! @param __n The amount of elements to decrement
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  _CCCL_API constexpr counting_iterator& operator-=(difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start> && !_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      if (__n >= difference_type(0))
      {
        __value_ -= static_cast<_Start>(__n);
      }
      else
      {
        __value_ += static_cast<_Start>(-__n);
      }
    }
    else if constexpr (_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      __value_ -= static_cast<_Start>(__n);
    }
    else
    {
      __value_ -= __n;
    }
    return *this;
  }

  //! @brief Creates a copy of a @c counting_iterator decremented by a given number of elements
  //! @param __iter The @c counting_iterator to decrement
  //! @param __n The amount of elements to decrement
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr counting_iterator
  operator-(counting_iterator __iter, difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    __iter -= __n;
    return __iter;
  }

  //! @brief Returns the distance between two @c counting_iterator
  //! @return The difference between the stored values
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const counting_iterator& __x, const counting_iterator& __y) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start> && !_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      if (__y.__value_ > __x.__value_)
      {
        return static_cast<difference_type>(-static_cast<difference_type>(__y.__value_ - __x.__value_));
      }
      return static_cast<difference_type>(__x.__value_ - __y.__value_);
    }
    else if constexpr (_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      return static_cast<difference_type>(
        static_cast<difference_type>(__x.__value_) - static_cast<difference_type>(__y.__value_));
    }
    else
    {
      return __x.__value_ - __y.__value_;
    }
    _CCCL_UNREACHABLE();
  }

  //! @brief Compares two @c counting_iterator for equality.
  //! @return True if the stored values compare equal
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() == _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ == __y.__value_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c counting_iterator for inequality.
  //! @return True if the stored values do not compare equal
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() != _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ != __y.__value_;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way compares two @c counting_iterator.
  //! @return The three-way comparison of the stored values
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() <=> _CUDA_VSTD::declval<const _Start2&>()))
    requires _CUDA_VSTD::totally_ordered<_Start> && _CUDA_VSTD::three_way_comparable<_Start>
  {
    return __x.__value_ <=> __y.__value_;
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  //! @brief Compares two @c counting_iterator for less than.
  //! @return True if stored values compare less than
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ < __y.__value_;
  }

  //! @brief Compares two @c counting_iterator for greater than.
  //! @return True if stored values compare greater than
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __y < __x;
  }

  //! @brief Compares two @c counting_iterator for less equal.
  //! @return True if stored values compare less equal
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return !(__y < __x);
  }

  //! @brief Compares two @c counting_iterator for greater equal.
  //! @return True if stored values compare greater equal
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return !(__x < __y);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

//! @brief Creates a @c counting_iterator from an __integer-like__ @c _Start
//! @param __start The __integer-like__ @c _Start representing the initial count
//! @relates counting_iterator
template <class _Start>
[[nodiscard]] _CCCL_API constexpr auto make_counting_iterator(_Start __start)
{
  return counting_iterator<_Start>{__start};
}

//! @} iterators

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_COUNTING_ITERATOR_H
