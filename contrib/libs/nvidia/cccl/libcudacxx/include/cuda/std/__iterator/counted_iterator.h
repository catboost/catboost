// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_COUNTED_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_COUNTED_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/common_with.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/add_pointer.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/exception_guard.h>
#include <cuda/std/__utility/move.h>

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/detail/libcxx/include/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class, class = void>
struct __counted_iterator_concept
{};

template <class _Iter>
struct __counted_iterator_concept<_Iter, void_t<typename _Iter::iterator_concept>>
{
  using iterator_concept = typename _Iter::iterator_concept;
};

template <class, class = void>
struct __counted_iterator_category
{};

template <class _Iter>
struct __counted_iterator_category<_Iter, void_t<typename _Iter::iterator_category>>
{
  using iterator_category = typename _Iter::iterator_category;
};

template <class, class = void>
struct __counted_iterator_value_type
{};

template <class _Iter>
struct __counted_iterator_value_type<_Iter, enable_if_t<indirectly_readable<_Iter>>>
{
  using value_type = iter_value_t<_Iter>;
};

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

#if _CCCL_HAS_CONCEPTS()
template <input_or_output_iterator _Iter>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Iter, enable_if_t<input_or_output_iterator<_Iter>, int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class counted_iterator
    : public __counted_iterator_concept<_Iter>
    , public __counted_iterator_category<_Iter>
    , public __counted_iterator_value_type<_Iter>
{
public:
  _CCCL_NO_UNIQUE_ADDRESS _Iter __current_ = _Iter();
  iter_difference_t<_Iter> __count_        = 0;

  using iterator_type   = _Iter;
  using difference_type = iter_difference_t<_Iter>;

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI counted_iterator()
    requires default_initializable<_Iter>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(default_initializable<_I2>)
  _CCCL_API constexpr counted_iterator() noexcept(is_nothrow_default_constructible_v<_I2>) {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  _CCCL_API constexpr counted_iterator(_Iter __iter,
                                       iter_difference_t<_Iter> __n) noexcept(is_nothrow_move_constructible_v<_Iter>)
      : __current_(_CUDA_VSTD::move(__iter))
      , __count_(__n)
  {
    _CCCL_ASSERT(__n >= 0, "__n must not be negative.");
  }

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(convertible_to<const _I2&, _Iter>)
  _CCCL_API constexpr counted_iterator(const counted_iterator<_I2>& __other) noexcept(
    is_nothrow_constructible_v<_Iter, const _I2&>)
      : __current_(__other.__current_)
      , __count_(__other.__count_)
  {}

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(assignable_from<_Iter&, const _I2&>)
  _CCCL_API constexpr counted_iterator&
  operator=(const counted_iterator<_I2>& __other) noexcept(is_nothrow_assignable_v<_Iter&, const _I2&>)
  {
    __current_ = __other.__current_;
    __count_   = __other.__count_;
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __current_;
  }

  [[nodiscard]] _CCCL_API constexpr _Iter base() &&
  {
    return _CUDA_VSTD::move(__current_);
  }

  [[nodiscard]] _CCCL_API constexpr iter_difference_t<_Iter> count() const noexcept
  {
    return __count_;
  }

  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*()
  {
    _CCCL_ASSERT(__count_ > 0, "Iterator is equal to or past end.");
    return *__current_;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(__dereferenceable<const _I2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() const
  {
    _CCCL_ASSERT(__count_ > 0, "Iterator is equal to or past end.");
    return *__current_;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(contiguous_iterator<_I2>)
  [[nodiscard]] _CCCL_API constexpr auto operator->() const noexcept
  {
    return _CUDA_VSTD::to_address(__current_);
  }

  _CCCL_API constexpr counted_iterator& operator++()
  {
    _CCCL_ASSERT(__count_ > 0, "Iterator already at or past end.");
    ++__current_;
    --__count_;
    return *this;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES((!forward_iterator<_I2>) )
  _CCCL_API inline decltype(auto) operator++(int)
  {
    _CCCL_ASSERT(__count_ > 0, "Iterator already at or past end.");
    --__count_;
#if _CCCL_HAS_EXCEPTIONS()
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        try { return __current_++; } catch (...) {
          ++__count_;
          throw;
        }),
      (return __current_++;))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
    return __current_++;
#endif // !_CCCL_HAS_EXCEPTIONS()
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(forward_iterator<_I2>)
  _CCCL_API constexpr counted_iterator operator++(int)
  {
    _CCCL_ASSERT(__count_ > 0, "Iterator already at or past end.");
    counted_iterator __tmp = *this;
    ++*this;
    return __tmp;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(bidirectional_iterator<_I2>)
  _CCCL_API constexpr counted_iterator& operator--()
  {
    --__current_;
    ++__count_;
    return *this;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(bidirectional_iterator<_I2>)
  _CCCL_API constexpr counted_iterator operator--(int)
  {
    counted_iterator __tmp = *this;
    --*this;
    return __tmp;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(random_access_iterator<_I2>)
  [[nodiscard]] _CCCL_API constexpr counted_iterator operator+(iter_difference_t<_I2> __n) const
  {
    return counted_iterator(__current_ + __n, __count_ - __n);
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(random_access_iterator<_I2>)
  [[nodiscard]] _CCCL_API friend constexpr counted_iterator
  operator+(iter_difference_t<_I2> __n, const counted_iterator& __x)
  {
    return __x + __n;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(random_access_iterator<_I2>)
  _CCCL_API constexpr counted_iterator& operator+=(iter_difference_t<_I2> __n)
  {
    _CCCL_ASSERT(__n <= __count_, "Cannot advance iterator past end.");
    __current_ += __n;
    __count_ -= __n;
    return *this;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(random_access_iterator<_I2>)
  [[nodiscard]] _CCCL_API constexpr counted_iterator operator-(iter_difference_t<_I2> __n) const
  {
    return counted_iterator(__current_ - __n, __count_ + __n);
  }

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_I2>
  operator-(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __rhs.__count_ - __lhs.__count_;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(random_access_iterator<_I2>)
  [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_I2>
  operator-(const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __rhs.__count_ - __lhs.__count_;
  }

  [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_Iter>
  operator-(const counted_iterator& __lhs, default_sentinel_t)
  {
    return -__lhs.__count_;
  }

  [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_Iter>
  operator-(default_sentinel_t, const counted_iterator& __rhs)
  {
    return __rhs.__count_;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(random_access_iterator<_I2>)
  _CCCL_API constexpr counted_iterator& operator-=(iter_difference_t<_I2> __n)
  {
    _CCCL_ASSERT(-__n <= __count_,
                 "Attempt to subtract too large of a size: "
                 "counted_iterator would be decremented before the "
                 "first element of its range.");
    __current_ -= __n;
    __count_ += __n;
    return *this;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(random_access_iterator<_I2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator[](iter_difference_t<_I2> __n) const
  {
    _CCCL_ASSERT(__n < __count_, "Subscript argument must be less than size.");
    return __current_[__n];
  }

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs) noexcept
  {
    return __lhs.__count_ == __rhs.__count_;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const counted_iterator& __lhs, const counted_iterator& __rhs) noexcept
  {
    return __lhs.__count_ == __rhs.__count_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const counted_iterator& __lhs, const counted_iterator& __rhs) noexcept
  {
    return __lhs.__count_ != __rhs.__count_;
  }

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs) noexcept
  {
    return __lhs.__count_ != __rhs.__count_;
  }
#endif // _CCCL_STD_VER <= 2017

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const counted_iterator& __lhs, default_sentinel_t) noexcept
  {
    return __lhs.__count_ == 0;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(default_sentinel_t, const counted_iterator& __lhs) noexcept
  {
    return __lhs.__count_ == 0;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const counted_iterator& __lhs, default_sentinel_t) noexcept
  {
    return __lhs.__count_ != 0;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(default_sentinel_t, const counted_iterator& __lhs) noexcept
  {
    return __lhs.__count_ != 0;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering
  operator<=>(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs) noexcept
  {
    return __rhs.__count_ <=> __lhs.__count_;
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs) noexcept
  {
    return __lhs.__count_ < __rhs.__count_;
  }

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs) noexcept
  {
    return __lhs.__count_ <= __rhs.__count_;
  }

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs) noexcept
  {
    return __lhs.__count_ > __rhs.__count_;
  }

  _CCCL_TEMPLATE(class _I2)
  _CCCL_REQUIRES(common_with<_I2, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs) noexcept
  {
    return __lhs.__count_ >= __rhs.__count_;
  }

  template <class _I2>
  _CCCL_API friend constexpr auto iter_swap(const counted_iterator& __x, const counted_iterator<_I2>& __y) noexcept(
    noexcept(_CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_)))
    _CCCL_TRAILING_REQUIRES(void)(indirectly_swappable<_I2, _Iter>)
  {
    _CCCL_ASSERT(__x.__count_ > 0 && __y.__count_ > 0, "Iterators must not be past end of range.");
    return _CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_);
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(counted_iterator);
_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(counted_iterator)

// Not a hidden friend because of MSVC
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(input_iterator<_Iter>)
[[nodiscard]] _CCCL_API constexpr decltype(auto) iter_move(const counted_iterator<_Iter>& __i) noexcept(
  noexcept(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<const _Iter&>())))
{
  _CCCL_ASSERT(__i.count() > 0, "Iterator must not be past end of range.");
  return _CUDA_VRANGES::iter_move(__i.base());
}

#if _CCCL_HAS_CONCEPTS()
template <input_iterator _Iter>
  requires same_as<_ITER_TRAITS<_Iter>, iterator_traits<_Iter>>
struct iterator_traits<counted_iterator<_Iter>> : iterator_traits<_Iter>
{
  using pointer = conditional_t<contiguous_iterator<_Iter>, add_pointer_t<iter_reference_t<_Iter>>, void>;
};
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Iter>
struct iterator_traits<counted_iterator<_Iter>,
                       enable_if_t<input_iterator<_Iter> && same_as<_ITER_TRAITS<_Iter>, iterator_traits<_Iter>>>>
    : iterator_traits<_Iter>
{
  using pointer = conditional_t<contiguous_iterator<_Iter>, add_pointer_t<iter_reference_t<_Iter>>, void>;
};

// In C++17 we end up in an infinite recursion trying to determine the return type of `to_address`
template <class _Iter>
struct pointer_traits<counted_iterator<_Iter>, enable_if_t<contiguous_iterator<_Iter>>>
{
  using pointer         = counted_iterator<_Iter>;
  using element_type    = typename pointer_traits<_Iter>::element_type;
  using difference_type = typename pointer_traits<_Iter>::difference_type;

  _CCCL_API static constexpr auto to_address(const pointer __iter) noexcept
  {
    return _CUDA_VSTD::to_address(__iter.base());
  }
};
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_COUNTED_ITERATOR_H
