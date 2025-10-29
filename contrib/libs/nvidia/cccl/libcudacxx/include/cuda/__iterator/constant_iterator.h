//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_CONSTANT_ITERATOR_H
#define _CUDA___ITERATOR_CONSTANT_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/compressed_movable_box.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @addtogroup iterators
//! @{

//! @brief The @c constant_iterator class represents an iterator in an infinite sequence of repeated values.
//! @tparam _Tp the value type of the @c constant_iterator.
//! @tparam _Index The index type of the @c constant_iterator. It can optionally be specified, but must satisfy
//! __integer-like__
//!
//! This iterator is useful for creating a range filled with the same value without explicitly storing it in memory.
//! Using @c constant_iterator saves both memory capacity and bandwidth.
//!
//! The following code snippet demonstrates how to create a @c constant_iterator whose @c value_type is @c int and whose
//! value is @c 10.
//!
//! @code{.cpp}
//! #include <cuda/iterator>
//!
//! cuda::constant_iterator iter(10);
//!
//! *iter;    // returns 10
//! iter[0];  // returns 10
//! iter[1];  // returns 10
//! iter[13]; // returns 10
//!
//! // and so on...
//! @endcode
template <class _Tp, class _Index = _CUDA_VSTD::ptrdiff_t>
class constant_iterator
{
private:
  static_assert(_CUDA_VSTD::__integer_like<_Index>, "The index type of cuda::constant_iterator must be integer-like!");

  // Not a base because then the friend operators would be ambiguous
  ::cuda::std::__compressed_movable_box<_Index, _Tp> __store_;

  [[nodiscard]] _CCCL_API constexpr _Index& __index() noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Index& __index() const noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr _Tp& __value() noexcept
  {
    return __store_.template __get<1>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp& __value() const noexcept
  {
    return __store_.template __get<1>();
  }

public:
  using iterator_concept  = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;

  // Those are technically not to spec, but pre-ranges iterator_traits do not work properly with iterators that do not
  // define all 5 aliases, see https://en.cppreference.com/w/cpp/iterator/iterator_traits.html
  using reference = _Tp;
  using pointer   = void;

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Tp2>)
  _CCCL_API constexpr constant_iterator() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Tp2>)
      : __store_()
  {}

  //! @brief Creates a @c constant_iterator from a value. The index is set to zero
  //! @param __value The value to store in the @c constant_iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr constant_iterator(_Tp __value) noexcept(::cuda::std::is_nothrow_move_constructible_v<_Tp>)
      : __store_(0, ::cuda::std::move(__value))
  {}

  //! @brief Creates @c constant_iterator from a value and an index
  //! @param __value The value to store in the @c constant_iterator
  //! @param __index The index in the sequence represented by this @c constant_iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(typename _Index2)
  _CCCL_REQUIRES(_CUDA_VSTD::__integer_like<_Index2>)
  _CCCL_API constexpr explicit constant_iterator(_Tp __value, _Index2 __index) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Tp>)
      : __store_(static_cast<_Index>(__index), ::cuda::std::move(__value))
  {}

  //! @brief Returns a the current index
  [[nodiscard]] _CCCL_API constexpr difference_type index() const noexcept
  {
    return static_cast<difference_type>(__index());
  }

  //! @brief Returns a const reference to the stored value
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator*() const noexcept
  {
    return __value();
  }

  //! @brief Returns a const reference to the stored value
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](difference_type) const noexcept
  {
    return __value();
  }

  //! @brief Increments the stored index
  _CCCL_API constexpr constant_iterator& operator++() noexcept
  {
    ++__index();
    return *this;
  }

  //! @brief Increments the stored index
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr constant_iterator operator++(int) noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Tp>)
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  //! @brief Decrements the stored index
  _CCCL_API constexpr constant_iterator& operator--() noexcept
  {
    if constexpr (_CUDA_VSTD::is_signed_v<_Index>)
    {
      _CCCL_ASSERT(__index() > 0, "The index must be greater than or equal to 0");
    }
    --__index();
    return *this;
  }

  //! @brief Decrements the stored index
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr constant_iterator operator--(int) noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Tp>)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Advances a @c constant_iterator by a given number of elements
  //! @param __n The amount of elements to advance
  _CCCL_API constexpr constant_iterator& operator+=(difference_type __n) noexcept
  {
    if constexpr (_CUDA_VSTD::is_signed_v<_Index>)
    {
      _CCCL_ASSERT(__index() + __n >= 0, "The index must be greater than or equal to 0");
    }
    __index() += static_cast<_Index>(__n);
    return *this;
  }

  //! @brief Creates a copy of a @c constant_iterator advanced by a given number of elements
  //! @param __iter The @c constant_iterator to advance
  //! @param __n The amount of elements to advance
  [[nodiscard]] _CCCL_API friend constexpr constant_iterator
  operator+(constant_iterator __iter, difference_type __n) noexcept
  {
    __iter += __n;
    return __iter;
  }

  //! @brief Creates a copy of a @c constant_iterator advanced by a given number of elements
  //! @param __n The amount of elements to advance
  //! @param __iter The @c constant_iterator to advance
  [[nodiscard]] _CCCL_API friend constexpr constant_iterator
  operator+(difference_type __n, constant_iterator __iter) noexcept
  {
    __iter += __n;
    return __iter;
  }

  //! @brief Decrements a @c constant_iterator by a given number of elements
  //! @param __n The amount of elements to decrement
  _CCCL_API constexpr constant_iterator& operator-=(difference_type __n) noexcept
  {
    if constexpr (_CUDA_VSTD::is_signed_v<_Index>)
    {
      _CCCL_ASSERT(__index() - __n >= 0, "The index must be greater than or equal to 0");
    }
    __index() -= static_cast<_Index>(__n);
    return *this;
  }

  //! @brief Creates a copy of a @c constant_iterator decremented by a given number of elements
  //! @param __n The amount of elements to decrement
  //! @param __iter The @c constant_iterator to decrement
  [[nodiscard]] _CCCL_API friend constexpr constant_iterator
  operator-(constant_iterator __iter, difference_type __n) noexcept
  {
    __iter -= __n;
    return __iter;
  }

  //! @brief Returns the distance between two @c constant_iterator
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return static_cast<difference_type>(__lhs.__index()) - static_cast<difference_type>(__rhs.__index());
  }

  //! @brief Compares two @c constant_iterator for equality by comparing the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index() == __rhs.__index();
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c constant_iterator for inequality by comparing the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index() != __rhs.__index();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two @c constant_iterator by comparing the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index() <=> __rhs.__index();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  //! @brief Compares two @c constant_iterator for less than by comparing the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index() < __rhs.__index();
  }
  //! @brief Compares two @c constant_iterator for less equal by comparing the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index() <= __rhs.__index();
  }
  //! @brief Compares two @c constant_iterator for greater than by comparing the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index() > __rhs.__index();
  }
  //! @brief Compares two @c constant_iterator for greater equal by comparing the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index() >= __rhs.__index();
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR()
};

template <class _Tp>
_CCCL_HOST_DEVICE constant_iterator(_Tp) -> constant_iterator<_Tp, _CUDA_VSTD::ptrdiff_t>;

_CCCL_TEMPLATE(class _Tp, typename _Index)
_CCCL_REQUIRES(_CUDA_VSTD::__integer_like<_Index>)
_CCCL_HOST_DEVICE constant_iterator(_Tp, _Index) -> constant_iterator<_Tp, _Index>;

//! @brief Creates a @c constant_iterator from a value and an index
//! @param __value The value to be stored
//! @param __index The optional index representing the position in a sequence. Defaults to 0.
//! @relates constant_iterator
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr auto make_constant_iterator(_Tp __value, _CUDA_VSTD::ptrdiff_t __index = 0)
{
  return constant_iterator<_Tp>{_CUDA_VSTD::move(__value), __index};
}

//! @} // end iterators

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_CONSTANT_ITERATOR_H
