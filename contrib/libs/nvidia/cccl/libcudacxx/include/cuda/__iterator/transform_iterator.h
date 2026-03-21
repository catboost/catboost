//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_TRANSFORM_ITERATOR_H
#define _CUDA___ITERATOR_TRANSFORM_ITERATOR_H

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
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/compressed_movable_box.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @addtogroup iterators
//! @{

template <class, class, class = void>
struct __transform_iterator_category_base
{};

template <class _Fn, class _Iter>
struct __transform_iterator_category_base<_Fn, _Iter, _CUDA_VSTD::enable_if_t<_CUDA_VSTD::__has_forward_traversal<_Iter>>>
{
  using _Cat = typename _CUDA_VSTD::iterator_traits<_Iter>::iterator_category;

  using iterator_category = _CUDA_VSTD::conditional_t<
    _CUDA_VSTD::is_reference_v<_CUDA_VSTD::invoke_result_t<_Fn&, _CUDA_VSTD::iter_reference_t<_Iter>>>,
    _CUDA_VSTD::conditional_t<_CUDA_VSTD::derived_from<_Cat, _CUDA_VSTD::contiguous_iterator_tag>,
                              _CUDA_VSTD::random_access_iterator_tag,
                              _Cat>,
    _CUDA_VSTD::input_iterator_tag>;
};

template <class _Fn, class _Iter, bool = (_CUDA_VSTD::__has_random_access_traversal<_Iter>)>
inline constexpr bool __transform_iterator_nothrow_subscript = false;

template <class _Fn, class _Iter>
inline constexpr bool __transform_iterator_nothrow_subscript<_Fn, _Iter, true> =
  noexcept(_CUDA_VSTD::invoke(_CUDA_VSTD::declval<_Fn&>(), _CUDA_VSTD::declval<_Iter&>()[0]));

//! @brief @c transform_iterator is an iterator which represents a pointer into a range of values after transformation
//! by a functor. This iterator is useful for creating a range filled with the result of applying an operation to
//! another range without either explicitly storing it in memory, or explicitly executing the transformation. Using
//! @c transform_iterator facilitates kernel fusion by deferring the execution of a transformation until the value is
//! needed while saving both memory capacity and bandwidth.
//!
//! The following code snippet demonstrates how to create a @c transform_iterator which represents the result of
//! @c sqrtf applied to the contents of a @c thrust::device_vector.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//!
//! struct square_root
//! {
//!   __host__ __device__
//!   float operator()(float x) const
//!   {
//!     return sqrtf(x);
//!   }
//! };
//!
//! int main()
//! {
//!   thrust::device_vector<float> v{1.0f, 4.0f, 9.0f, 16.0f};
//!
//!   using FloatIterator = thrust::device_vector<float>::iterator;
//!
//!   cuda::transform_iterator iter(v.begin(), square_root{});
//!
//!   *iter;   // returns 1.0f
//!   iter[0]; // returns 1.0f;
//!   iter[1]; // returns 2.0f;
//!   iter[2]; // returns 3.0f;
//!   iter[3]; // returns 4.0f;
//!
//!   // iter[4] is an out-of-bounds error
//! }
//! @endcode
//!
//! This next example demonstrates how to use a @c transform_iterator with the @c thrust::reduce functor to compute the
//! sum of squares of a sequence. We will create temporary @c transform_iterators utilising class template argument
//! deduction avoid explicitly specifying their type:
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//! #include <thrust/reduce.h>
//! #include <iostream>
//!
//! struct square
//! {
//!   __host__ __device__
//!   float operator()(float x) const
//!   {
//!     return x * x;
//!   }
//! };
//!
//! int main()
//! {
//!   // initialize a device array
//!   thrust::device_vector<float> v(4);
//!   v[0] = 1.0f;
//!   v[1] = 2.0f;
//!   v[2] = 3.0f;
//!   thrust::device_vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
//!   thrust::reduce(cuda::transform_iterator{v.begin(), square{}},
//!                  cuda::transform_iterator{v.end(),   square{}});
//!
//!   std::cout << "sum of squares: " << sum_of_squares << std::endl;
//!   return 0;
//! }
//! @endcode
template <class _Fn, class _Iter>
class transform_iterator : public __transform_iterator_category_base<_Fn, _Iter>
{
  static_assert(_CUDA_VSTD::is_object_v<_Fn>, "cuda::transform_iterator requires that _Fn is a functor object");
  static_assert(_CUDA_VSTD::regular_invocable<_Fn&, _CUDA_VSTD::iter_reference_t<_Iter>>,
                "cuda::transform_iterator requires that _Fn is invocable with iter_reference_t<_Iter>");
  static_assert(_CUDA_VSTD::__can_reference<_CUDA_VSTD::invoke_result_t<_Fn&, _CUDA_VSTD::iter_reference_t<_Iter>>>,
                "cuda::transform_iterator requires that the return type of _Fn is referenceable");

  // Not a base because then the friend operators would be ambiguous
  ::cuda::std::__compressed_movable_box<_Iter, _Fn> __store_;

  [[nodiscard]] _CCCL_API constexpr _Iter& __iter() noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Iter& __iter() const noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr _Fn& __func() noexcept
  {
    return __store_.template __get<1>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Fn& __func() const noexcept
  {
    return __store_.template __get<1>();
  }

public:
  using iterator_concept = ::cuda::std::conditional_t<
    ::cuda::std::__has_random_access_traversal<_Iter>,
    ::cuda::std::random_access_iterator_tag,
    ::cuda::std::conditional_t<::cuda::std::__has_bidirectional_traversal<_Iter>,
                               ::cuda::std::bidirectional_iterator_tag,
                               ::cuda::std::conditional_t<::cuda::std::__has_forward_traversal<_Iter>,
                                                          ::cuda::std::forward_iterator_tag,
                                                          ::cuda::std::input_iterator_tag>>>;
  using value_type =
    ::cuda::std::remove_cvref_t<::cuda::std::invoke_result_t<_Fn&, ::cuda::std::iter_reference_t<_Iter>>>;
  using difference_type = ::cuda::std::iter_difference_t<_Iter>;

  // Those are technically not to spec, but pre-ranges iterator_traits do not work properly with iterators that do not
  // define all 5 aliases, see https://en.cppreference.com/w/cpp/iterator/iterator_traits.html
  using reference = ::cuda::std::invoke_result_t<_Fn&, ::cuda::std::iter_reference_t<_Iter>>;
  using pointer   = void;

  //! @brief Default constructs a @c transform_iterator with a value initialized iterator and functor
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter, class _Fn2 = _Fn)
  _CCCL_REQUIRES(_CUDA_VSTD::default_initializable<_Iter2> _CCCL_AND _CUDA_VSTD::default_initializable<_Fn2>)
  _CCCL_API constexpr transform_iterator() noexcept(
    ::cuda::std::is_nothrow_default_constructible_v<_Iter2> && ::cuda::std::is_nothrow_default_constructible_v<_Fn2>)
      : __store_()
  {}

  //! @brief Constructs a @c transform_iterator with a given iterator and functor
  //! @param __iter The iterator to transform
  //! @param __func The functor to apply to the iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_iterator(_Iter __iter, _Fn __func) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Iter> && ::cuda::std::is_nothrow_move_constructible_v<_Fn>)
      : __store_(::cuda::std::move(__iter), ::cuda::std::move(__func))
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

  //! @brief Dereferences the stored iterator and applies the stored functor to the result
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::regular_invocable<const _Fn&, _CUDA_VSTD::iter_reference_t<const _Iter2>>)
  [[nodiscard]] _CCCL_API constexpr reference operator*() const
    noexcept(noexcept(::cuda::std::invoke(::cuda::std::declval<const _Fn&>(), *::cuda::std::declval<const _Iter2&>())))
  {
    return ::cuda::std::invoke(__func(), *__iter());
  }

  //! @cond
  //! @brief Dereferences the stored iterator and applies the stored functor to the result
  //! @note This is a cludge against the fact that the iterator concepts requires `const Iter` but a user might have
  //! forgotten to const qualify the call operator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES((!_CUDA_VSTD::regular_invocable<const _Fn&, _CUDA_VSTD::iter_reference_t<const _Iter2>>) )
  [[nodiscard]] _CCCL_API constexpr reference operator*() const
    noexcept(noexcept(::cuda::std::invoke(::cuda::std::declval<_Fn&>(), *::cuda::std::declval<const _Iter2&>())))
  {
    return ::cuda::std::invoke(const_cast<_Fn&>(__func()), *__iter());
  }
  //! @endcond

  //! @brief Dereferences the stored iterator and applies the stored functor to the result
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr reference
  operator*() noexcept(noexcept(::cuda::std::invoke(::cuda::std::declval<_Fn&>(), *::cuda::std::declval<_Iter&>())))
  {
    return ::cuda::std::invoke(__func(), *__iter());
  }

  //! @brief Subscripts the stored iterator by a number of elements and applies the stored functor to the result
  //! @param __n The number of elements to advance by
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__has_random_access_traversal<_Iter2> _CCCL_AND
                   _CUDA_VSTD::regular_invocable<const _Fn&, _CUDA_VSTD::iter_reference_t<const _Iter2>>)
  [[nodiscard]] _CCCL_API constexpr reference operator[](difference_type __n) const
    noexcept(__transform_iterator_nothrow_subscript<const _Fn, _Iter2>)
  {
    return ::cuda::std::invoke(__func(), __iter()[__n]);
  }

  //! @cond
  //! @brief Subscripts the stored iterator by a number of elements and applies the stored functor to the result
  //! @param __n The number of elements to advance by
  //! @note This is a cludge against the fact that the iterator concepts requires `const Iter` but a user might have
  //! forgotten to const qualify the call operator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__has_random_access_traversal<_Iter2> _CCCL_AND(
    !_CUDA_VSTD::regular_invocable<const _Fn&, _CUDA_VSTD::iter_reference_t<const _Iter2>>))
  [[nodiscard]] _CCCL_API constexpr reference operator[](difference_type __n) const
    noexcept(__transform_iterator_nothrow_subscript<_Fn, _Iter2>)
  {
    return ::cuda::std::invoke(const_cast<_Fn&>(__func()), __iter()[__n]);
  }
  //! @endcond

  //! @brief Subscripts the stored iterator by a number of elements and applies the stored functor to the result
  //! @param __n The number of elements to advance by
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  [[nodiscard]] _CCCL_API constexpr reference
  operator[](difference_type __n) noexcept(__transform_iterator_nothrow_subscript<_Fn, _Iter2>)
  {
    return ::cuda::std::invoke(__func(), __iter()[__n]);
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_iterator& operator++() noexcept(noexcept(++::cuda::std::declval<_Iter&>()))
  {
    ++__iter();
    return *this;
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr auto operator++(int) noexcept(noexcept(++::cuda::std::declval<_Iter&>()))
  {
    if constexpr (_CUDA_VSTD::__has_forward_traversal<_Iter>)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }
    else
    {
      ++__iter();
    }
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__has_bidirectional_traversal<_Iter2>)
  _CCCL_API constexpr transform_iterator& operator--() noexcept(noexcept(--::cuda::std::declval<_Iter2&>()))
  {
    --__iter();
    return *this;
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__has_bidirectional_traversal<_Iter2>)
  _CCCL_API constexpr transform_iterator operator--(int) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && noexcept(--::cuda::std::declval<_Iter2&>()))
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Increments the @c transform_iterator by a given number of elements
  //! @param __n The number of elements to increment
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__has_random_access_traversal<_Iter2>)
  _CCCL_API constexpr transform_iterator&
  operator+=(difference_type __n) noexcept(noexcept(::cuda::std::declval<_Iter2&>() += __n))
  {
    __iter() += __n;
    return *this;
  }

  //! @brief Returns a copy of a @c transform_iterator advanced by a given number of elements
  //! @param __iter The @c transform_iterator to advance
  //! @param __n The amount of elements to increment
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto operator+(const transform_iterator& __iter, difference_type __n)
    _CCCL_TRAILING_REQUIRES(transform_iterator)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return transform_iterator{__iter.__iter() + __n, __iter.__func()};
  }

  //! @brief Returns a copy of a @c transform_iterator advanced by a given number of elements
  //! @param __n The amount of elements to increment
  //! @param __iter The @c transform_iterator to advance
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto operator+(difference_type __n, const transform_iterator& __iter)
    _CCCL_TRAILING_REQUIRES(transform_iterator)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return transform_iterator{__iter.__iter() + __n, __iter.__func()};
  }

  //! @brief Decrements the @c transform_iterator by a given number of elements
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__has_random_access_traversal<_Iter2>)
  _CCCL_API constexpr transform_iterator&
  operator-=(difference_type __n) noexcept(noexcept(::cuda::std::declval<_Iter2&>() -= __n))
  {
    __iter() -= __n;
    return *this;
  }

  //! @brief Returns a copy of a @c transform_iterator decremented by a given number of elements
  //! @param __iter The @c transform_iterator to decrement
  //! @param __n The amount of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto operator-(const transform_iterator& __iter, difference_type __n)
    _CCCL_TRAILING_REQUIRES(transform_iterator)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return transform_iterator{__iter.__iter() - __n, __iter.__func()};
  }

  template <class _Iter2>
  static constexpr bool __can_difference =
    (::cuda::std::__has_random_access_traversal<_Iter2> || ::cuda::std::sized_sentinel_for<_Iter2, _Iter2>);

  template <class _Iter2>
  static constexpr bool __noexcept_difference =
    noexcept(::cuda::std::declval<const _Iter2&>() - ::cuda::std::declval<const _Iter2&>());

  //! @brief Returns the distance between two @c transform_iterator
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator-(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(__noexcept_difference<_Iter2>)
    _CCCL_TRAILING_REQUIRES(difference_type)(__can_difference<_Iter2>)
  {
    return __lhs.__iter() - __rhs.__iter();
  }

  //! @brief Compares two @c transform_iterator for equality by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator==(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() == _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::equality_comparable<_Iter2>)
  {
    return __lhs.__iter() == __rhs.__iter();
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c transform_iterator for inequality by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator!=(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() != _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::equality_comparable<_Iter2>)
  {
    return __lhs.__iter() != __rhs.__iter();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two @c transform_iterator, directly three-way-comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() <=> _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(
      _CUDA_VSTD::__has_random_access_traversal<_Iter2>&& _CUDA_VSTD::three_way_comparable<_Iter2>)
  {
    return __lhs.__iter() <=> __rhs.__iter();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  //! @brief Compares two @c transform_iterator for less than by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() < __rhs.__iter();
  }

  //! @brief Compares two @c transform_iterator for greater than by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() > __rhs.__iter();
  }

  //! @brief Compares two @c transform_iterator for less equal by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() <= __rhs.__iter();
  }

  //! @brief Compares two @c transform_iterator for greater equal by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>=(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() >= __rhs.__iter();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

//! @brief Creates a @c transform_iterator from a base iterator and a functor
//! @param __iter The iterator of the input range
//! @param __fun The functor used to transform the input range
//! @relates transform_iterator
template <class _Fn, class _Iter>
[[nodiscard]] _CCCL_API constexpr auto make_transform_iterator(_Iter __iter, _Fn __fun)
{
  return transform_iterator<_Fn, _Iter>{__iter, __fun};
}

//! @}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_TRANSFORM_ITERATOR_H
