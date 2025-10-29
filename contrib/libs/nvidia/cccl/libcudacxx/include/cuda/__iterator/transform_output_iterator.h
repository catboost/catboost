//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_TRANSFORM_OUTPUT_ITERATOR_H
#define _CUDA___ITERATOR_TRANSFORM_OUTPUT_ITERATOR_H

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
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
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

template <class _Fn, class _Iter>
class __transform_output_proxy
{
private:
  template <class, class>
  friend class transform_output_iterator;

  _Iter __iter_;
  _Fn& __func_;

  template <class _MaybeConstFn, class _Arg>
  using _Ret = _CUDA_VSTD::invoke_result_t<_MaybeConstFn&, _Arg>;

public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr explicit __transform_output_proxy(_Iter __iter, _Fn& __func) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter>)
      : __iter_(__iter)
      , __func_(__func)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(
    (!_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Arg>, __transform_output_proxy>)
      _CCCL_AND _CUDA_VSTD::is_invocable_v<_Fn&, _Arg> _CCCL_AND
        _CUDA_VSTD::is_assignable_v<_CUDA_VSTD::iter_reference_t<_Iter>, _CUDA_VSTD::invoke_result_t<_Fn&, _Arg>>)
  _CCCL_API constexpr __transform_output_proxy&
  operator=(_Arg&& __arg) noexcept(noexcept(*__iter_ = _CUDA_VSTD::invoke(__func_, _CUDA_VSTD::forward<_Arg>(__arg))))
  {
    *__iter_ = _CUDA_VSTD::invoke(__func_, _CUDA_VSTD::forward<_Arg>(__arg));
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES((!_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Arg>, __transform_output_proxy>)
                   _CCCL_AND _CUDA_VSTD::is_invocable_v<const _Fn&, _Arg> _CCCL_AND
                     _CUDA_VSTD::is_assignable_v<_CUDA_VSTD::iter_reference_t<const _Iter>,
                                                 _CUDA_VSTD::invoke_result_t<const _Fn&, _Arg>>)
  _CCCL_API constexpr const __transform_output_proxy& operator=(_Arg&& __arg) const
    noexcept(noexcept(*__iter_ = _CUDA_VSTD::invoke(__func_, _CUDA_VSTD::forward<_Arg>(__arg))))
  {
    *__iter_ = _CUDA_VSTD::invoke(__func_, _CUDA_VSTD::forward<_Arg>(__arg));
    return *this;
  }
};

//! @brief @c transform_output_iterator is a special kind of output iterator which transforms a value written upon
//! dereference. This iterator is useful for transforming an output from algorithms without explicitly storing the
//! intermediate result in the memory and applying subsequent transformation, thereby avoiding wasting memory capacity
//! and bandwidth. Using @c transform_iterator facilitates kernel fusion by deferring execution of transformation until
//! the value is written while saving both memory capacity and bandwidth.
//!
//! The following code snippet demonstrated how to create a @c transform_output_iterator which applies @c sqrtf to the
//! assigning value.
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
//!     return cuda::std::sqrtf(x);
//!   }
//! };
//!
//! int main()
//! {
//!   thrust::device_vector<float> v(4);
//!   cuda::transform_output_iterator iter(v.begin(), square_root());
//!
//!   iter[0] =  1.0f;    // stores sqrtf( 1.0f)
//!   iter[1] =  4.0f;    // stores sqrtf( 4.0f)
//!   iter[2] =  9.0f;    // stores sqrtf( 9.0f)
//!   iter[3] = 16.0f;    // stores sqrtf(16.0f)
//!   // iter[4] is an out-of-bounds error
//!
//!   v[0]; // returns 1.0f;
//!   v[1]; // returns 2.0f;
//!   v[2]; // returns 3.0f;
//!   v[3]; // returns 4.0f;
//!
//! }
//! @endcode
template <class _Fn, class _Iter>
class transform_output_iterator
{
  static_assert(_CUDA_VSTD::is_object_v<_Fn>, "cuda::transform_output_iterator requires that _Fn is a function object");

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
                                                          ::cuda::std::output_iterator_tag>>>;
  using iterator_category = ::cuda::std::output_iterator_tag;
  using difference_type   = ::cuda::std::iter_difference_t<_Iter>;
  using value_type        = void;
  using pointer           = void;
  using reference         = void;

  //! @brief Default constructs a @c transform_output_iterator with a value initialized iterator and functor
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter, class _Fn2 = _Fn)
  _CCCL_REQUIRES(_CUDA_VSTD::default_initializable<_Iter2> _CCCL_AND _CUDA_VSTD::default_initializable<_Fn2>)
  _CCCL_API constexpr transform_output_iterator() noexcept(
    ::cuda::std::is_nothrow_default_constructible_v<_Iter2> && ::cuda::std::is_nothrow_default_constructible_v<_Fn2>)
      : __store_()
  {}

  //! @brief Constructs a @c transform_output_iterator with a given iterator and output functor
  //! @param __iter The iterator to transform
  //! @param __func The output function to apply to the iterator on assignment
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_output_iterator(_Iter __iter, _Fn __func) noexcept(
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

  //! @brief Returns a proxy that transforms the input upon assignment
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr auto operator*() const noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter>)
  {
    return __transform_output_proxy{__iter(), const_cast<_Fn&>(__func())};
  }

  //! @brief Returns a proxy that transforms the input upon assignment
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr auto operator*() noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter>)
  {
    return __transform_output_proxy{__iter(), __func()};
  }

  //! @brief Subscripts the @c transform_output_iterator
  //! @returns A proxy that transforms the input upon assignment storing the current iterator advanced by a given
  //! @param __n The number of elements to advance by
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__iter_can_subscript<_Iter2>)
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) const
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter2>
             && noexcept(::cuda::std::declval<const _Iter2&>() + __n))
  {
    return __transform_output_proxy{__iter() + __n, const_cast<_Fn&>(__func())};
  }

  //! @brief Subscripts the @c transform_output_iterator
  //! @returns A proxy that transforms the input upon assignment storing the current iterator advanced by a given
  //! @param __n The number of elements to advance by
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__iter_can_subscript<_Iter2>)
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter2> && noexcept(::cuda::std::declval<_Iter2&>() + __n))
  {
    return __transform_output_proxy{__iter() + __n, const_cast<_Fn&>(__func())};
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_output_iterator& operator++() noexcept(noexcept(++::cuda::std::declval<_Iter&>()))
  {
    ++__iter();
    return *this;
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr auto operator++(int) noexcept(noexcept(++::cuda::std::declval<_Iter&>()))
  {
    if constexpr (_CUDA_VSTD::__has_forward_traversal<_Iter> || _CUDA_VSTD::output_iterator<_Iter, value_type>)
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
  _CCCL_REQUIRES(::cuda::std::__iter_can_decrement<_Iter2>)
  _CCCL_API constexpr transform_output_iterator& operator--() noexcept(noexcept(--::cuda::std::declval<_Iter2&>()))
  {
    --__iter();
    return *this;
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_decrement<_Iter2>)
  _CCCL_API constexpr transform_output_iterator operator--(int) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && noexcept(--::cuda::std::declval<_Iter2&>()))
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Increments the @c transform_output_iterator by a given number of elements
  //! @param __n The number of elements to increment
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_plus_equal<_Iter2>)
  _CCCL_API constexpr transform_output_iterator&
  operator+=(difference_type __n) noexcept(noexcept(::cuda::std::declval<_Iter2&>() += __n))
  {
    __iter() += __n;
    return *this;
  }

  //! @brief Returns a copy of a @c transform_output_iterator incremented by a given number of elements
  //! @param __iter The @c transform_output_iterator to increment
  //! @param __n The number of elements to increment
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator+(const transform_output_iterator& __iter, difference_type __n) //
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter2>
             && noexcept(::cuda::std::declval<const _Iter2&>() + difference_type{}))
      _CCCL_TRAILING_REQUIRES(transform_output_iterator)(::cuda::std::__iter_can_plus<_Iter2>)
  {
    return transform_output_iterator{__iter.__iter() + __n, __iter.__func()};
  }

  //! @brief Returns a copy of a @c transform_output_iterator incremented by a given number of elements
  //! @param __n The number of elements to increment
  //! @param __iter The @c transform_output_iterator to increment
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator+(difference_type __n, const transform_output_iterator& __iter) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter2>
    && noexcept(_CUDA_VSTD::declval<const _Iter2&>() + difference_type{}))
    _CCCL_TRAILING_REQUIRES(transform_output_iterator)(_CUDA_VSTD::__iter_can_plus<_Iter2>)
  {
    return transform_output_iterator{__iter.__iter() + __n, __iter.__func()};
  }

  //! @brief Decrements the @c transform_output_iterator by a given number of elements
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_minus_equal<_Iter2>)
  _CCCL_API constexpr transform_output_iterator&
  operator-=(difference_type __n) noexcept(noexcept(::cuda::std::declval<_Iter2&>() -= __n))
  {
    __iter() -= __n;
    return *this;
  }

  //! @brief Returns a copy of a @c transform_output_iterator decremented by a given number of elements
  //! @param __iter The @c transform_output_iterator to decrement
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator-(const transform_output_iterator& __iter, difference_type __n) //
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter2>
             && noexcept(::cuda::std::declval<const _Iter2&>() - difference_type{}))
      _CCCL_TRAILING_REQUIRES(transform_output_iterator)(::cuda::std::__iter_can_minus<_Iter2>)
  {
    return transform_output_iterator{__iter.__iter() - __n, __iter.__func()};
  }

  template <class _Iter2>
  static constexpr bool __can_difference =
    (::cuda::std::__has_random_access_traversal<_Iter2> || ::cuda::std::sized_sentinel_for<_Iter2, _Iter2>);

  template <class _Iter2>
  static constexpr bool __noexcept_difference =
    noexcept(::cuda::std::declval<const _Iter2&>() - ::cuda::std::declval<const _Iter2&>());

  //! @brief Returns the distance between two @c transform_output_iterator
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto operator-(const transform_output_iterator& __lhs,
                                                          const transform_output_iterator& __rhs) //
    noexcept(__noexcept_difference<_Iter2>) _CCCL_TRAILING_REQUIRES(difference_type)(__can_difference<_Iter2>)
  {
    return __lhs.__iter() - __rhs.__iter();
  }

  //! @brief Compares two @c transform_output_iterator for equality by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator==(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() == _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::equality_comparable<_Iter2>)
  {
    return __lhs.__iter() == __rhs.__iter();
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c transform_output_iterator for inequality by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator!=(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() != _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::equality_comparable<_Iter2>)
  {
    return __lhs.__iter() != __rhs.__iter();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two @c transform_output_iterator by three-way-comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() <=> _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(
      _CUDA_VSTD::__has_random_access_traversal<_Iter2>&& _CUDA_VSTD::three_way_comparable<_Iter2>)
  {
    return __lhs.__iter() <=> __rhs.__iter();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  //! @brief Compares two @c transform_output_iterator for less than by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() < __rhs.__iter();
  }

  //! @brief Compares two @c transform_output_iterator for greater than by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() > __rhs.__iter();
  }

  //! @brief Compares two @c transform_output_iterator for less equal by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() <= __rhs.__iter();
  }

  //! @brief Compares two @c transform_output_iterator for greater equal by comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>=(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::__has_random_access_traversal<_Iter2>)
  {
    return __lhs.__iter() >= __rhs.__iter();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

//! @brief Creates a @c transform_output_iterator from an iterator and an output function.
//! @param __iter The iterator of the input range
//! @param __fun The output function
//! @relates transform_output_iterator
template <class _Fn, class _Iter>
[[nodiscard]] _CCCL_API constexpr auto make_transform_output_iterator(_Iter __iter, _Fn __fun)
{
  return transform_output_iterator<_Fn, _Iter>{__iter, __fun};
}

//! @}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_TRANSFORM_OUTPUT_ITERATOR_H
