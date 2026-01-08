//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_TABULATE_OUTPUT_ITERATOR_H
#define _CUDA___ITERATOR_TABULATE_OUTPUT_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/compressed_movable_box.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @addtogroup iterators
//! @{

//! @brief @c tabulate_output_iterator is a special kind of output iterator which, whenever a value is assigned to a
//! dereferenced iterator, calls the given callable with the index that corresponds to the offset of the dereferenced
//! iterator and the assigned value.
//!
//! The following code snippet demonstrates how to create a @c tabulate_output_iterator which prints the index and the
//! assigned value.
//!
//! @code
//! #include <cuda/iterator>
//!
//! struct print_op
//! {
//!   __host__ __device__ void operator()(int index, float value) const
//!   {
//!     printf("%d: %f\n", index, value);
//!   }
//! };
//!
//! int main()
//! {
//!   auto tabulate_it = cuda::make_tabulate_output_iterator(print_op{});
//!
//!   tabulate_it[0] =  1.0f;    // prints: 0: 1.0
//!   tabulate_it[1] =  3.0f;    // prints: 1: 3.0
//!   tabulate_it[9] =  5.0f;    // prints: 9: 5.0
//! }
//! @endcode
template <class _Fn, class _Index = _CUDA_VSTD::ptrdiff_t>
class tabulate_output_iterator;

template <class _Fn, class _Index>
class __tabulate_proxy
{
private:
  template <class, class>
  friend class tabulate_output_iterator;

  _Fn& __func_;
  _Index __index_;

public:
  _CCCL_API constexpr explicit __tabulate_proxy(_Fn& __func, _Index __index) noexcept
      : __func_(__func)
      , __index_(__index)
  {}

  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(_CUDA_VSTD::is_invocable_v<_Fn&, _Index, _Arg> _CCCL_AND(
    !_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Arg>, __tabulate_proxy>))
  _CCCL_API constexpr const __tabulate_proxy&
  operator=(_Arg&& __arg) noexcept(_CUDA_VSTD::is_nothrow_invocable_v<_Fn&, _Index, _Arg>)
  {
    _CUDA_VSTD::invoke(__func_, __index_, _CUDA_VSTD::forward<_Arg>(__arg));
    return *this;
  }

  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(_CUDA_VSTD::is_invocable_v<const _Fn&, _Index, _Arg> _CCCL_AND(
    !_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Arg>, __tabulate_proxy>))
  _CCCL_API constexpr const __tabulate_proxy& operator=(_Arg&& __arg) const
    noexcept(_CUDA_VSTD::is_nothrow_invocable_v<const _Fn&, _Index, _Arg>)
  {
    _CUDA_VSTD::invoke(__func_, __index_, _CUDA_VSTD::forward<_Arg>(__arg));
    return *this;
  }
};

template <class _Fn, class _Index>
class tabulate_output_iterator
{
private:
  // Not a base because then the friend operators would be ambiguous
  ::cuda::std::__compressed_movable_box<_Index, _Fn> __store_;

  [[nodiscard]] _CCCL_API constexpr _Index& __index() noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Index& __index() const noexcept
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

  static_assert(::cuda::std::is_signed_v<_Index>, "tabulate_output_iterator: _Index must be a signed integer");

public:
  using iterator_concept  = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using difference_type   = _Index;
  using value_type        = void;
  using pointer           = void;
  using reference         = void;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Fn2 = _Fn)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Fn2>)
  _CCCL_API constexpr tabulate_output_iterator() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Fn2>)
      : __store_()
  {}

  //! @brief Constructs a @c tabulate_output_iterator with a given functor and an optional index
  //! @param __func the output function
  //! @param __index the position in the output sequence
  _CCCL_API constexpr tabulate_output_iterator(_Fn __func, _Index __index = 0) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Fn>)
      : __store_(__index, ::cuda::std::move(__func))
  {}

  //! @brief Returns the stored index
  [[nodiscard]] _CCCL_API constexpr difference_type index() const noexcept
  {
    return __index();
  }

  //! @brief Dereferences the @c tabulate_output_iterator
  //! @returns A proxy that applies the stored function and index on assignment
  [[nodiscard]] _CCCL_API constexpr auto operator*() const noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{const_cast<_Fn&>(__func()), __index()};
  }

  //! @brief Dereferences the @c tabulate_output_iterator
  //! @returns A proxy that applies the stored function and index on assignment
  [[nodiscard]] _CCCL_API constexpr auto operator*() noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{__func(), __index()};
  }

  //! @brief Subscripts the @c tabulate_output_iterator with a given offset
  //! @param __n The additional offset to advance the stored index
  //! @returns A proxy that applies the stored function and index on assignment
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) const noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{const_cast<_Fn&>(__func()), __index() + __n};
  }

  //! @brief Subscripts the @c tabulate_output_iterator with a given offset
  //! @param __n The additional offset to advance the stored index
  //! @returns A proxy that applies the stored function and index on assignment
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{__func(), __index() + __n};
  }

  //! @brief Increments the @c tabulate_output_iterator by incrementing the stored index
  _CCCL_API constexpr tabulate_output_iterator& operator++() noexcept
  {
    ++__index();
    return *this;
  }

  //! @brief Increments the @c tabulate_output_iterator by incrementing the stored index
  _CCCL_API constexpr tabulate_output_iterator operator++(int) noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Fn>)
  {
    tabulate_output_iterator __tmp = *this;
    ++__index();
    return __tmp;
  }

  //! @brief Decrements the @c tabulate_output_iterator by decrementing the stored index
  _CCCL_API constexpr tabulate_output_iterator& operator--() noexcept
  {
    --__index();
    return *this;
  }

  //! @brief Decrements the @c tabulate_output_iterator by decrementing the stored index
  _CCCL_API constexpr tabulate_output_iterator operator--(int) noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Fn>)
  {
    tabulate_output_iterator __tmp = *this;
    --__index();
    return __tmp;
  }

  //! @brief Returns a copy of this @c tabulate_output_iterator advanced a given number of elements
  //! @param __n The number of elements to advance
  [[nodiscard]] _CCCL_API friend constexpr tabulate_output_iterator
  operator+(const tabulate_output_iterator& __iter, difference_type __n) //
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
  {
    return tabulate_output_iterator{__iter.__func(), __iter.__index() + __n};
  }

  //! @brief Returns a copy of a @c tabulate_output_iterator advanced a given number of elements
  //! @param __n The number of elements to advance
  //! @param __iter The original @c tabulate_output_iterator
  [[nodiscard]] _CCCL_API friend constexpr tabulate_output_iterator
  operator+(difference_type __n, const tabulate_output_iterator& __iter) //
    noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Fn>)
  {
    return tabulate_output_iterator{__iter.__func(), __iter.__index() + __n};
  }

  //! @brief Advances the @c tabulate_output_iterator by a given number of elements
  //! @param __n The number of elements to advance
  _CCCL_API constexpr tabulate_output_iterator& operator+=(difference_type __n) noexcept
  {
    __index() += __n;
    return *this;
  }

  //! @brief Returns a copy of this @c tabulate_output_iterator decremented a given number of elements
  //! @param __n The number of elements to decremented
  [[nodiscard]] _CCCL_API friend constexpr tabulate_output_iterator
  operator-(const tabulate_output_iterator& __iter, difference_type __n) //
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
  {
    return tabulate_output_iterator{__iter.__func(), __iter.__index() - __n};
  }

  //! @brief Returns the distance between two @c tabulate_output_iterator 's
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __rhs.__index() - __lhs.__index();
  }

  //! @brief Decrements the @c tabulate_output_iterator by a given number of elements
  //! @param __n The number of elements to decrement
  _CCCL_API constexpr tabulate_output_iterator& operator-=(difference_type __n) noexcept
  {
    __index() -= __n;
    return *this;
  }

  //! @brief Compares two @c tabulate_output_iterator for equality by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index() == __rhs.__index();
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c tabulate_output_iterator for inequality by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index() != __rhs.__index();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two @c tabulate_output_iterator by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering
  operator<=>(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index() <=> __rhs.__index();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  //! @brief Compares two @c tabulate_output_iterator for less than by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index() < __rhs.__index();
  }

  //! @brief Compares two @c tabulate_output_iterator for less equal by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index() <= __rhs.__index();
  }

  //! @brief Compares two @c tabulate_output_iterator for greater than by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index() > __rhs.__index();
  }

  //! @brief Compares two @c tabulate_output_iterator for greater equal by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index() >= __rhs.__index();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

template <class _Fn>
_CCCL_HOST_DEVICE tabulate_output_iterator(_Fn) -> tabulate_output_iterator<_Fn, _CUDA_VSTD::ptrdiff_t>;

_CCCL_TEMPLATE(class _Fn, class _Index)
_CCCL_REQUIRES(_CUDA_VSTD::__integer_like<_Index>)
_CCCL_HOST_DEVICE tabulate_output_iterator(_Fn, _Index) -> tabulate_output_iterator<_Fn, _Index>;

//! @brief Creates a @c tabulate_output_iterator from an output function and an optional index.
//! @param __func The output function
//! @param __index The index of the @c tabulate_output_iterator within a range. The default index is @c 0.
//! @return A new @c tabulate_output_iterator with @c __index as the counter.
//! @relates tabulate_output_iterator
_CCCL_TEMPLATE(class _Fn, class _Integer = _CUDA_VSTD::ptrdiff_t)
_CCCL_REQUIRES(_CUDA_VSTD::__integer_like<_Integer>)
[[nodiscard]] _CCCL_API constexpr auto make_tabulate_output_iterator(_Fn __func, _Integer __index = 0)
{
  return tabulate_output_iterator{_CUDA_VSTD::move(__func), __index};
}

//! @}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_TABULATE_OUTPUT_ITERATOR_H
