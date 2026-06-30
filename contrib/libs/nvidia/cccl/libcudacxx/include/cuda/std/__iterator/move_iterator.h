// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/compare_three_way_result.h>
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/move_sentinel.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()
template <class _Iter, class = void>
struct __move_iter_category_base
{};

template <class _Iter>
  requires requires { typename iterator_traits<_Iter>::iterator_category; }
struct __move_iter_category_base<_Iter>
{
  using iterator_category =
    _If<derived_from<typename iterator_traits<_Iter>::iterator_category, random_access_iterator_tag>,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category>;
};

template <class _Iter, class _Sent>
concept __move_iter_comparable = requires {
  { declval<const _Iter&>() == declval<_Sent>() } -> convertible_to<bool>;
};

template <class _Iter>
inline constexpr bool __noexcept_move_iter_iter_move = noexcept(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<_Iter>()));
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Iter, class = void>
struct __move_iter_category_base
{};

template <class _Iter>
struct __move_iter_category_base<_Iter, enable_if_t<__has_iter_category<iterator_traits<_Iter>>>>
{
  using iterator_category =
    _If<derived_from<typename iterator_traits<_Iter>::iterator_category, random_access_iterator_tag>,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category>;
};

template <class _Iter, class _Sent>
_CCCL_CONCEPT_FRAGMENT(
  __move_iter_comparable_,
  requires()(requires(convertible_to<decltype(declval<const _Iter&>() == declval<_Sent>()), bool>)));

template <class _Iter, class _Sent>
_CCCL_CONCEPT __move_iter_comparable = _CCCL_FRAGMENT(__move_iter_comparable_, _Iter, _Sent);

template <class _Iter>
inline constexpr bool __noexcept_move_iter_iter_move = noexcept(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<_Iter>()));
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

template <class _Iter>
class _CCCL_TYPE_VISIBILITY_DEFAULT move_iterator : public __move_iter_category_base<_Iter>
{
private:
  template <class _It2>
  friend class move_iterator;

  _Iter __current_;

  _CCCL_API static constexpr auto __mi_get_iter_concept()
  {
    if constexpr (random_access_iterator<_Iter>)
    {
      return random_access_iterator_tag{};
    }
    else if constexpr (bidirectional_iterator<_Iter>)
    {
      return bidirectional_iterator_tag{};
    }
    else if constexpr (forward_iterator<_Iter>)
    {
      return forward_iterator_tag{};
    }
    else
    {
      return input_iterator_tag{};
    }
    _CCCL_UNREACHABLE();
  }

public:
  using iterator_type    = _Iter;
  using iterator_concept = decltype(__mi_get_iter_concept());

  // iterator_category is inherited and not always present
  using value_type      = iter_value_t<_Iter>;
  using difference_type = iter_difference_t<_Iter>;
  using pointer         = _Iter;
  using reference       = iter_rvalue_reference_t<_Iter>;

  _CCCL_API constexpr explicit move_iterator(_Iter __i)
      : __current_(_CUDA_VSTD::move(__i))
  {}

  _CCCL_API constexpr move_iterator& operator++()
  {
    ++__current_;
    return *this;
  }

  _LIBCUDACXX_DEPRECATED_IN_CXX20 _CCCL_API constexpr pointer operator->() const
  {
    return __current_;
  }

#if _CCCL_HAS_CONCEPTS()
  _CCCL_API constexpr move_iterator()
    requires is_constructible_v<_Iter>
      : __current_()
  {}
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _It2 = _Iter)
  _CCCL_REQUIRES(is_constructible_v<_It2>)
  _CCCL_API constexpr move_iterator()
      : __current_()
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!_IsSame<_Up, _Iter>::value) && convertible_to<const _Up&, _Iter>)
  _CCCL_API constexpr move_iterator(const move_iterator<_Up>& __u)
      : __current_(__u.base())
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!_IsSame<_Up, _Iter>::value)
                 && convertible_to<const _Up&, _Iter> && assignable_from<_Iter&, const _Up&>)
  _CCCL_API constexpr move_iterator& operator=(const move_iterator<_Up>& __u)
  {
    __current_ = __u.base();
    return *this;
  }

  _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __current_;
  }
  _CCCL_API constexpr _Iter base() &&
  {
    return _CUDA_VSTD::move(__current_);
  }

  _CCCL_API constexpr reference operator*() const
  {
    return _CUDA_VRANGES::iter_move(__current_);
  }
  _CCCL_API constexpr reference operator[](difference_type __n) const
  {
    return _CUDA_VRANGES::iter_move(__current_ + __n);
  }

  _CCCL_TEMPLATE(class _It2 = _Iter)
  _CCCL_REQUIRES(forward_iterator<_It2>)
  _CCCL_API constexpr auto operator++(int)
  {
    move_iterator __tmp(*this);
    ++__current_;
    return __tmp;
  }

  _CCCL_TEMPLATE(class _It2 = _Iter)
  _CCCL_REQUIRES((!forward_iterator<_It2>) )
  _CCCL_API constexpr void operator++(int)
  {
    ++__current_;
  }

  _CCCL_API constexpr move_iterator& operator--()
  {
    --__current_;
    return *this;
  }
  _CCCL_API constexpr move_iterator operator--(int)
  {
    move_iterator __tmp(*this);
    --__current_;
    return __tmp;
  }
  _CCCL_API constexpr move_iterator operator+(difference_type __n) const
  {
    return move_iterator(__current_ + __n);
  }
  _CCCL_API constexpr move_iterator& operator+=(difference_type __n)
  {
    __current_ += __n;
    return *this;
  }
  _CCCL_API constexpr move_iterator operator-(difference_type __n) const
  {
    return move_iterator(__current_ - __n);
  }
  _CCCL_API constexpr move_iterator& operator-=(difference_type __n)
  {
    __current_ -= __n;
    return *this;
  }

  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  friend _CCCL_API constexpr bool operator==(const move_iterator& __x, const move_sentinel<_Sent>& __y)
  {
    return __x.base() == __y.base();
  }

#if _CCCL_STD_VER < 2020
  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  friend _CCCL_API constexpr bool operator==(const move_sentinel<_Sent>& __y, const move_iterator& __x)
  {
    return __y.base() == __x.base();
  }

  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  friend _CCCL_API constexpr bool operator!=(const move_iterator& __x, const move_sentinel<_Sent>& __y)
  {
    return __x.base() != __y.base();
  }

  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  friend _CCCL_API constexpr bool operator!=(const move_sentinel<_Sent>& __y, const move_iterator& __x)
  {
    return __y.base() != __x.base();
  }
#endif // _CCCL_STD_VER < 2020

  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sized_sentinel_for<_Sent, _Iter>)
  friend _CCCL_API constexpr iter_difference_t<_Iter>
  operator-(const move_sentinel<_Sent>& __x, const move_iterator& __y)
  {
    return __x.base() - __y.base();
  }

  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sized_sentinel_for<_Sent, _Iter>)
  friend _CCCL_API constexpr iter_difference_t<_Iter>
  operator-(const move_iterator& __x, const move_sentinel<_Sent>& __y)
  {
    return __x.base() - __y.base();
  }

  _CCCL_API friend constexpr iter_rvalue_reference_t<_Iter>
  iter_move(const move_iterator& __i) noexcept(__noexcept_move_iter_iter_move<_Iter>)
  {
    return _CUDA_VRANGES::iter_move(__i.__current_);
  }

  template <class _Iter2>
  _CCCL_API friend constexpr auto
  iter_swap(const move_iterator& __x, const move_iterator<_Iter2>& __y) noexcept(__noexcept_swappable<_Iter, _Iter2>)
    _CCCL_TRAILING_REQUIRES(void)(indirectly_swappable<_Iter2, _Iter>)
  {
    return _CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_);
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(move_iterator);
_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(move_iterator)

// Some compilers have issues determining _IsFancyPointer
#if _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)
template <class _Iter>
struct _IsFancyPointer<move_iterator<_Iter>> : _IsFancyPointer<_Iter>
{};
#endif // _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator==(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
  return __x.base() == __y.base();
}

#if _CCCL_STD_VER <= 2017
template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator!=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
  return __x.base() != __y.base();
}
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class _Iter1, three_way_comparable_with<_Iter1> _Iter2>
_CCCL_API constexpr auto operator<=>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
  -> compare_three_way_result_t<_Iter1, _Iter2>
{
  return __x.base() <=> __y.base();
}

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator<(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
  return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator<=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
  return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator>=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
  return __x.base() >= __y.base();
}
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class _Iter1, class _Iter2>
_CCCL_API constexpr auto operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
  -> decltype(__x.base() - __y.base())
{
  return __x.base() - __y.base();
}

#if _CCCL_HAS_CONCEPTS()
template <class _Iter>
_CCCL_API constexpr move_iterator<_Iter> operator+(iter_difference_t<_Iter> __n, const move_iterator<_Iter>& __x)
  requires requires {
    { __x.base() + __n } -> same_as<_Iter>;
  }
{
  return __x + __n;
}
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Iter>
_CCCL_API constexpr move_iterator<_Iter>
operator+(typename move_iterator<_Iter>::difference_type __n, const move_iterator<_Iter>& __x)
{
  return move_iterator<_Iter>(__x.base() + __n);
}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

template <class _Iter>
_CCCL_API constexpr move_iterator<_Iter> make_move_iterator(_Iter __i)
{
  return move_iterator<_Iter>(_CUDA_VSTD::move(__i));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H
