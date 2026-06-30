// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_UNINITIALIZED_ALGORITHMS_H
#define _LIBCUDACXX___MEMORY_UNINITIALIZED_ALGORITHMS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/move.h>
#include <cuda/std/__algorithm/unwrap_iter.h>
#include <cuda/std/__algorithm/unwrap_range.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__memory/voidify.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__new_>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__type_traits/is_unbounded_array.h>
#include <cuda/std/__type_traits/negation.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_extent.h>
#include <cuda/std/__utility/exception_guard.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __AlwaysFalse
{
  template <class... _Args>
  _CCCL_API constexpr bool operator()(_Args&&...) const noexcept
  {
    return false;
  }
};

template <class _ForwardIterator>
struct __simple_rollback
{
  _ForwardIterator& __first_;
  _ForwardIterator& __current_;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline __simple_rollback(_ForwardIterator& __first, _ForwardIterator& __current)
      : __first_(__first)
      , __current_(__current)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline void operator()() const noexcept
  {
    _CUDA_VSTD::__destroy(__first_, __current_);
  }
};

// uninitialized_copy

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _InputIterator, class _Sentinel1, class _ForwardIterator, class _EndPredicate>
_CCCL_API inline pair<_InputIterator, _ForwardIterator> __uninitialized_copy(
  _InputIterator __ifirst, _Sentinel1 __ilast, _ForwardIterator __ofirst, _EndPredicate __stop_copying)
{
  _ForwardIterator __idx = __ofirst;
  auto __guard           = __make_exception_guard(__simple_rollback<_ForwardIterator>{__ofirst, __idx});
  for (; __ifirst != __ilast && !__stop_copying(__idx); ++__ifirst, (void) ++__idx)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType(*__ifirst);
  }
  __guard.__complete();

  return pair<_InputIterator, _ForwardIterator>(_CUDA_VSTD::move(__ifirst), _CUDA_VSTD::move(__idx));
}

template <class _InputIterator, class _ForwardIterator>
_CCCL_API inline _ForwardIterator
uninitialized_copy(_InputIterator __ifirst, _InputIterator __ilast, _ForwardIterator __ofirst)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  auto __result    = _CUDA_VSTD::__uninitialized_copy<_ValueType>(
    _CUDA_VSTD::move(__ifirst), _CUDA_VSTD::move(__ilast), _CUDA_VSTD::move(__ofirst), __AlwaysFalse{});
  return _CUDA_VSTD::move(__result.second);
}

// uninitialized_copy_n

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _InputIterator, class _Size, class _ForwardIterator, class _EndPredicate>
_CCCL_API inline pair<_InputIterator, _ForwardIterator>
__uninitialized_copy_n(_InputIterator __ifirst, _Size __n, _ForwardIterator __ofirst, _EndPredicate __stop_copying)
{
  _ForwardIterator __idx = __ofirst;
  auto __guard           = __make_exception_guard(__simple_rollback<_ForwardIterator>{__ofirst, __idx});
  for (; __n > 0 && !__stop_copying(__idx); ++__ifirst, (void) ++__idx, (void) --__n)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType(*__ifirst);
  }
  __guard.__complete();

  return pair<_InputIterator, _ForwardIterator>(_CUDA_VSTD::move(__ifirst), _CUDA_VSTD::move(__idx));
}

template <class _InputIterator, class _Size, class _ForwardIterator>
_CCCL_API inline _ForwardIterator uninitialized_copy_n(_InputIterator __ifirst, _Size __n, _ForwardIterator __ofirst)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  auto __result    = _CUDA_VSTD::__uninitialized_copy_n<_ValueType>(
    _CUDA_VSTD::move(__ifirst), __n, _CUDA_VSTD::move(__ofirst), __AlwaysFalse{});
  return _CUDA_VSTD::move(__result.second);
}

// uninitialized_fill

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _ForwardIterator, class _Sentinel, class _Tp>
_CCCL_API inline _ForwardIterator __uninitialized_fill(_ForwardIterator __first, _Sentinel __last, const _Tp& __x)
{
  _ForwardIterator __idx = __first;
  auto __guard           = __make_exception_guard(__simple_rollback<_ForwardIterator>{__first, __idx});
  for (; __idx != __last; ++__idx)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType(__x);
  }
  __guard.__complete();

  return __idx;
}

template <class _ForwardIterator, class _Tp>
_CCCL_API inline void uninitialized_fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __x)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  (void) _CUDA_VSTD::__uninitialized_fill<_ValueType>(__first, __last, __x);
}

// uninitialized_fill_n

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _ForwardIterator, class _Size, class _Tp>
_CCCL_API inline _ForwardIterator __uninitialized_fill_n(_ForwardIterator __first, _Size __n, const _Tp& __x)
{
  _ForwardIterator __idx = __first;
  auto __guard           = __make_exception_guard(__simple_rollback<_ForwardIterator>{__first, __idx});
  for (; __n > 0; ++__idx, (void) --__n)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType(__x);
  }
  __guard.__complete();

  return __idx;
}

template <class _ForwardIterator, class _Size, class _Tp>
_CCCL_API inline _ForwardIterator uninitialized_fill_n(_ForwardIterator __first, _Size __n, const _Tp& __x)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  return _CUDA_VSTD::__uninitialized_fill_n<_ValueType>(__first, __n, __x);
}

// uninitialized_default_construct

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _ForwardIterator, class _Sentinel>
_CCCL_API inline _ForwardIterator __uninitialized_default_construct(_ForwardIterator __first, _Sentinel __last)
{
  auto __idx   = __first;
  auto __guard = __make_exception_guard(__simple_rollback<_ForwardIterator>{__first, __idx});
  for (; __idx != __last; ++__idx)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType;
  }
  __guard.__complete();

  return __idx;
}

template <class _ForwardIterator>
_CCCL_API inline void uninitialized_default_construct(_ForwardIterator __first, _ForwardIterator __last)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  (void) _CUDA_VSTD::__uninitialized_default_construct<_ValueType>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last));
}

// uninitialized_default_construct_n

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _ForwardIterator, class _Size>
_CCCL_API inline _ForwardIterator __uninitialized_default_construct_n(_ForwardIterator __first, _Size __n)
{
  auto __idx   = __first;
  auto __guard = __make_exception_guard(__simple_rollback<_ForwardIterator>{__first, __idx});
  for (; __n > 0; ++__idx, (void) --__n)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType;
  }
  __guard.__complete();

  return __idx;
}

template <class _ForwardIterator, class _Size>
_CCCL_API inline _ForwardIterator uninitialized_default_construct_n(_ForwardIterator __first, _Size __n)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  return _CUDA_VSTD::__uninitialized_default_construct_n<_ValueType>(_CUDA_VSTD::move(__first), __n);
}

// uninitialized_value_construct

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _ForwardIterator, class _Sentinel>
_CCCL_API inline _ForwardIterator __uninitialized_value_construct(_ForwardIterator __first, _Sentinel __last)
{
  auto __idx   = __first;
  auto __guard = __make_exception_guard(__simple_rollback<_ForwardIterator>{__first, __idx});
  for (; __idx != __last; ++__idx)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType();
  }
  __guard.__complete();

  return __idx;
}

template <class _ForwardIterator>
_CCCL_API inline void uninitialized_value_construct(_ForwardIterator __first, _ForwardIterator __last)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  (void) _CUDA_VSTD::__uninitialized_value_construct<_ValueType>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last));
}

// uninitialized_value_construct_n

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _ForwardIterator, class _Size>
_CCCL_API inline _ForwardIterator __uninitialized_value_construct_n(_ForwardIterator __first, _Size __n)
{
  auto __idx   = __first;
  auto __guard = __make_exception_guard(__simple_rollback<_ForwardIterator>{__first, __idx});
  for (; __n > 0; ++__idx, (void) --__n)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType();
  }
  __guard.__complete();

  return __idx;
}

template <class _ForwardIterator, class _Size>
_CCCL_API inline _ForwardIterator uninitialized_value_construct_n(_ForwardIterator __first, _Size __n)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  return _CUDA_VSTD::__uninitialized_value_construct_n<_ValueType>(_CUDA_VSTD::move(__first), __n);
}

// uninitialized_move

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType,
          class _IterOps,
          class _InputIterator,
          class _Sentinel1,
          class _ForwardIterator,
          class _EndPredicate>
_CCCL_API inline pair<_InputIterator, _ForwardIterator> __uninitialized_move(
  _InputIterator __ifirst, _Sentinel1 __ilast, _ForwardIterator __ofirst, _EndPredicate __stop_moving)
{
  auto __idx   = __ofirst;
  auto __guard = __make_exception_guard(__simple_rollback<_ForwardIterator>{__ofirst, __idx});
  for (; __ifirst != __ilast && !__stop_moving(__idx); ++__idx, (void) ++__ifirst)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType(_IterOps::__iter_move(__ifirst));
  }
  __guard.__complete();

  return {_CUDA_VSTD::move(__ifirst), _CUDA_VSTD::move(__idx)};
}

template <class _InputIterator, class _ForwardIterator>
_CCCL_API inline _ForwardIterator
uninitialized_move(_InputIterator __ifirst, _InputIterator __ilast, _ForwardIterator __ofirst)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  auto __result    = _CUDA_VSTD::__uninitialized_move<_ValueType, _IterOps<_ClassicAlgPolicy>>(
    _CUDA_VSTD::move(__ifirst), _CUDA_VSTD::move(__ilast), _CUDA_VSTD::move(__ofirst), __AlwaysFalse{});
  return _CUDA_VSTD::move(__result.second);
}

// uninitialized_move_n

_CCCL_EXEC_CHECK_DISABLE
template <class _ValueType, class _IterOps, class _InputIterator, class _Size, class _ForwardIterator, class _EndPredicate>
_CCCL_API inline pair<_InputIterator, _ForwardIterator>
__uninitialized_move_n(_InputIterator __ifirst, _Size __n, _ForwardIterator __ofirst, _EndPredicate __stop_moving)
{
  auto __idx   = __ofirst;
  auto __guard = __make_exception_guard(__simple_rollback<_ForwardIterator>{__ofirst, __idx});
  for (; __n > 0 && !__stop_moving(__idx); ++__idx, (void) ++__ifirst, --__n)
  {
    ::new (_CUDA_VSTD::__voidify(*__idx)) _ValueType(_IterOps::__iter_move(__ifirst));
  }
  __guard.__complete();

  return {_CUDA_VSTD::move(__ifirst), _CUDA_VSTD::move(__idx)};
}

template <class _InputIterator, class _Size, class _ForwardIterator>
_CCCL_API inline pair<_InputIterator, _ForwardIterator>
uninitialized_move_n(_InputIterator __ifirst, _Size __n, _ForwardIterator __ofirst)
{
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  return _CUDA_VSTD::__uninitialized_move_n<_ValueType, _IterOps<_ClassicAlgPolicy>>(
    _CUDA_VSTD::move(__ifirst), __n, _CUDA_VSTD::move(__ofirst), __AlwaysFalse{});
}

// TODO: Rewrite this to iterate left to right and use reverse_iterators when calling
// Destroys every element in the range [first, last) FROM RIGHT TO LEFT using allocator
// destruction. If elements are themselves C-style arrays, they are recursively destroyed
// in the same manner.
//
// This function assumes that destructors do not throw, and that the allocator is bound to
// the correct type.
template <class _Alloc, class _BidirIter, enable_if_t<__has_bidirectional_traversal<_BidirIter>, int> = 0>
_CCCL_API constexpr void
__allocator_destroy_multidimensional(_Alloc& __alloc, _BidirIter __first, _BidirIter __last) noexcept
{
  using _ValueType = typename iterator_traits<_BidirIter>::value_type;
  static_assert(_CCCL_TRAIT(is_same, typename allocator_traits<_Alloc>::value_type, _ValueType),
                "The allocator should already be rebound to the correct type");

  if (__first == __last)
  {
    return;
  }

  if constexpr (_CCCL_TRAIT(is_array, _ValueType))
  {
    static_assert(!is_unbounded_array_v<_ValueType>,
                  "arrays of unbounded arrays don't exist, but if they did we would mess up here");

    using _Element = remove_extent_t<_ValueType>;
    __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
    do
    {
      --__last;
      auto&& __array = *__last;
      _CUDA_VSTD::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + _CCCL_TRAIT(extent, _ValueType));
    } while (__last != __first);
  }
  else
  {
    do
    {
      --__last;
      allocator_traits<_Alloc>::destroy(__alloc, _CUDA_VSTD::addressof(*__last));
    } while (__last != __first);
  }
}

// Constructs the object at the given location using the allocator's construct method.
//
// If the object being constructed is an array, each element of the array is allocator-constructed,
// recursively. If an exception is thrown during the construction of an array, the initialized
// elements are destroyed in reverse order of initialization using allocator destruction.
//
// This function assumes that the allocator is bound to the correct type.
template <class _Alloc, class _Tp>
_CCCL_API constexpr void __allocator_construct_at_multidimensional(_Alloc& __alloc, _Tp* __loc)
{
  static_assert(_CCCL_TRAIT(is_same, typename allocator_traits<_Alloc>::value_type, _Tp),
                "The allocator should already be rebound to the correct type");

  if constexpr (_CCCL_TRAIT(is_array, _Tp))
  {
    using _Element = remove_extent_t<_Tp>;
    __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
    size_t __i   = 0;
    _Tp& __array = *__loc;

    // If an exception is thrown, destroy what we have constructed so far in reverse order.
    auto __guard = _CUDA_VSTD::__make_exception_guard([&]() {
      _CUDA_VSTD::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + __i);
    });

    for (; __i != _CCCL_TRAIT(extent, _Tp); ++__i)
    {
      _CUDA_VSTD::__allocator_construct_at_multidimensional(__elem_alloc, _CUDA_VSTD::addressof(__array[__i]));
    }
    __guard.__complete();
  }
  else
  {
    allocator_traits<_Alloc>::construct(__alloc, __loc);
  }
}

// Constructs the object at the given location using the allocator's construct method, passing along
// the provided argument.
//
// If the object being constructed is an array, the argument is also assumed to be an array. Each
// each element of the array being constructed is allocator-constructed from the corresponding
// element of the argument array. If an exception is thrown during the construction of an array,
// the initialized elements are destroyed in reverse order of initialization using allocator
// destruction.
//
// This function assumes that the allocator is bound to the correct type.
template <class _Alloc, class _Tp, class _Arg>
_CCCL_API constexpr void __allocator_construct_at_multidimensional(_Alloc& __alloc, _Tp* __loc, _Arg const& __arg)
{
  static_assert(_CCCL_TRAIT(is_same, typename allocator_traits<_Alloc>::value_type, _Tp),
                "The allocator should already be rebound to the correct type");

  if constexpr (_CCCL_TRAIT(is_array, _Tp))
  {
    static_assert(_CCCL_TRAIT(is_array, _Arg),
                  "Provided non-array initialization argument to __allocator_construct_at_multidimensional when "
                  "trying to construct an array.");

    using _Element = remove_extent_t<_Tp>;
    __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
    size_t __i   = 0;
    _Tp& __array = *__loc;

    // If an exception is thrown, destroy what we have constructed so far in reverse order.
    auto __guard = _CUDA_VSTD::__make_exception_guard([&]() {
      _CUDA_VSTD::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + __i);
    });
    for (; __i != _CCCL_TRAIT(extent, _Tp); ++__i)
    {
      _CUDA_VSTD::__allocator_construct_at_multidimensional(
        __elem_alloc, _CUDA_VSTD::addressof(__array[__i]), __arg[__i]);
    }
    __guard.__complete();
  }
  else
  {
    allocator_traits<_Alloc>::construct(__alloc, __loc, __arg);
  }
}

// Given a range starting at it and containing n elements, initializes each element in the
// range from left to right using the construct method of the allocator (rebound to the
// correct type).
//
// If an exception is thrown, the initialized elements are destroyed in reverse order of
// initialization using allocator_traits destruction. If the elements in the range are C-style
// arrays, they are initialized element-wise using allocator construction, and recursively so.
template <class _Alloc, class _BidirIter, class _Tp, class _Size = typename iterator_traits<_BidirIter>::difference_type>
_CCCL_API constexpr void
__uninitialized_allocator_fill_n_multidimensional(_Alloc& __alloc, _BidirIter __it, _Size __n, _Tp const& __value)
{
  using _ValueType = typename iterator_traits<_BidirIter>::value_type;
  __allocator_traits_rebind_t<_Alloc, _ValueType> __value_alloc(__alloc);
  _BidirIter __begin = __it;

  // If an exception is thrown, destroy what we have constructed so far in reverse order.
  auto __guard = _CUDA_VSTD::__make_exception_guard([&]() {
    _CUDA_VSTD::__allocator_destroy_multidimensional(__value_alloc, __begin, __it);
  });
  for (; __n != 0; --__n, ++__it)
  {
    _CUDA_VSTD::__allocator_construct_at_multidimensional(__value_alloc, _CUDA_VSTD::addressof(*__it), __value);
  }
  __guard.__complete();
}

// Same as __uninitialized_allocator_fill_n_multidimensional, but doesn't pass any initialization argument
// to the allocator's construct method, which results in value initialization.
template <class _Alloc, class _BidirIter, class _Size = typename iterator_traits<_BidirIter>::difference_type>
_CCCL_API constexpr void
__uninitialized_allocator_value_construct_n_multidimensional(_Alloc& __alloc, _BidirIter __it, _Size __n)
{
  using _ValueType = typename iterator_traits<_BidirIter>::value_type;
  __allocator_traits_rebind_t<_Alloc, _ValueType> __value_alloc(__alloc);
  _BidirIter __begin = __it;

  // If an exception is thrown, destroy what we have constructed so far in reverse order.
  auto __guard = _CUDA_VSTD::__make_exception_guard([&]() {
    _CUDA_VSTD::__allocator_destroy_multidimensional(__value_alloc, __begin, __it);
  });
  for (; __n != 0; --__n, ++__it)
  {
    _CUDA_VSTD::__allocator_construct_at_multidimensional(__value_alloc, _CUDA_VSTD::addressof(*__it));
  }
  __guard.__complete();
}

// Destroy all elements in [__first, __last) from left to right using allocator destruction.
template <class _Alloc, class _Iter, class _Sent>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 void __allocator_destroy(_Alloc& __alloc, _Iter __first, _Sent __last)
{
  for (; __first != __last; ++__first)
  {
    allocator_traits<_Alloc>::destroy(__alloc, _CUDA_VSTD::__to_address(__first));
  }
}

template <class _Alloc, class _Iter>
class _AllocatorDestroyRangeReverse
{
public:
  _CCCL_API constexpr _AllocatorDestroyRangeReverse(_Alloc& __alloc, _Iter& __first, _Iter& __last)
      : __alloc_(__alloc)
      , __first_(__first)
      , __last_(__last)
  {}

  _CCCL_API constexpr void operator()() const
  {
    _CUDA_VSTD::__allocator_destroy(
      __alloc_, _CUDA_VSTD::reverse_iterator<_Iter>(__last_), _CUDA_VSTD::reverse_iterator<_Iter>(__first_));
  }

private:
  _Alloc& __alloc_;
  _Iter& __first_;
  _Iter& __last_;
};

// Copy-construct [__first1, __last1) in [__first2, __first2 + N), where N is distance(__first1, __last1).
//
// The caller has to ensure that __first2 can hold at least N uninitialized elements. If an exception is thrown the
// already copied elements are destroyed in reverse order of their construction.
template <class _Alloc, class _Iter1, class _Sent1, class _Iter2>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _Iter2
__uninitialized_allocator_copy_impl(_Alloc& __alloc, _Iter1 __first1, _Sent1 __last1, _Iter2 __first2)
{
  auto __destruct_first = __first2;
  auto __guard          = _CUDA_VSTD::__make_exception_guard(
    _AllocatorDestroyRangeReverse<_Alloc, _Iter2>(__alloc, __destruct_first, __first2));
  while (__first1 != __last1)
  {
    allocator_traits<_Alloc>::construct(__alloc, _CUDA_VSTD::__to_address(__first2), *__first1);
    ++__first1;
    ++__first2;
  }
  __guard.__complete();
  return __first2;
}

template <class _Alloc, class _Type>
struct __allocator_has_trivial_copy_construct : _Not<__has_construct<_Alloc, _Type*, const _Type&>>
{};

template <class _Type>
struct __allocator_has_trivial_copy_construct<allocator<_Type>, _Type> : true_type
{};

template <
  class _Alloc,
  class _In,
  class _RawTypeIn = remove_const_t<_In>,
  class _Out,
  enable_if_t<
    // using _RawTypeIn because of the allocator<T const> extension
    _CCCL_TRAIT(is_trivially_copy_constructible, _RawTypeIn) && _CCCL_TRAIT(is_trivially_copy_assignable, _RawTypeIn)
    && _CCCL_TRAIT(is_same, remove_const_t<_In>, remove_const_t<_Out>)
    && __allocator_has_trivial_copy_construct<_Alloc, _RawTypeIn>::value>* = nullptr>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _Out*
__uninitialized_allocator_copy_impl(_Alloc&, _In* __first1, _In* __last1, _Out* __first2)
{
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    while (__first1 != __last1)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::__to_address(__first2), *__first1);
      ++__first1;
      ++__first2;
    }
    return __first2;
  }
  else
  {
    return _CUDA_VSTD::copy(__first1, __last1, __first2);
  }
}

template <class _Alloc, class _Iter1, class _Sent1, class _Iter2>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _Iter2
__uninitialized_allocator_copy(_Alloc& __alloc, _Iter1 __first1, _Sent1 __last1, _Iter2 __first2)
{
  auto __unwrapped_range = _CUDA_VSTD::__unwrap_range(__first1, __last1);
  auto __result          = _CUDA_VSTD::__uninitialized_allocator_copy_impl(
    __alloc, __unwrapped_range.first, __unwrapped_range.second, _CUDA_VSTD::__unwrap_iter(__first2));
  return _CUDA_VSTD::__rewrap_iter(__first2, __result);
}

// Move-construct the elements [__first1, __last1) into [__first2, __first2 + N)
// if the move constructor is noexcept, where N is distance(__first1, __last1).
//
// Otherwise try to copy all elements. If an exception is thrown the already copied
// elements are destroyed in reverse order of their construction.
template <class _Alloc, class _Iter1, class _Sent1, class _Iter2>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _Iter2
__uninitialized_allocator_move_if_noexcept(_Alloc& __alloc, _Iter1 __first1, _Sent1 __last1, _Iter2 __first2)
{
  static_assert(__is_cpp17_move_insertable<_Alloc>::value,
                "The specified type does not meet the requirements of Cpp17MoveInsertable");
  auto __destruct_first = __first2;
  auto __guard          = _CUDA_VSTD::__make_exception_guard(
    _AllocatorDestroyRangeReverse<_Alloc, _Iter2>(__alloc, __destruct_first, __first2));
  while (__first1 != __last1)
  {
#if _CCCL_HAS_EXCEPTIONS()
    allocator_traits<_Alloc>::construct(
      __alloc, _CUDA_VSTD::__to_address(__first2), _CUDA_VSTD::move_if_noexcept(*__first1));
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
    allocator_traits<_Alloc>::construct(__alloc, _CUDA_VSTD::__to_address(__first2), _CUDA_VSTD::move(*__first1));
#endif // !_CCCL_HAS_EXCEPTIONS()
    ++__first1;
    ++__first2;
  }
  __guard.__complete();
  return __first2;
}

template <class _Alloc, class _Type>
struct __allocator_has_trivial_move_construct : _Not<__has_construct<_Alloc, _Type*, _Type&&>>
{};

template <class _Type>
struct __allocator_has_trivial_move_construct<allocator<_Type>, _Type> : true_type
{};

#if !_CCCL_COMPILER(GCC)
template <class _Alloc,
          class _Iter1,
          class _Iter2,
          class _Type = typename iterator_traits<_Iter1>::value_type,
          class       = enable_if_t<_CCCL_TRAIT(is_trivially_move_constructible, _Type)
                                    && _CCCL_TRAIT(is_trivially_move_assignable, _Type)
                                    && __allocator_has_trivial_move_construct<_Alloc, _Type>::value>>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _Iter2
__uninitialized_allocator_move_if_noexcept(_Alloc&, _Iter1 __first1, _Iter1 __last1, _Iter2 __first2)
{
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    while (__first1 != __last1)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::__to_address(__first2), _CUDA_VSTD::move(*__first1));
      ++__first1;
      ++__first2;
    }
    return __first2;
  }
  else
  {
    return _CUDA_VSTD::move(__first1, __last1, __first2);
  }
}
#endif // !_CCCL_COMPILER(GCC)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_UNINITIALIZED_ALGORITHMS_H
