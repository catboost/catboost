//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_UNIQUE_COPY_H
#define _LIBCUDACXX___ALGORITHM_UNIQUE_COPY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __unique_copy_tags
{

struct __reread_from_input_tag
{};
struct __reread_from_output_tag
{};
struct __read_from_tmp_value_tag
{};

} // namespace __unique_copy_tags

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BinaryPredicate, class _InputIterator, class _Sent, class _OutputIterator>
constexpr _CCCL_API inline pair<_InputIterator, _OutputIterator> __unique_copy(
  _InputIterator __first,
  _Sent __last,
  _OutputIterator __result,
  _BinaryPredicate&& __pred,
  __unique_copy_tags::__read_from_tmp_value_tag)
{
  if (__first != __last)
  {
    typename _IterOps<_AlgPolicy>::template __value_type<_InputIterator> __t(*__first);
    *__result = __t;
    ++__result;
    while (++__first != __last)
    {
      if (!__pred(__t, *__first))
      {
        __t       = *__first;
        *__result = __t;
        ++__result;
      }
    }
  }
  return pair<_InputIterator, _OutputIterator>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__result));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BinaryPredicate, class _ForwardIterator, class _Sent, class _OutputIterator>
constexpr _CCCL_API inline pair<_ForwardIterator, _OutputIterator> __unique_copy(
  _ForwardIterator __first,
  _Sent __last,
  _OutputIterator __result,
  _BinaryPredicate&& __pred,
  __unique_copy_tags::__reread_from_input_tag)
{
  if (__first != __last)
  {
    _ForwardIterator __i = __first;
    *__result            = *__i;
    ++__result;
    while (++__first != __last)
    {
      if (!__pred(*__i, *__first))
      {
        *__result = *__first;
        ++__result;
        __i = __first;
      }
    }
  }
  return pair<_ForwardIterator, _OutputIterator>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__result));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BinaryPredicate, class _InputIterator, class _Sent, class _InputAndOutputIterator>
constexpr _CCCL_API inline pair<_InputIterator, _InputAndOutputIterator> __unique_copy(
  _InputIterator __first,
  _Sent __last,
  _InputAndOutputIterator __result,
  _BinaryPredicate&& __pred,
  __unique_copy_tags::__reread_from_output_tag)
{
  if (__first != __last)
  {
    *__result = *__first;
    while (++__first != __last)
    {
      if (!__pred(*__result, *__first))
      {
        *++__result = *__first;
      }
    }
    ++__result;
  }
  return pair<_InputIterator, _InputAndOutputIterator>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__result));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator, class _OutputIterator, class _BinaryPredicate>
_CCCL_API constexpr _OutputIterator
unique_copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryPredicate __pred)
{
  using __algo_tag = conditional_t<
    _CCCL_TRAIT(is_base_of, forward_iterator_tag, __iterator_category_type<_InputIterator>),
    __unique_copy_tags::__reread_from_input_tag,
    conditional_t<_CCCL_TRAIT(is_base_of, forward_iterator_tag, __iterator_category_type<_OutputIterator>)
                    && _CCCL_TRAIT(is_same, __iter_value_type<_InputIterator>, __iter_value_type<_OutputIterator>),
                  __unique_copy_tags::__reread_from_output_tag,
                  __unique_copy_tags::__read_from_tmp_value_tag>>;
  return _CUDA_VSTD::__unique_copy<_ClassicAlgPolicy>(
           _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), _CUDA_VSTD::move(__result), __pred, __algo_tag())
    .second;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator, class _OutputIterator>
_CCCL_API constexpr _OutputIterator unique_copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  return _CUDA_VSTD::unique_copy(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), _CUDA_VSTD::move(__result), __equal_to{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_UNIQUE_COPY_H
