//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MINMAX_H
#define _LIBCUDACXX___ALGORITHM_MINMAX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/minmax_element.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr pair<const _Tp&, const _Tp&> minmax(const _Tp& __a, const _Tp& __b, _Compare __comp)
{
  return __comp(__b, __a) ? pair<const _Tp&, const _Tp&>(__b, __a) : pair<const _Tp&, const _Tp&>(__a, __b);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr pair<const _Tp&, const _Tp&> minmax(const _Tp& __a, const _Tp& __b)
{
  return _CUDA_VSTD::minmax(__a, __b, __less{});
}

template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr pair<_Tp, _Tp> minmax(initializer_list<_Tp> __t, _Compare __comp)
{
  static_assert(__is_callable<_Compare, _Tp, _Tp>::value, "The comparator has to be callable");
  identity __proj{};
  auto __ret = _CUDA_VSTD::__minmax_element_impl(__t.begin(), __t.end(), __comp, __proj);
  return pair<_Tp, _Tp>(*__ret.first, *__ret.second);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr pair<_Tp, _Tp> minmax(initializer_list<_Tp> __t)
{
  return _CUDA_VSTD::minmax(__t, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_MINMAX_H
