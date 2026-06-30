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

#ifndef _LIBCUDACXX___MEMORY_USES_ALLOCATOR_H
#define _LIBCUDACXX___MEMORY_USES_ALLOCATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, class = void>
inline constexpr bool __has_allocator_type_v = false;
template <class _Tp>
inline constexpr bool __has_allocator_type_v<_Tp, void_t<typename _Tp::allocator_type>> = true;

template <class _Tp, class _Alloc, bool = _CCCL_TRAIT(__has_allocator_type, _Tp)>
inline constexpr bool __uses_allocator_v = false;
template <class _Tp, class _Alloc>
inline constexpr bool __uses_allocator_v<_Tp, _Alloc, true> = is_convertible_v<_Alloc, typename _Tp::allocator_type>;

template <class _Tp, class _Alloc>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
uses_allocator : public integral_constant<bool, _CCCL_TRAIT(__uses_allocator, _Tp, _Alloc)>
{};

template <class _Tp, class _Alloc>
inline constexpr bool uses_allocator_v = _CCCL_TRAIT(__uses_allocator, _Tp, _Alloc);

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_USES_ALLOCATOR_H
