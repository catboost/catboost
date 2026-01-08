//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_COPY_CV_H
#define _LIBCUDACXX___TYPE_TRAITS_COPY_CV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/copy_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Let COPYCV(FROM, TO) be an alias for type TO with the addition of FROM's
// top-level cv-qualifiers.
template <class>
extern __apply_cvref_ __apply_cv;
template <class _Tp>
extern __apply_cvref_c __apply_cv<const _Tp>;
template <class _Tp>
extern __apply_cvref_v __apply_cv<volatile _Tp>;
template <class _Tp>
extern __apply_cvref_cv __apply_cv<const volatile _Tp>;

template <class _Tp>
using __apply_cv_fn = decltype(__apply_cv<_Tp>);

template <class _From, class _To>
using __copy_cv_t = typename __apply_cv_fn<_From>::template __call<_To>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_COPY_CV_H
