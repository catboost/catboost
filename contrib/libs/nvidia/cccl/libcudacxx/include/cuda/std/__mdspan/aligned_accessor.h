//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
// ************************************************************************
//@HEADER

#ifndef _LIBCUDACXX___MDSPAN_ALIGNED_ACCESSOR_H
#define _LIBCUDACXX___MDSPAN_ALIGNED_ACCESSOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__type_traits/is_abstract.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_object.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ElementType, size_t _ByteAlignment>
class aligned_accessor
{
public:
  static constexpr auto byte_alignment = _ByteAlignment;

  static_assert(_CUDA_VSTD::has_single_bit(byte_alignment), "byte_alignment must be a power of two.");

  static_assert(byte_alignment >= alignof(_ElementType), "Insufficient byte alignment for _ElementType");

  static_assert(_CCCL_TRAIT(is_object, _ElementType) && !_CCCL_TRAIT(is_abstract, _ElementType)
                  && !_CCCL_TRAIT(is_array, _ElementType),
                "_ElementType must be a complete object type that is neither an abstract class type nor an array "
                "type.");

  using offset_policy    = default_accessor<_ElementType>;
  using element_type     = _ElementType;
  using reference        = _ElementType&;
  using data_handle_type = _ElementType*;

  _CCCL_HIDE_FROM_ABI aligned_accessor() noexcept = default;

  _CCCL_TEMPLATE(class _OtherElementType, size_t _OtherByteAlignment)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _OtherElementType (*)[], element_type (*)[])
                   _CCCL_AND((_OtherByteAlignment >= byte_alignment)))
  _CCCL_API constexpr aligned_accessor(aligned_accessor<_OtherElementType, _OtherByteAlignment>) noexcept {}

  _CCCL_TEMPLATE(class _OtherElementType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _CCCL_API constexpr explicit aligned_accessor(default_accessor<_OtherElementType>) noexcept {}

  _CCCL_TEMPLATE(class _OtherElementType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _CCCL_API constexpr operator default_accessor<_OtherElementType>() const noexcept
  {
    return {};
  }

  _CCCL_API constexpr reference access(data_handle_type __p, size_t __i) const noexcept
  {
    return _CUDA_VSTD::assume_aligned<byte_alignment>(__p)[__i];
  }

  _CCCL_API constexpr typename offset_policy::data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return _CUDA_VSTD::assume_aligned<byte_alignment>(__p) + __i;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_ALIGNED_ACCESSOR_H
