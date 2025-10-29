//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_SUBMDSPAN_HELPER_H
#define _LIBCUDACXX___MDSPAN_SUBMDSPAN_HELPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/tuple>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [mdspan.sub.overview]-2.5
template <class _IndexType, class... _SliceTypes>
[[nodiscard]] _CCCL_API constexpr array<size_t, sizeof...(_SliceTypes)> __map_rank(size_t __count = 0) noexcept
{
  return {(convertible_to<_SliceTypes, _IndexType> ? dynamic_extent : __count++)...};
}

// [mdspan.submdspan.strided.slice]
template <class _OffsetType, class _ExtentType, class _StrideType>
struct strided_slice
{
  using offset_type = _OffsetType;
  using extent_type = _ExtentType;
  using stride_type = _StrideType;

  static_assert(__index_like<offset_type>,
                "[mdspan.submdspan.strided.slice] cuda::std::strided_slice::offset_type must be signed or unsigned or "
                "integral-constant-like");
  static_assert(__index_like<extent_type>,
                "[mdspan.submdspan.strided.slice] cuda::std::strided_slice::extent_type must be signed or unsigned or "
                "integral-constant-like");
  static_assert(__index_like<stride_type>,
                "[mdspan.submdspan.strided.slice] cuda::std::strided_slice::stride_type must be signed or unsigned or "
                "integral-constant-like");

  _CCCL_NO_UNIQUE_ADDRESS offset_type offset{};
  _CCCL_NO_UNIQUE_ADDRESS extent_type extent{};
  _CCCL_NO_UNIQUE_ADDRESS stride_type stride{};
};

template <class _OffsetType, class _ExtentType, class _StrideType>
_CCCL_HOST_DEVICE strided_slice(_OffsetType, _ExtentType, _StrideType)
  -> strided_slice<_OffsetType, _ExtentType, _StrideType>;

template <typename>
inline constexpr bool __is_strided_slice = false;

template <class _OffsetType, class _ExtentType, class _StrideType>
inline constexpr bool __is_strided_slice<strided_slice<_OffsetType, _ExtentType, _StrideType>> = true;

struct full_extent_t
{
  _CCCL_HIDE_FROM_ABI explicit full_extent_t() = default;
};
inline constexpr full_extent_t full_extent{};

// [mdspan.submdspan.helpers]
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__integral_constant_like<_Tp>) )
[[nodiscard]] _CCCL_API constexpr _Tp __de_ice(_Tp __val) noexcept
{
  return __val;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__integral_constant_like<_Tp>)
[[nodiscard]] _CCCL_API constexpr auto __de_ice(_Tp) noexcept
{
  return _Tp::value;
}

template <class _IndexType, class _From>
[[nodiscard]] _CCCL_API constexpr auto __index_cast(_From&& __from) noexcept
{
  if constexpr (_CCCL_TRAIT(is_integral, _From) && !_CCCL_TRAIT(is_same, _From, bool))
  {
    return __from;
  }
  else
  {
    return static_cast<_IndexType>(__from);
  }
}

template <size_t _Index, class... _Slices>
[[nodiscard]] _CCCL_API constexpr decltype(auto) __get_slice_at(_Slices&&... __slices) noexcept
{
  return _CUDA_VSTD::get<_Index>(_CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::forward<_Slices>(__slices)...));
}

template <size_t _Index, class... _Slices>
using __get_slice_type = tuple_element_t<_Index, __tuple_types<_Slices...>>;

template <class _IndexType, size_t _Index, class... _Slices>
[[nodiscard]] _CCCL_API constexpr _IndexType __first_extent_from_slice(_Slices... __slices) noexcept
{
  static_assert(_CCCL_TRAIT(is_signed, _IndexType) || _CCCL_TRAIT(is_unsigned, _IndexType),
                "[mdspan.sub.helpers] mandates IndexType to be a signed or unsigned integral");
  using _SliceType                     = __get_slice_type<_Index, _Slices...>;
  [[maybe_unused]] _SliceType& __slice = _CUDA_VSTD::__get_slice_at<_Index>(__slices...);
  if constexpr (convertible_to<_SliceType, _IndexType>)
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(__slice);
  }
  else
  {
    if constexpr (__index_pair_like<_SliceType, _IndexType>)
    {
      return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::get<0>(__slice));
    }
    else if constexpr (__is_strided_slice<_SliceType>)
    {
      return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::__de_ice(__slice.offset));
    }
    else
    {
      return 0;
    }
  }
  _CCCL_UNREACHABLE();
}

template <size_t _Index, class _Extents, class... _Slices>
[[nodiscard]] _CCCL_API constexpr typename _Extents::index_type
__last_extent_from_slice(const _Extents& __src, _Slices... __slices) noexcept
{
  static_assert(_CCCL_TRAIT(__mdspan_detail::__is_extents, _Extents),
                "[mdspan.sub.helpers] mandates Extents to be a specialization of extents");
  using _IndexType                     = typename _Extents::index_type;
  using _SliceType                     = __get_slice_type<_Index, _Slices...>;
  [[maybe_unused]] _SliceType& __slice = _CUDA_VSTD::__get_slice_at<_Index>(__slices...);
  if constexpr (convertible_to<_SliceType, _IndexType>)
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::__de_ice(__slice) + 1);
  }
  else
  {
    if constexpr (__index_pair_like<_SliceType, _IndexType>)
    {
      return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::get<1>(__slice));
    }
    else if constexpr (__is_strided_slice<_SliceType>)
    {
      return _CUDA_VSTD::__index_cast<_IndexType>(
        _CUDA_VSTD::__de_ice(__slice.offset) * _CUDA_VSTD::__de_ice(__slice.extent));
    }
    else
    {
      return _CUDA_VSTD::__index_cast<_IndexType>(__src.extent(_Index));
    }
  }
  _CCCL_UNREACHABLE();
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_SUBMDSPAN_HELPER_H
