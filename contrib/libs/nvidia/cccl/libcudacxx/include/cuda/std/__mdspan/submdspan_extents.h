//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_SUBMDSPAN_EXTENTS_H
#define _LIBCUDACXX___MDSPAN_SUBMDSPAN_EXTENTS_H

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
#include <cuda/std/__mdspan/submdspan_helper.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/tuple>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Helper to get an index_sequence of all slices that are not convertible to index_type
template <class _IndexType, class... _Slices, size_t... _FilteredIndices>
[[nodiscard]] _CCCL_API constexpr auto
__filter_slices_convertible_to_index(index_sequence<_FilteredIndices...>, index_sequence<>) noexcept
{
  return index_sequence<_FilteredIndices...>{};
}

template <class _IndexType, class... _Slices, size_t... _SliceIndices, size_t _CurrentIndex, size_t... _Remaining>
[[nodiscard]] _CCCL_API constexpr auto __filter_slices_convertible_to_index(
  index_sequence<_SliceIndices...>, index_sequence<_CurrentIndex, _Remaining...>) noexcept
{
  using _SliceType = __get_slice_type<_CurrentIndex, _Slices...>;
  if constexpr (convertible_to<_SliceType, _IndexType>)
  {
    return _CUDA_VSTD::__filter_slices_convertible_to_index<_IndexType, _Slices...>(
      index_sequence<_SliceIndices...>{}, index_sequence<_Remaining...>{});
  }
  else
  {
    return _CUDA_VSTD::__filter_slices_convertible_to_index<_IndexType, _Slices...>(
      index_sequence<_SliceIndices..., _CurrentIndex>{}, index_sequence<_Remaining...>{});
  }
  _CCCL_UNREACHABLE();
}

// [mdspan.sub.extents]
// [mdspan.sub.extents-4.2.2]
template <class _Extents, class _SliceType>
_CCCL_CONCEPT __subextents_is_index_pair = _CCCL_REQUIRES_EXPR((_Extents, _SliceType))(
  requires(__index_pair_like<_SliceType, typename _Extents::index_type>),
  requires(__integral_constant_like<tuple_element_t<0, _SliceType>>),
  requires(__integral_constant_like<tuple_element_t<1, _SliceType>>));

// [mdspan.sub.extents-4.2.3]
template <class _Extents, class _SliceType>
_CCCL_CONCEPT __subextents_is_strided_slice_zero_extent = _CCCL_REQUIRES_EXPR((_Extents, _SliceType))(
  requires(__is_strided_slice<remove_cv_t<_SliceType>>),
  requires(__integral_constant_like<typename _SliceType::extent_type>),
  requires(typename _SliceType::extent_type() == 0));

// [mdspan.sub.extents-4.2.4]
template <class _SliceType>
_CCCL_CONCEPT __subextents_is_strided_slice = _CCCL_REQUIRES_EXPR((_SliceType))(
  requires(__is_strided_slice<remove_cv_t<_SliceType>>),
  requires(__integral_constant_like<typename _SliceType::extent_type>),
  requires(__integral_constant_like<typename _SliceType::stride_type>));

struct __get_subextent
{
  template <class _Extents, size_t _SliceIndex, class _SliceType>
  [[nodiscard]] _CCCL_API static constexpr size_t __get_static_subextents() noexcept
  {
    // [mdspan.sub.extents-4.2.1]
    if constexpr (convertible_to<_SliceType, full_extent_t>)
    {
      return _Extents::static_extent(_SliceIndex);
    }
    // [mdspan.sub.extents-4.2.2]
    else if constexpr (__subextents_is_index_pair<_Extents, _SliceType>)
    {
      return _CUDA_VSTD::__de_ice(tuple_element_t<1, _SliceType>())
           - _CUDA_VSTD::__de_ice(tuple_element_t<0, _SliceType>());
    }
    // [mdspan.sub.extents-4.2.3]
    else if constexpr (__subextents_is_strided_slice_zero_extent<_Extents, _SliceType>)
    {
      return 0;
    }
    // [mdspan.sub.extents-4.2.4]
    else if constexpr (__subextents_is_strided_slice<_SliceType>)
    {
      return 1
           + (_CUDA_VSTD::__de_ice(typename _SliceType::extent_type()) - 1)
               / _CUDA_VSTD::__de_ice(typename _SliceType::stride_type());
    }
    else
    {
      // [mdspan.sub.extents-4.2.5]
      return dynamic_extent;
    }
    _CCCL_UNREACHABLE();
  }

  template <size_t _SliceIndex, class _Extents, class... _Slices>
  [[nodiscard]] _CCCL_API static constexpr typename _Extents::index_type
  __get_dynamic_subextents(const _Extents& __src, _Slices... __slices) noexcept
  {
    using _SliceType = __get_slice_type<_SliceIndex, _Slices...>;
    // [mdspan.sub.extents-5.1]
    if constexpr (__is_strided_slice<remove_cv_t<_SliceType>>)
    {
      _SliceType& __slice = _CUDA_VSTD::__get_slice_at<_SliceIndex>(__slices...);
      return __slice.extent == 0
             ? 0
             : 1 + (_CUDA_VSTD::__de_ice(__slice.extent) - 1) / _CUDA_VSTD::__de_ice(__slice.stride);
    }
    // [mdspan.sub.extents-5.2]
    else
    {
      return _CUDA_VSTD::__last_extent_from_slice<_SliceIndex>(__src, __slices...)
           - _CUDA_VSTD::__first_extent_from_slice<typename _Extents::index_type, _SliceIndex>(__slices...);
    }
    _CCCL_UNREACHABLE();
  }

  template <class _Extents, class... _Slices, size_t... _SliceIndices>
  [[nodiscard]] _CCCL_API constexpr auto
  __impl(index_sequence<_SliceIndices...>, const _Extents& __src, _Slices... __slices) noexcept
  {
    using _IndexType = typename _Extents::index_type;
    using _SubExtents =
      extents<_IndexType,
              __get_static_subextents<_Extents, _SliceIndices, __get_slice_type<_SliceIndices, _Slices...>>()...>;
    return _SubExtents{__get_dynamic_subextents<_SliceIndices>(__src, __slices...)...};
  }

  template <class _Extents, class... _Slices>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Extents& __src, _Slices... __slices) noexcept
  {
    const auto __filtered_indices = __filter_slices_convertible_to_index<typename _Extents::index_type, _Slices...>(
      index_sequence<>{}, _CUDA_VSTD::index_sequence_for<_Slices...>());
    return __impl(__filtered_indices, __src, __slices...);
  }
};

template <class _IndexType, class _SliceType>
inline constexpr bool __is_valid_subextents =
  convertible_to<_SliceType, _IndexType> || __index_pair_like<_SliceType, _IndexType>
  || _CCCL_TRAIT(is_convertible, _SliceType, full_extent_t) || __is_strided_slice<remove_cv_t<_SliceType>>;

_CCCL_TEMPLATE(class _Extents, class... _Slices)
_CCCL_REQUIRES((_Extents::rank() == sizeof...(_Slices)))
[[nodiscard]] _CCCL_API constexpr auto submdspan_extents(const _Extents& __src, _Slices... __slices)
{
  static_assert(((__is_valid_subextents<typename _Extents::index_type, _Slices>) && ... && true),
                "[mdspan.sub.extents] For each rank index k of src.extents(), exactly one of the following is true:");
  return __get_subextent{}(__src, __slices...);
}

template <class _Extents, class... _Slices>
using __get_subextents_t =
  decltype(_CUDA_VSTD::submdspan_extents(_CUDA_VSTD::declval<_Extents>(), _CUDA_VSTD::declval<_Slices>()...));

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_SUBMDSPAN_EXTENTS_H
