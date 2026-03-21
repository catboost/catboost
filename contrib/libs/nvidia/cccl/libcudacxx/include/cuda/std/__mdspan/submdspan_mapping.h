//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_SUBMDSPAN_MAPPING_H
#define _LIBCUDACXX___MDSPAN_SUBMDSPAN_MAPPING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/layout_left.h>
#include <cuda/std/__mdspan/layout_right.h>
#include <cuda/std/__mdspan/layout_stride.h>
#include <cuda/std/__mdspan/mdspan.h>
#include <cuda/std/__mdspan/submdspan_extents.h>
#include <cuda/std/__mdspan/submdspan_helper.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [mdspan.sub.map]

// [mdspan.submdspan.submdspan.mapping.result]
template <class _LayoutMapping>
struct submdspan_mapping_result
{
  static_assert(__mdspan_detail::__layout_mapping_req<_LayoutMapping>,
                "[mdspan.submdspan.submdspan.mapping.result] shall meet the layout mapping requirements");

  _CCCL_NO_UNIQUE_ADDRESS _LayoutMapping mapping{};
  size_t offset{};
};

// [mdspan.sub.map.common]
// mdspan.sub.map.common-2
template <class _Extents, class... _Slices>
_CCCL_CONCEPT __matching_number_of_slices = sizeof...(_Slices) == _Extents::rank();

template <size_t _SliceIndex, class _LayoutMapping, class... _Slices>
[[nodiscard]] _CCCL_API constexpr auto
__get_submdspan_strides(const _LayoutMapping& __mapping, _Slices... __slices) noexcept
{
  using _SliceType = __get_slice_type<_SliceIndex, _Slices...>;
  using _Extents   = typename _LayoutMapping::extents_type;
  using _IndexType = typename _Extents::index_type;
  if constexpr (__is_strided_slice<remove_cv_t<_SliceType>>)
  {
    _SliceType& __slice     = _CUDA_VSTD::__get_slice_at<_SliceIndex>(__slices...);
    using __unsigned_stride = make_unsigned_t<typename _SliceType::stride_type>;
    using __unsigned_extent = make_unsigned_t<typename _SliceType::extent_type>;
    return static_cast<_IndexType>(
      __mapping.stride(_SliceIndex)
      * (static_cast<__unsigned_stride>(__slice.stride) < static_cast<__unsigned_extent>(__slice.extent)
           ? _CUDA_VSTD::__de_ice(__slice.stride)
           : 1));
  }
  else
  {
    return static_cast<_IndexType>(__mapping.stride(_SliceIndex));
  }
}

template <class _LayoutMapping, class... _Slices, size_t... _SliceIndices>
[[nodiscard]] _CCCL_API constexpr auto
__submdspan_strides(index_sequence<_SliceIndices...>, const _LayoutMapping& __mapping, _Slices... __slices) noexcept
{
  using _Extents    = typename _LayoutMapping::extents_type;
  using _IndexType  = typename _Extents::index_type;
  using _SubExtents = __get_subextents_t<_Extents, _Slices...>;
  return array<_IndexType, _SubExtents::rank()>{
    _CUDA_VSTD::__get_submdspan_strides<_SliceIndices>(__mapping, __slices...)...};
}

_CCCL_TEMPLATE(class _LayoutMapping, class... _Slices)
_CCCL_REQUIRES(__matching_number_of_slices<typename _LayoutMapping::extents_type, _Slices...>)
[[nodiscard]] _CCCL_API constexpr auto __submdspan_strides(const _LayoutMapping& __mapping, _Slices... __slices)
{
  using _Extents                = typename _LayoutMapping::extents_type;
  using _IndexType              = typename _Extents::index_type;
  const auto __filtered_indices = __filter_slices_convertible_to_index<_IndexType, _Slices...>(
    index_sequence<>{}, _CUDA_VSTD::index_sequence_for<_Slices...>());
  return _CUDA_VSTD::__submdspan_strides(__filtered_indices, __mapping, __slices...);
}

// [mdspan.sub.map.common-8]
template <class _LayoutMapping, class... _Slices, size_t... _SliceIndices>
[[nodiscard]] _CCCL_API constexpr size_t
__submdspan_offset(index_sequence<_SliceIndices...>, const _LayoutMapping& __mapping, _Slices... __slices)
{
  using _Extents   = typename _LayoutMapping::extents_type;
  using _IndexType = typename _Extents::index_type;
  // If first_<index_type, k>(slices...)
  const array<_IndexType, _Extents::rank()> __offsets = {
    _CUDA_VSTD::__first_extent_from_slice<_IndexType, _SliceIndices>(__slices...)...};

  using _SubExtents = __get_subextents_t<_Extents, _Slices...>;
  for (size_t __index = 0; __index != _SubExtents::rank(); ++__index)
  {
    // If first_<index_type, k>(slices...) equals extents().extent(k) for any rank index k of extents()
    if (__offsets[__index] == __mapping.extents().extent(__index))
    {
      // then let offset be a value of type size_t equal to (*this).required_span_size()
      return static_cast<size_t>(__mapping.required_span_size());
    }
  }
  // Otherwise, let offset be a value of type size_t equal to (*this)(first_<index_type, P>(slices...)...).
  return static_cast<size_t>(__mapping(__offsets[_SliceIndices]...));
}

_CCCL_TEMPLATE(class _LayoutMapping, class... _Slices)
_CCCL_REQUIRES(__matching_number_of_slices<typename _LayoutMapping::extents_type, _Slices...>)
[[nodiscard]] _CCCL_API constexpr size_t __submdspan_offset(const _LayoutMapping& __mapping, _Slices... __slices)
{
  return _CUDA_VSTD::__submdspan_offset(_CUDA_VSTD::index_sequence_for<_Slices...>(), __mapping, __slices...);
}

// [mdspan.sub.map.common-9]
// [mdspan.sub.map.common-9.1]
template <class _SliceType>
_CCCL_CONCEPT __is_strided_slice_stride_of_one = _CCCL_REQUIRES_EXPR((_SliceType))(
  requires(__is_strided_slice<remove_cv_t<_SliceType>>),
  requires(__integral_constant_like<typename _SliceType::stride_type>),
  requires(_SliceType::stride_type::value == 1));

template <class _LayoutMapping, class _SliceType>
_CCCL_API constexpr bool __is_unit_stride_slice()
{
  // [mdspan.sub.map.common-9.1]
  if constexpr (__is_strided_slice_stride_of_one<_SliceType>)
  {
    return true;
  }
  // [mdspan.sub.map.common-9.2]
  else if constexpr (__index_pair_like<_SliceType, typename _LayoutMapping::index_type>)
  {
    return true;
  }
  // [mdspan.sub.map.common-9.3]
  else if constexpr (_CCCL_TRAIT(is_convertible, _SliceType, full_extent_t))
  {
    return true;
  }
  else
  {
    return false;
  }
  _CCCL_UNREACHABLE();
}

// [mdspan.sub.map.left]
template <class _LayoutMapping, class _SubExtents, class _Slice, class... _OtherSlices>
_CCCL_API constexpr bool __can_layout_left()
{
  // [mdspan.sub.map.left-1.2]
  if constexpr (_SubExtents::rank() == 0)
  {
    return true;
  }
  // [mdspan.sub.map.left-1.3.2]
  else if constexpr (sizeof...(_OtherSlices) == 0)
  {
    return _CUDA_VSTD::__is_unit_stride_slice<_LayoutMapping, _Slice>();
  }
  // [mdspan.sub.map.left-1.3.1]
  else if constexpr (_CCCL_TRAIT(is_convertible, _Slice, full_extent_t))
  {
    return _CUDA_VSTD::__can_layout_left<_LayoutMapping, _SubExtents, _OtherSlices...>();
  }
  else
  {
    return false;
  }
  _CCCL_UNREACHABLE();
}

_CCCL_TEMPLATE(class _Extents, class... _Slices)
_CCCL_REQUIRES(__matching_number_of_slices<_Extents, _Slices...>)
[[nodiscard]] _CCCL_API constexpr auto
__submdspan_mapping_impl(const typename layout_left::mapping<_Extents>& __mapping, _Slices... __slices)
{
  // [mdspan.sub.map.left-1.1]
  if constexpr (_Extents::rank() == 0)
  {
    return submdspan_mapping_result{__mapping, 0};
  }
  else
  {
    // [mdspan.sub.map.left-1.2]
    // [mdspan.sub.map.left-1.3]
    using _SubExtents    = __get_subextents_t<_Extents, _Slices...>;
    const auto __sub_ext = _CUDA_VSTD::submdspan_extents(__mapping.extents(), __slices...);
    const auto __offset  = _CUDA_VSTD::__submdspan_offset(__mapping, __slices...);
    if constexpr (_CUDA_VSTD::__can_layout_left<typename layout_left::mapping<_Extents>, _SubExtents, _Slices...>())
    {
      using __sub_mapping_t = layout_left::template mapping<_SubExtents>;
      return submdspan_mapping_result<__sub_mapping_t>{__sub_mapping_t{__sub_ext}, __offset};
    }
    // [mdspan.sub.map.left-1.4]
    // TODO: Implement padded layouts
    else
    {
      // [mdspan.sub.map.left-1.5]
      using __sub_mapping_t    = layout_stride::template mapping<_SubExtents>;
      const auto __sub_strides = _CUDA_VSTD::__submdspan_strides(__mapping, __slices...);
      return submdspan_mapping_result<__sub_mapping_t>{__sub_mapping_t{__sub_ext, __sub_strides}, __offset};
    }
  }
  _CCCL_UNREACHABLE();
}

template <class _LayoutMapping, class _SubExtents, class _Slice, class... _OtherSlices>
_CCCL_API constexpr bool __can_layout_right()
{
  // [mdspan.sub.map.right-1.2]
  if constexpr (_SubExtents::rank() == 0)
  {
    return true;
  }
  // [mdspan.sub.map.right-1.3.2]
  else if constexpr (sizeof...(_OtherSlices) == 0)
  {
    return _CUDA_VSTD::__is_unit_stride_slice<_LayoutMapping, _Slice>();
  }
  // [mdspan.sub.map.right-1.3.1]
  else if constexpr (_CCCL_TRAIT(is_convertible, _Slice, full_extent_t))
  {
    return _CUDA_VSTD::__can_layout_left<_LayoutMapping, _SubExtents, _OtherSlices...>();
  }
  else
  {
    return false;
  }
  _CCCL_UNREACHABLE();
}

_CCCL_TEMPLATE(class _Extents, class... _Slices)
_CCCL_REQUIRES(__matching_number_of_slices<_Extents, _Slices...>)
[[nodiscard]] _CCCL_API constexpr auto
__submdspan_mapping_impl(const typename layout_right::mapping<_Extents>& __mapping, _Slices... __slices)
{
  // [mdspan.sub.map.right-1.1]
  if constexpr (_Extents::rank() == 0)
  {
    return submdspan_mapping_result{__mapping, 0};
  }
  else
  {
    // [mdspan.sub.map.right-1.2]
    // [mdspan.sub.map.right-1.3]
    using _SubExtents    = __get_subextents_t<_Extents, _Slices...>;
    const auto __sub_ext = _CUDA_VSTD::submdspan_extents(__mapping.extents(), __slices...);
    const auto __offset  = _CUDA_VSTD::__submdspan_offset(__mapping, __slices...);
    if constexpr (_CUDA_VSTD::__can_layout_right<typename layout_left::mapping<_Extents>, _SubExtents, _Slices...>())
    {
      using __sub_mapping_t = layout_right::template mapping<_SubExtents>;
      return submdspan_mapping_result<__sub_mapping_t>{__sub_mapping_t{__sub_ext}, __offset};
    }
    // [mdspan.sub.map.right-1.4]
    // TODO: Implement padded layouts
    else
    {
      // [mdspan.sub.map.right-1.5]
      using __sub_mapping_t    = layout_stride::template mapping<_SubExtents>;
      const auto __sub_strides = _CUDA_VSTD::__submdspan_strides(__mapping, __slices...);
      return submdspan_mapping_result<__sub_mapping_t>{__sub_mapping_t{__sub_ext, __sub_strides}, __offset};
    }
  }
  _CCCL_UNREACHABLE();
}

_CCCL_TEMPLATE(class _Extents, class... _Slices)
_CCCL_REQUIRES(__matching_number_of_slices<_Extents, _Slices...>)
[[nodiscard]] _CCCL_API constexpr auto
__submdspan_mapping_impl(const typename layout_stride::mapping<_Extents>& __mapping, _Slices... __slices)
{
  // [mdspan.sub.map.stride-1.1]
  if constexpr (_Extents::rank() == 0)
  {
    return submdspan_mapping_result{__mapping, 0};
  }
  else
  {
    // [mdspan.sub.map.stride-1.2]
    using _SubExtents        = __get_subextents_t<_Extents, _Slices...>;
    using __sub_mapping_t    = layout_stride::template mapping<_SubExtents>;
    const auto __sub_ext     = _CUDA_VSTD::submdspan_extents(__mapping.extents(), __slices...);
    const auto __offset      = _CUDA_VSTD::__submdspan_offset(__mapping, __slices...);
    const auto __sub_strides = _CUDA_VSTD::__submdspan_strides(__mapping, __slices...);
    return submdspan_mapping_result<__sub_mapping_t>{__sub_mapping_t{__sub_ext, __sub_strides}, __offset};
  }
}

_CCCL_TEMPLATE(class _LayoutMapping, class... _Slices)
_CCCL_REQUIRES(__matching_number_of_slices<typename _LayoutMapping::extents_type, _Slices...>)
[[nodiscard]] _CCCL_API constexpr auto submdspan_mapping(const _LayoutMapping& __mapping, _Slices... __slices)
{
  return _CUDA_VSTD::__submdspan_mapping_impl(__mapping, __slices...);
}

// [mdspan.sub.sub]
template <class _LayoutMapping, class... _Slices>
_CCCL_CONCEPT __can_submdspan_mapping =
  _CCCL_REQUIRES_EXPR((_LayoutMapping, variadic _Slices), const _LayoutMapping& __mapping, _Slices... __slices)(
    (_CUDA_VSTD::submdspan_mapping(__mapping, __slices...)));

_CCCL_TEMPLATE(class _Tp, class _Extents, class _Layout, class _Accessor, class... _Slices)
_CCCL_REQUIRES(__matching_number_of_slices<_Extents, _Slices...> _CCCL_AND
                 __can_submdspan_mapping<typename _Layout::template mapping<_Extents>, _Slices...>)
[[nodiscard]] _CCCL_API constexpr auto
submdspan(const mdspan<_Tp, _Extents, _Layout, _Accessor>& __src, _Slices... __slices)
{
  auto __sub_map_result = _CUDA_VSTD::submdspan_mapping(__src.mapping(), __slices...);
  return mdspan(__src.accessor().offset(__src.data_handle(), __sub_map_result.offset),
                __sub_map_result.mapping,
                typename _Accessor::offset_policy(__src.accessor()));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_SUBMDSPAN_MAPPING_H
