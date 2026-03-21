// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_LAYOUT_LEFT_H
#define _LIBCUDACXX___MDSPAN_LAYOUT_LEFT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/empty_base.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Helper for lightweight test checking that one did pass a layout policy as LayoutPolicy template argument
template <class _Extents>
class _CCCL_DECLSPEC_EMPTY_BASES layout_left::mapping : private __mdspan_ebco<_Extents>
{
public:
  static_assert(__mdspan_detail::__is_extents<_Extents>::value,
                "layout_left::mapping template argument must be a specialization of extents.");

  using extents_type = _Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_left;
  using __base       = __mdspan_ebco<_Extents>;

  template <class, class, class, class>
  friend class mdspan;

private:
  [[nodiscard]] _CCCL_API static constexpr bool __mul_overflow(index_type x, index_type y, index_type* res) noexcept
  {
    *res = x * y;
    return x && ((*res / x) != y);
  }

  [[nodiscard]] _CCCL_API static constexpr bool __required_span_size_is_representable(const extents_type& __ext)
  {
    if constexpr (extents_type::rank() != 0)
    {
      index_type __prod = __ext.extent(0);
      for (rank_type __r = 1; __r < extents_type::rank(); __r++)
      {
        if (__mul_overflow(__prod, __ext.extent(__r), &__prod))
        {
          return false;
        }
      }
    }
    return true;
  }

  static_assert((extents_type::rank_dynamic() > 0) || __required_span_size_is_representable(extents_type()),
                "layout_left::mapping product of static extents must be representable as index_type.");

public:
  // [mdspan.layout.left.cons], constructors
  _CCCL_HIDE_FROM_ABI constexpr mapping() noexcept               = default;
  _CCCL_HIDE_FROM_ABI constexpr mapping(const mapping&) noexcept = default;

  _CCCL_API constexpr mapping(const extents_type& __ext) noexcept
      : __base(__ext)
  {
    // not catching this could lead to out-of-bounds access later when used inside mdspan
    // mapping<dextents<char, 2>> map(dextents<char, 2>(40,40)); map(10, 3) == -126
    _CCCL_ASSERT(__required_span_size_is_representable(__ext),
                 "layout_left::mapping extents ctor: product of extents must be representable as index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents)
                   _CCCL_AND _CCCL_TRAIT(is_convertible, _OtherExtents, extents_type))
  _CCCL_API constexpr mapping(const mapping<_OtherExtents>& __other) noexcept
      : __base(__other.extents())
  {
    // not catching this could lead to out-of-bounds access later when used inside mdspan
    // mapping<dextents<char, 2>> map(mapping<dextents<int, 2>>(dextents<int, 2>(40,40))); map(10, 3) == -126
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left::mapping converting ctor: other.required_span_size() must be representable as "
                 "index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents)
                   _CCCL_AND(!_CCCL_TRAIT(is_convertible, _OtherExtents, extents_type)))
  _CCCL_API explicit constexpr mapping(const mapping<_OtherExtents>& __other) noexcept
      : __base(__other.extents())
  {
    // not catching this could lead to out-of-bounds access later when used inside mdspan
    // mapping<dextents<char, 2>> map(mapping<dextents<int, 2>>(dextents<int, 2>(40,40))); map(10, 3) == -126
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left::mapping converting ctor: other.required_span_size() must be representable as "
                 "index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES((_OtherExtents::rank() <= 1) _CCCL_AND _CCCL_TRAIT(is_constructible, extents_type, _OtherExtents)
                   _CCCL_AND _CCCL_TRAIT(is_convertible, _OtherExtents, extents_type))
  _CCCL_API constexpr mapping(const layout_right::mapping<_OtherExtents>& __other) noexcept
      : __base(__other.extents())
  {
    // not catching this could lead to out-of-bounds access later when used inside mdspan
    // Note: since this is constraint to rank 1, extents itself would catch the invalid conversion first
    //       and thus this assertion should never be triggered, but keeping it here for consistency
    // layout_left::mapping<dextents<char, 1>> map(
    //           layout_right::mapping<dextents<unsigned, 1>>(dextents<unsigned, 1>(200))); map.extents().extent(0) ==
    //           -56
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left::mapping converting ctor: other.required_span_size() must be representable as "
                 "index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES((_OtherExtents::rank() <= 1) _CCCL_AND _CCCL_TRAIT(is_constructible, extents_type, _OtherExtents)
                   _CCCL_AND(!_CCCL_TRAIT(is_convertible, _OtherExtents, extents_type)))
  _CCCL_API explicit constexpr mapping(const layout_right::mapping<_OtherExtents>& __other) noexcept
      : __base(__other.extents())
  {
    // not catching this could lead to out-of-bounds access later when used inside mdspan
    // Note: since this is constraint to rank 1, extents itself would catch the invalid conversion first
    //       and thus this assertion should never be triggered, but keeping it here for consistency
    // layout_left::mapping<dextents<char, 1>> map(
    //           layout_right::mapping<dextents<unsigned, 1>>(dextents<unsigned, 1>(200))); map.extents().extent(0) ==
    //           -56
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left::mapping converting ctor: other.required_span_size() must be representable as "
                 "index_type.");
  }

  template <class _OtherMappping>
  [[nodiscard]] _CCCL_API constexpr bool __check_strides(const _OtherMappping& __other) const noexcept
  {
    // avoid warning when comparing signed and unsigner integers and pick the wider of two types
    using _CommonType = common_type_t<index_type, typename _OtherMappping::index_type>;
    for (rank_type __r = 0; __r != extents_type::rank(); __r++)
    {
      if (static_cast<_CommonType>(stride(__r)) != static_cast<_CommonType>(__other.stride(__r)))
      {
        return false;
      }
    }
    return true;
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents) _CCCL_AND(extents_type::rank() > 0))
  _CCCL_API explicit constexpr mapping(const layout_stride::mapping<_OtherExtents>& __other) noexcept
      : __base(__other.extents())
  {
    _CCCL_ASSERT(__check_strides(__other),
                 "layout_left::mapping from layout_stride ctor: strides are not compatible with layout_left.");
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left::mapping from layout_stride ctor: other.required_span_size() must be representable as "
                 "index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents) _CCCL_AND(extents_type::rank() == 0))
  _CCCL_API constexpr mapping(const layout_stride::mapping<_OtherExtents>& __other) noexcept
      : __base(__other.extents())
  {}

  _CCCL_HIDE_FROM_ABI constexpr mapping& operator=(const mapping&) noexcept = default;

  // [mdspan.layout.left.obs], observers
  [[nodiscard]] _CCCL_API constexpr const extents_type& extents() const noexcept
  {
    return this->template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr index_type required_span_size() const noexcept
  {
    index_type __size = 1;
    if constexpr (extents_type::rank() != 0)
    {
      for (size_t __r = 0; __r != extents_type::rank(); __r++)
      {
        __size *= extents().extent(__r);
      }
    }
    return __size;
  }

  template <size_t... _Pos>
  [[nodiscard]] _CCCL_API constexpr index_type
  __op_index(const array<index_type, _Extents::rank()>& __idx_a, index_sequence<_Pos...>) const noexcept
  {
    index_type __res = 0;
    ((__res = __idx_a[extents_type::rank() - 1 - _Pos] + extents().extent(extents_type::rank() - 1 - _Pos) * __res),
     ...);
    return __res;
  }
  [[nodiscard]] _CCCL_API constexpr index_type
  __op_index(const array<index_type, extents_type::rank()>&, index_sequence<>) const noexcept
  {
    return 0;
  }

  _CCCL_TEMPLATE(class... _Indices)
  _CCCL_REQUIRES((sizeof...(_Indices) == extents_type::rank())
                   _CCCL_AND __mdspan_detail::__all_convertible_to_index_type<index_type, _Indices...>)
  [[nodiscard]] _CCCL_API constexpr index_type operator()(_Indices... __idx) const noexcept
  {
    // Mappings are generally meant to be used for accessing allocations and are meant to guarantee to never
    // return a value exceeding required_span_size(), which is used to know how large an allocation one needs
    // Thus, this is a canonical point in multi-dimensional data structures to make invalid element access checks
    // However, mdspan does check this on its own, so for now we avoid double checking in hardened mode
    _CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(extents(), __idx...),
                 "layout_left::mapping: out of bounds indexing");

    const array<index_type, extents_type::rank()> __idx_a{static_cast<index_type>(__idx)...};
    return __op_index(__idx_a, make_index_sequence<sizeof...(_Indices)>());
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_always_unique() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_always_exhaustive() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_always_strided() noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_unique() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_exhaustive() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_strided() noexcept
  {
    return true;
  }

  _CCCL_TEMPLATE(class _Extents2 = _Extents)
  _CCCL_REQUIRES((_Extents2::rank() > 0))
  [[nodiscard]] _CCCL_API constexpr index_type stride(rank_type __r) const noexcept
  {
    // While it would be caught by extents itself too, using a too large __r
    // is functionally an out of bounds access on the stored information needed to compute strides
    _CCCL_ASSERT(__r < extents_type::rank(), "layout_left::mapping::stride(): invalid rank index");
    index_type __s = 1;
    for (rank_type __i = 0; __i < __r; __i++)
    {
      __s *= extents().extent(__i);
    }
    return __s;
  }

  template <class _OtherExtents, class _Extents2 = _Extents>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator==(const mapping& __lhs, const mapping<_OtherExtents>& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)((_OtherExtents::rank() == _Extents2::rank()))
  {
    return __lhs.extents() == __rhs.extents();
  }

#if _CCCL_STD_VER <= 2017
  template <class _OtherExtents, class _Extents2 = _Extents>
  [[nodiscard]]
  _CCCL_API friend constexpr auto operator!=(const mapping& __lhs, const mapping<_OtherExtents>& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)((_OtherExtents::rank() == _Extents2::rank()))
  {
    return __lhs.extents() != __rhs.extents();
  }
#endif // _CCCL_STD_VER <= 2017
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_LAYOUT_LEFT_H
