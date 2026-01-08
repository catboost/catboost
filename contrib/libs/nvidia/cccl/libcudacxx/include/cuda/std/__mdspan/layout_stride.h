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

#ifndef _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_H
#define _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_H

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
#include <cuda/std/__mdspan/empty_base.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/as_const.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __layout_stride_detail
{

template <class _StridedLayoutMapping, class _Extents>
_CCCL_CONCEPT __can_convert = _CCCL_REQUIRES_EXPR((_StridedLayoutMapping, _Extents))(
  requires(__mdspan_detail::__layout_mapping_alike<_StridedLayoutMapping>),
  requires(_StridedLayoutMapping::is_always_unique()),
  requires(_StridedLayoutMapping::is_always_strided()),
  requires(_CCCL_TRAIT(is_constructible, _Extents, typename _StridedLayoutMapping::extents_type)));

struct __constraints
{
  template <class _StridedLayoutMapping, class _Extents>
  static constexpr bool __converts_implicit =
    _CCCL_TRAIT(is_convertible, typename _StridedLayoutMapping::extents_type, _Extents)
    && (__mdspan_detail::__is_mapping_of<layout_left, _StridedLayoutMapping>
        || __mdspan_detail::__is_mapping_of<layout_right, _StridedLayoutMapping>
        || __mdspan_detail::__is_mapping_of<layout_stride, _StridedLayoutMapping>);
};

} // namespace __layout_stride_detail

template <class _Extents>
class _CCCL_DECLSPEC_EMPTY_BASES layout_stride::mapping
    : private __mdspan_ebco<_Extents,
                            __mdspan_detail::__possibly_empty_array<typename _Extents::index_type, _Extents::rank()>>
{
public:
  static_assert(__mdspan_detail::__is_extents<_Extents>::value,
                "layout_stride::mapping template argument must be a specialization of extents.");

  using extents_type = _Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_stride;
  using __base = __mdspan_ebco<_Extents, __mdspan_detail::__possibly_empty_array<index_type, extents_type::rank()>>;

  template <class, class, class, class>
  friend class mdspan;

private:
  static constexpr rank_type __rank_    = extents_type::rank();
  static constexpr auto __rank_sequence = _CUDA_VSTD::make_index_sequence<extents_type::rank()>();

  using __stride_array = __mdspan_detail::__possibly_empty_array<index_type, extents_type::rank()>;

  // Used for default construction check and mandates
  [[nodiscard]] _CCCL_API static constexpr bool
  __mul_overflow(index_type __x, index_type __y, index_type* __res) noexcept
  {
    *__res = __x * __y;
    return __x && ((*__res / __x) != __y);
  }
  [[nodiscard]] _CCCL_API static constexpr bool
  __add_overflow(index_type __x, index_type __y, index_type* __res) noexcept
  {
    *__res = __x + __y;
    return *__res < __y;
  }

  [[nodiscard]] _CCCL_API static constexpr bool __required_span_size_is_representable(const extents_type& __ext) noexcept
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

  template <class _OtherIndexType>
  [[nodiscard]] _CCCL_API static constexpr bool
  __conversion_may_overflow([[maybe_unused]] _OtherIndexType __stride) noexcept
  {
    // nvcc believes stride is unused here
    if constexpr (_CCCL_TRAIT(is_integral, _OtherIndexType))
    {
      using _CommonType = common_type_t<index_type, _OtherIndexType>;
      return static_cast<_CommonType>(__stride) > static_cast<_CommonType>((numeric_limits<index_type>::max)());
    }
    else
    {
      return false;
    }
    _CCCL_UNREACHABLE();
  }

  template <class _OtherIndexType>
  [[nodiscard]] _CCCL_API static constexpr bool __required_span_size_is_representable(
    const extents_type& __ext, [[maybe_unused]] span<_OtherIndexType, extents_type::rank()> __strides)
  {
    // nvcc believes strides is unused here
    if constexpr (extents_type::rank() != 0)
    {
      index_type __size = 1;
      for (rank_type __r = 0; __r != extents_type::rank(); __r++)
      {
        // We can only check correct conversion of _OtherIndexType if it is an integral
        if (__conversion_may_overflow(__strides[__r]))
        {
          return false;
        }
        if (__ext.extent(__r) == index_type{0})
        {
          return true;
        }

        index_type __prod = (__ext.extent(__r) - 1);
        if (__mul_overflow(__prod, static_cast<index_type>(__strides[__r]), &__prod))
        {
          return false;
        }
        if (__add_overflow(__size, __prod, &__size))
        {
          return false;
        }
      }
    }

    return true;
  }

  // compute offset of a strided layout mapping
  template <class _StridedMapping, size_t... _Pos>
  [[nodiscard]] _CCCL_API static constexpr auto
  __offset(const _StridedMapping& __mapping, index_sequence<_Pos...>) noexcept
  {
    return static_cast<typename _StridedMapping::index_type>(__mapping((_Pos ? 0 : 0)...));
  }

  template <class _StridedMapping>
  [[nodiscard]] _CCCL_API static constexpr index_type __offset(const _StridedMapping& __mapping)
  {
    using _StridedExtents = typename _StridedMapping::extents_type;
    if constexpr (_StridedExtents::rank() != 0)
    {
      if (__mapping.required_span_size() == typename _StridedMapping::index_type{0})
      {
        return index_type{0};
      }
      return static_cast<index_type>(__offset(__mapping, __rank_sequence));
    }
    else
    {
      return static_cast<index_type>(__mapping());
    }
    _CCCL_UNREACHABLE();
  }

  static_assert((extents_type::rank_dynamic() > 0) || __required_span_size_is_representable(extents_type()),
                "layout_stride::mapping product of static extents must be representable as index_type.");

public:
  // [mdspan.layout.stride.cons], constructors
  _CCCL_API constexpr mapping() noexcept
      : __base(extents_type())
  {
    if constexpr (extents_type::rank() > 0)
    {
      index_type __stride = 1;
      for (rank_type __r = __rank_ - 1; __r > rank_type{0}; __r--)
      {
        __strides()[__r] = __stride;
        __stride *= extents().extent(__r);
      }
      __strides()[0] = __stride;
    }
  }

  _CCCL_HIDE_FROM_ABI constexpr mapping(const mapping&) noexcept = default;

  template <class _OtherIndexType, size_t... _Pos>
  [[nodiscard]] _CCCL_API static constexpr auto __to_strides_array(
    [[maybe_unused]] span<_OtherIndexType, extents_type::rank()> __strides, index_sequence<_Pos...>) noexcept
  {
    // nvcc believes strides is unused here
    return __stride_array{static_cast<index_type>(_CUDA_VSTD::as_const(__strides[_Pos]))...};
  }

  template <class _OtherIndexType, size_t... _Pos>
  [[nodiscard]] _CCCL_API static constexpr auto __check_strides(
    [[maybe_unused]] span<_OtherIndexType, extents_type::rank()> __strides, index_sequence<_Pos...>) noexcept
  {
    // nvcc believes strides is unused here
    if constexpr (_CCCL_TRAIT(is_integral, _OtherIndexType))
    {
      return ((__strides[_Pos] > _OtherIndexType{0}) && ... && true);
    }
    else
    {
      return ((static_cast<index_type>(__strides[_Pos]) > index_type{0}) && ... && true);
    }
    _CCCL_UNREACHABLE();
  }

  // compute the permutation for sorting the stride array
  // we never actually sort the stride array
  _CCCL_API constexpr void __bubble_sort_by_strides(array<rank_type, extents_type::rank()>& __permute) const noexcept
  {
    for (rank_type __i = __rank_ - 1; __i > 0; __i--)
    {
      for (rank_type __r = 0; __r < __i; __r++)
      {
        if (__strides()[__permute[__r]] > __strides()[__permute[__r + 1]])
        {
          swap(__permute[__r], __permute[__r + 1]);
        }
        else
        {
          // if two strides are the same then one of the associated extents must be 1 or 0
          // both could be, but you can't have one larger than 1 come first
          if ((__strides()[__permute[__r]] == __strides()[__permute[__r + 1]])
              && (extents().extent(__permute[__r]) > index_type{1}))
          {
            swap(__permute[__r], __permute[__r + 1]);
          }
        }
      }
    }
  }

  template <size_t... _Pos>
  [[nodiscard]] _CCCL_API constexpr bool __check_unique_mapping(index_sequence<_Pos...>) const noexcept
  {
    // basically sort the dimensions based on strides and extents, sorting is represented in permute array
    array<rank_type, extents_type::rank()> __permute{_Pos...};
    __bubble_sort_by_strides(__permute);

    // check that this permutations represents a growing set
    for (rank_type __i = 1; __i < __rank_; __i++)
    {
      if (static_cast<index_type>(__strides()[__permute[__i]])
          < static_cast<index_type>(__strides()[__permute[__i - 1]]) * extents().extent(__permute[__i - 1]))
      {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] _CCCL_API constexpr bool __check_unique_mapping(index_sequence<>) const noexcept
  {
    return true;
  }

  // nvcc cannot deduce this constructor when using _CCCL_REQUIRES
  template <class _OtherIndexType,
            enable_if_t<_CCCL_TRAIT(is_constructible, index_type, const _OtherIndexType&), int> = 0,
            enable_if_t<_CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type), int>   = 0>
  _CCCL_API constexpr mapping(const extents_type& __ext, span<_OtherIndexType, extents_type::rank()> __strides) noexcept
      : __base(__ext, __to_strides_array(__strides, __rank_sequence))
  {
    _CCCL_ASSERT(__check_strides(__strides, __rank_sequence),
                 "layout_stride::mapping ctor: all strides must be greater than 0");
    _CCCL_ASSERT(__required_span_size_is_representable(__ext, __strides),
                 "layout_stride::mapping ctor: required span size is not representable as index_type.");
    _CCCL_ASSERT(__check_unique_mapping(__rank_sequence),
                 "layout_stride::mapping ctor: the provided extents and strides lead to a non-unique mapping");
  }

  // nvcc cannot deduce this constructor when using _CCCL_REQUIRES
  template <class _OtherIndexType,
            enable_if_t<_CCCL_TRAIT(is_constructible, index_type, const _OtherIndexType&), int> = 0,
            enable_if_t<_CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type), int>   = 0>
  _CCCL_API constexpr mapping(const extents_type& __ext,
                              const array<_OtherIndexType, extents_type::rank()>& __strides) noexcept
      : mapping(__ext, span<const _OtherIndexType, extents_type::rank()>(__strides))
  {}

  template <class _StridedLayoutMapping, size_t... _Pos>
  [[nodiscard]] _CCCL_API static constexpr auto
  __to_strides_array(const _StridedLayoutMapping& __other, index_sequence<_Pos...>) noexcept
  {
    return __stride_array{static_cast<index_type>(__other.stride(_Pos))...};
  }

  // stride() only compiles for rank > 0
  template <class _StridedLayoutMapping, size_t... _Pos>
  [[nodiscard]] _CCCL_API static constexpr auto
  __check_mapped_strides(const _StridedLayoutMapping& __other, index_sequence<_Pos...>) noexcept
  {
    return ((static_cast<index_type>(__other.stride(_Pos)) > index_type{0}) && ... && true);
  }
  template <class _StridedLayoutMapping>
  [[nodiscard]] _CCCL_API static constexpr auto
  __check_mapped_strides(const _StridedLayoutMapping&, index_sequence<>) noexcept
  {
    return true;
  }

  _CCCL_TEMPLATE(class _StridedLayoutMapping)
  _CCCL_REQUIRES(__layout_stride_detail::__can_convert<_StridedLayoutMapping, _Extents> _CCCL_AND
                   __layout_stride_detail::__constraints::__converts_implicit<_StridedLayoutMapping, _Extents>)
  _CCCL_API constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __base(__other.extents(), __to_strides_array(__other, __rank_sequence))
  {
    _CCCL_ASSERT(__check_mapped_strides(__other, __rank_sequence),
                 "layout_stride::mapping converting ctor: all strides must be greater than 0");
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_stride::mapping converting ctor: other.required_span_size() must be representable as "
                 "index_type.");
    _CCCL_ASSERT(index_type{0} == __offset(__other),
                 "layout_stride::mapping converting ctor: base offset of mapping must be zero.");
  }
  _CCCL_TEMPLATE(class _StridedLayoutMapping)
  _CCCL_REQUIRES(__layout_stride_detail::__can_convert<_StridedLayoutMapping, _Extents> _CCCL_AND(
    !__layout_stride_detail::__constraints::__converts_implicit<_StridedLayoutMapping, _Extents>))
  _CCCL_API explicit constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __base(__other.extents(), __to_strides_array(__other, __rank_sequence))
  {
    _CCCL_ASSERT(__check_mapped_strides(__other, __rank_sequence),
                 "layout_stride::mapping converting ctor: all strides must be greater than 0");
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_stride::mapping converting ctor: other.required_span_size() must be representable as "
                 "index_type.");
    _CCCL_ASSERT(index_type{0} == __offset(__other),
                 "layout_stride::mapping converting ctor: base offset of mapping must be zero.");
  }

  _CCCL_HIDE_FROM_ABI constexpr mapping& operator=(const mapping&) noexcept = default;

  // [mdspan.layout.stride.obs], observers
  [[nodiscard]] _CCCL_API constexpr const extents_type& extents() const noexcept
  {
    return this->template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr __stride_array& __strides() noexcept
  {
    return this->template __get<1>();
  }

  [[nodiscard]] _CCCL_API constexpr const __stride_array& __strides() const noexcept
  {
    return this->template __get<1>();
  }

  template <size_t... _Pos>
  _CCCL_API constexpr array<index_type, extents_type::rank()> __to_strides(index_sequence<_Pos...>) const noexcept
  {
    return array<index_type, extents_type::rank()>{__strides()[_Pos]...};
  }

  [[nodiscard]] _CCCL_API constexpr array<index_type, extents_type::rank()> strides() const noexcept
  {
    return __to_strides(__rank_sequence);
  }

  template <size_t... _Pos>
  [[nodiscard]] _CCCL_API constexpr index_type __required_span_size(index_sequence<_Pos...>) const noexcept
  {
    const index_type __product = (index_type{1} * ... * extents().extent(_Pos));
    if (__product == index_type{0})
    {
      return index_type{0};
    }
    else
    {
      return (index_type{1} + ... + ((extents().extent(_Pos) - index_type{1}) * __strides()[_Pos]));
    }
  }

  [[nodiscard]] _CCCL_API constexpr index_type required_span_size() const noexcept
  {
    if constexpr (extents_type::rank() == 0)
    {
      return index_type{1};
    }
    else
    {
      return __required_span_size(__rank_sequence);
    }
  }

  template <size_t... _Pos, class... _Indices>
  [[nodiscard]] _CCCL_API static constexpr index_type
  __op_index(const __stride_array& __strides, index_sequence<_Pos...>, _Indices... __idx) noexcept
  {
    return (index_type{0} + ... + (static_cast<index_type>(__idx) * __strides[_Pos]));
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
    //_CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(__extents_, __idx...),
    //             "layout_stride::mapping: out of bounds indexing");
    return __op_index(__strides(), _CUDA_VSTD::make_index_sequence<sizeof...(_Indices)>(), __idx...);
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_always_unique() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_always_exhaustive() noexcept
  {
    return false;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_always_strided() noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_unique() noexcept
  {
    return true;
  }

  // The answer of this function is fairly complex in the case where one or more
  // extents are zero.
  // Technically it is meaningless to query is_exhaustive() in that case, but unfortunately
  // the way the standard defines this function, we can't give a simple true or false then.
  template <size_t... _Pos>
  [[nodiscard]] _CCCL_API constexpr index_type __to_total_size(index_sequence<_Pos...>) const noexcept
  {
    return (index_type{1} * ... * (extents().extent(_Pos)));
  }

  [[nodiscard]] _CCCL_API constexpr bool is_exhaustive() const noexcept
  {
    if constexpr (extents_type::rank() == 0)
    {
      return true;
    }
    else
    {
      const index_type __span_size = required_span_size();
      if (__span_size == index_type{0})
      {
        if constexpr (extents_type::rank() == 1)
        {
          return __strides()[0] == 1;
        }
        else
        {
          rank_type __r_largest = 0;
          for (rank_type __r = 1; __r < __rank_; __r++)
          {
            if (__strides()[__r] > __strides()[__r_largest])
            {
              __r_largest = __r;
            }
          }
          for (rank_type __r = 0; __r != __rank_; __r++)
          {
            if (extents().extent(__r) == 0 && __r != __r_largest)
            {
              return false;
            }
          }
          return true;
        }
      }
      else
      {
        const index_type __total_size = __to_total_size(__rank_sequence);
        return __span_size == __total_size;
      }
    }
    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_strided() noexcept
  {
    return true;
  }

  // according to the standard layout_stride does not have a constraint on stride(r) for rank>0
  // it still has the precondition though
  [[nodiscard]] _CCCL_API constexpr index_type stride(rank_type __r) const noexcept
  {
    _CCCL_ASSERT(__r < __rank_, "layout_stride::mapping::stride(): invalid rank index");
    return __strides()[__r];
  }

  template <class _OtherMapping, size_t... _Pos>
  [[nodiscard]] _CCCL_API static constexpr bool
  __op_eq(const mapping& __lhs, const _OtherMapping& __rhs, index_sequence<_Pos...>) noexcept
  {
    // avoid warning when comparing signed and unsigner integers and pick the wider of two types
    using _CommonType = common_type_t<index_type, typename _OtherMapping::index_type>;
    return ((static_cast<_CommonType>(__lhs.stride(_Pos)) == static_cast<_CommonType>(__rhs.stride(_Pos))) && ...
            && true);
  }

  template <class _OtherMapping>
  [[nodiscard]] _CCCL_API static constexpr bool __op_eq(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    if constexpr (extents_type::rank() > 0)
    {
      if (__offset(__rhs))
      {
        return false;
      }
      return __lhs.extents() == __rhs.extents() && __op_eq(__lhs, __rhs, __rank_sequence);
    }
    else
    {
      return (!__offset(__rhs));
    }
    _CCCL_UNREACHABLE();
  }

  template <class _OtherMapping, class _OtherExtents = typename _OtherMapping::extents_type>
  static constexpr bool __can_compare =
    __mdspan_detail::__layout_mapping_alike<_OtherMapping> && (_OtherExtents::rank() == _Extents::rank())
    && _OtherMapping::is_always_strided();

  template <class _OtherMapping>
  [[nodiscard]] _CCCL_API friend constexpr auto operator==(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__can_compare<_OtherMapping>)
  {
    return __op_eq(__lhs, __rhs);
  }

#if _CCCL_STD_VER <= 2017
  template <class _OtherMapping>
  [[nodiscard]] _CCCL_API friend constexpr auto operator==(const _OtherMapping& __lhs, const mapping& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)((!__mdspan_detail::__is_mapping_of<layout_stride, _OtherMapping>)
                                  && __can_compare<_OtherMapping>)
  {
    return __op_eq(__rhs, __lhs);
  }
  template <class _OtherMapping, class _Extents2 = _Extents>
  [[nodiscard]] _CCCL_API friend constexpr auto operator!=(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__can_compare<_OtherMapping>)
  {
    return !__op_eq(__lhs, __rhs);
  }
  template <class _OtherMapping, class _Extents2 = _Extents>
  [[nodiscard]] _CCCL_API friend constexpr auto operator!=(const _OtherMapping& __lhs, const mapping& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)((!__mdspan_detail::__is_mapping_of<layout_stride, _OtherMapping>)
                                  && __can_compare<_OtherMapping>)
  {
    return __op_eq(__rhs, __lhs);
  }
#endif // _CCCL_STD_VER <= 2017
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_H
