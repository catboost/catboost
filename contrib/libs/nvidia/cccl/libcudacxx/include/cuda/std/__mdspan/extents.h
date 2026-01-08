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

#ifndef _LIBCUDACXX___MDSPAN_EXTENTS_HPP
#define _LIBCUDACXX___MDSPAN_EXTENTS_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/unreachable.h>
#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __mdspan_detail
{

// ------------------------------------------------------------------
// ------------ __static_array --------------------------------------
// ------------------------------------------------------------------
// array like class which provides an array of static values with get
template <class _Tp, _Tp... _Values>
struct __static_array
{
  [[nodiscard]] _CCCL_API static constexpr size_t __size() noexcept
  {
    return sizeof...(_Values);
  }

  [[nodiscard]] _CCCL_API static constexpr _Tp __get(size_t __index) noexcept
  {
    constexpr array<_Tp, sizeof...(_Values)> __array = {_Values...};
    return __array[__index];
  }

  template <size_t _Index>
  [[nodiscard]] _CCCL_API static constexpr _Tp __get()
  {
    return __get(_Index);
  }
};

// ------------------------------------------------------------------
// ------------ __possibly_empty_array  -----------------------------
// ------------------------------------------------------------------

// array like class which provides get function and operator [], and
// has a specialization for the size 0 case.
// This is needed to make the __maybe_static_array be truly empty, for
// all static values.

template <class _Tp, size_t _Size>
struct __possibly_empty_array
{
  _Tp __vals_[_Size];
  [[nodiscard]] _CCCL_API constexpr _Tp& operator[](size_t __index)
  {
    return __vals_[__index];
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](size_t __index) const
  {
    return __vals_[__index];
  }
};

template <class _Tp>
struct __possibly_empty_array<_Tp, 0>
{
#if _CCCL_COMPILER(MSVC)
  _CCCL_API constexpr _Tp& operator[](size_t __index)
  {
    return *__get(__index);
  }
  _CCCL_API constexpr const _Tp& operator[](size_t __index) const
  {
    return *__get(__index);
  }

  _CCCL_API constexpr _Tp* __get(size_t)
  {
    return nullptr;
  }
  _CCCL_API constexpr const _Tp* __get(size_t) const
  {
    return nullptr;
  }
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  _CCCL_API constexpr _Tp& operator[](size_t)
  {
    _CCCL_UNREACHABLE();
  }
  _CCCL_API constexpr const _Tp& operator[](size_t) const
  {
    _CCCL_UNREACHABLE();
  }
#endif // !_CCCL_COMPILER(MSVC)
};

// ------------------------------------------------------------------
// ------------ static_partial_sums ---------------------------------
// ------------------------------------------------------------------

// Provides a compile time partial sum one can index into

template <size_t... _Values>
struct __static_partial_sums
{
  [[nodiscard]] _CCCL_API static constexpr array<size_t, sizeof...(_Values)> __static_partial_sums_impl()
  {
    array<size_t, sizeof...(_Values)> __values{_Values...};
    array<size_t, sizeof...(_Values)> __partial_sums{{}};
    size_t __running_sum = 0;
    for (int __i = 0; __i != sizeof...(_Values); ++__i)
    {
      __partial_sums[__i] = __running_sum;
      __running_sum += __values[__i];
    }
    return __partial_sums;
  }

  [[nodiscard]] _CCCL_API static constexpr size_t __get(size_t __index)
  {
    constexpr array<size_t, sizeof...(_Values)> __result = __static_partial_sums_impl();
    return __result[__index];
  }
};

// ------------------------------------------------------------------
// ------------ __maybe_static_array --------------------------------
// ------------------------------------------------------------------

template <class _TStatic, _TStatic _DynTag, _TStatic... _Values>
constexpr size_t __count_dynamic_v = (size_t{0} + ... + static_cast<size_t>(_Values == _DynTag));

// array like class which has a mix of static and runtime values but
// only stores the runtime values.
// The type of the static and the runtime values can be different.
// The position of a dynamic value is indicated through a tag value.
// We manually implement EBCO because MSVC and some odler compiler fail hard with [[no_unique_address]]
template <class _TDynamic, class _TStatic, _TStatic _DynTag, _TStatic... _Values>
struct _CCCL_DECLSPEC_EMPTY_BASES
__maybe_static_array : private __possibly_empty_array<_TDynamic, __count_dynamic_v<_TStatic, _DynTag, _Values...>>
{
  static_assert(_CCCL_TRAIT(is_convertible, _TStatic, _TDynamic),
                "__maybe_static_array: _TStatic must be convertible to _TDynamic");
  static_assert(_CCCL_TRAIT(is_convertible, _TDynamic, _TStatic),
                "__maybe_static_array: _TDynamic must be convertible to _TStatic");

private:
  // Static values member
  static constexpr size_t __size_         = sizeof...(_Values);
  static constexpr size_t __size_dynamic_ = __count_dynamic_v<_TStatic, _DynTag, _Values...>;
  using _StaticValues                     = __static_array<_TStatic, _Values...>;
  using _DynamicValues                    = __possibly_empty_array<_TDynamic, __size_dynamic_>;

  // static mapping of indices to the position in the dynamic values array
  using _DynamicIdxMap = __static_partial_sums<static_cast<size_t>(_Values == _DynTag)...>;

  template <size_t... Indices>
  [[nodiscard]] _CCCL_API static constexpr _DynamicValues __zeros(index_sequence<Indices...>) noexcept
  {
    return _DynamicValues{((void) Indices, 0)...};
  }

public:
  _CCCL_API constexpr __maybe_static_array() noexcept
      : _DynamicValues{__zeros(make_index_sequence<__size_dynamic_>())}
  {}

  template <class _Tp, size_t _Size>
  _CCCL_API constexpr __maybe_static_array(span<_Tp, _Size> __vals) noexcept
      : _DynamicValues{}
  {
    if constexpr (_Size == __size_dynamic_)
    {
      for (size_t __i = 0; __i != _Size; __i++)
      {
        (*static_cast<_DynamicValues*>(this))[__i] = static_cast<_TDynamic>(__vals[__i]);
      }
    }
    else
    {
      for (size_t __i = 0; __i != __size_; __i++)
      {
        _TStatic __static_val = _StaticValues::__get(__i);
        if (__static_val == _DynTag)
        {
          (*static_cast<_DynamicValues*>(this))[_DynamicIdxMap::__get(__i)] = static_cast<_TDynamic>(__vals[__i]);
        }
        else
        {
          // Not catching this could lead to out of bounds errors later
          // e.g. using my_mdspan_t = mdspan<int, extents<int, 10>>; my_mdspan_t = m(new int[N], span<int,1>(&N));
          // Right-hand-side construction looks ok with allocation and size matching,
          // but since (potentially elsewhere defined) my_mdspan_t has static size m now thinks its range is 10 not N
          _CCCL_ASSERT(static_cast<_TDynamic>(__vals[__i]) == static_cast<_TDynamic>(__static_val),
                       "extents construction: mismatch of provided arguments with static extents.");
        }
      }
    }
  }

  // constructors from dynamic values only -- this covers the case for rank() == 0
  _CCCL_TEMPLATE(class... _DynVals)
  _CCCL_REQUIRES((sizeof...(_DynVals) == __size_dynamic_) && (!__all<__is_std_span_v<_DynVals>...>::value))
  _CCCL_API constexpr __maybe_static_array(_DynVals... __vals) noexcept
      : _DynamicValues{static_cast<_TDynamic>(__vals)...}
  {}

  // constructors from all values -- here rank will be greater than 0
  _CCCL_TEMPLATE(class... _DynVals)
  _CCCL_REQUIRES((sizeof...(_DynVals) != __size_dynamic_) && (!__all<__is_std_span_v<_DynVals>...>::value))
  _CCCL_API constexpr __maybe_static_array(_DynVals... __vals)
      : _DynamicValues{}
  {
    static_assert((sizeof...(_DynVals) == __size_), "Invalid number of values.");
    _TDynamic __values[__size_] = {static_cast<_TDynamic>(__vals)...};
    for (size_t __i = 0; __i < __size_; __i++)
    {
      _TStatic __static_val = _StaticValues::__get(__i);
      if (__static_val == _DynTag)
      {
        (*static_cast<_DynamicValues*>(this))[_DynamicIdxMap::__get(__i)] = __values[__i];
      }
      else
      {
        // Not catching this could lead to out of bounds errors later
        // e.g. using my_mdspan_t = mdspan<int, extents<int, 10>>; my_mdspan_t = m(new int[5], 5);
        // Right-hand-side construction looks ok with allocation and size matching,
        // but since (potentially elsewhere defined) my_mdspan_t has static size m now thinks its range is 10 not 5
        _CCCL_ASSERT(__values[__i] == static_cast<_TDynamic>(__static_val),
                     "extents construction: mismatch of provided arguments with static extents.");
      }
    }
  }

  // access functions
  [[nodiscard]] _CCCL_API static constexpr _TStatic __static_value(size_t __i) noexcept
  {
    if constexpr (__size_ > 0)
    {
      _CCCL_ASSERT(__i < __size_, "extents access: index must be less than rank");
    }
    return _StaticValues::__get(__i);
  }

  [[nodiscard]] _CCCL_API constexpr _TDynamic __value(size_t __i) const
  {
    if constexpr (__size_ > 0)
    {
      _CCCL_ASSERT(__i < __size_, "extents access: index must be less than rank");
    }
    _TStatic __static_val = _StaticValues::__get(__i);
    return __static_val == _DynTag
           ? (*static_cast<const _DynamicValues*>(this))[_DynamicIdxMap::__get(__i)]
           : static_cast<_TDynamic>(__static_val);
  }

  [[nodiscard]] _CCCL_API constexpr _TDynamic operator[](size_t __i) const
  {
    if constexpr (__size_ > 0)
    {
      _CCCL_ASSERT(__i < __size_, "extents access: index must be less than rank");
    }
    return __value(__i);
  }

  // observers
  [[nodiscard]] _CCCL_API static constexpr size_t __size()
  {
    return __size_;
  }
  [[nodiscard]] _CCCL_API static constexpr size_t __size_dynamic()
  {
    return __size_dynamic_;
  }
};

template <class _To, class _From>
static constexpr bool __potentially_narrowing =
  static_cast<make_unsigned_t<_To>>((numeric_limits<_To>::max)())
  < static_cast<make_unsigned_t<_From>>((numeric_limits<_From>::max)());

// Function to check whether a value is representable as another type
// value must be a positive integer otherwise returns false
// if _From is not an integral, we just check positivity
_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES(integral<_To>)
[[nodiscard]] _CCCL_API constexpr bool __is_representable_as([[maybe_unused]] _From __value)
{
  if constexpr (integral<_From>)
  {
    if constexpr (_CCCL_TRAIT(is_signed, _From))
    {
      if constexpr (__potentially_narrowing<_To, _From>)
      {
        using _To_u   = make_unsigned_t<_To>;
        using _From_u = make_unsigned_t<_From>;
        if (__value < 0)
        {
          return false;
        }
        return static_cast<_To_u>((numeric_limits<_To>::max)()) >= static_cast<_From_u>(__value);
      }
      else // !__potentially_narrowing<_To, _From>
      {
        return __value >= 0;
      }
    }
    else // !_CCCL_TRAIT(is_signed, _From)
    {
      if constexpr (__potentially_narrowing<_To, _From>)
      {
        using _To_u   = make_unsigned_t<_To>;
        using _From_u = make_unsigned_t<_From>;
        return static_cast<_To_u>((numeric_limits<_To>::max)()) >= static_cast<_From_u>(__value);
      }
      else // !__potentially_narrowing<_To, _From>
      {
        return true;
      }
    }
  }
  else // !integral<_From>
  {
    if constexpr (_CCCL_TRAIT(is_signed, _To))
    {
      return static_cast<_To>(__value) >= 0;
    }
    else // !_CCCL_TRAIT(is_signed, _To)
    {
      return true;
    }
  }
  _CCCL_UNREACHABLE();
}

_CCCL_TEMPLATE(class _To, class... _From)
_CCCL_REQUIRES(integral<_To>)
[[nodiscard]] _CCCL_API constexpr bool __are_representable_as(_From... __values)
{
  return (__mdspan_detail::__is_representable_as<_To>(__values) && ... && true);
}

_CCCL_TEMPLATE(class _To, class _From, size_t _Size)
_CCCL_REQUIRES(integral<_To>)
[[nodiscard]] _CCCL_API constexpr bool __are_representable_as(span<_From, _Size> __values)
{
  for (size_t __i = 0; __i != _Size; __i++)
  {
    if (!__mdspan_detail::__is_representable_as<_To>(__values[__i]))
    {
      return false;
    }
  }
  return true;
}

} // namespace __mdspan_detail

// ------------------------------------------------------------------
// ------------ extents ---------------------------------------------
// ------------------------------------------------------------------

// Class to delegate between the different (non-)explicit constructors
struct __extent_delegate_tag
{};

// Class to describe the extents of a multi dimensional array.
// Used by mdspan, mdarray and layout mappings.
// See ISO C++ standard [mdspan.extents]
template <class _IndexType, size_t... _Extents>
class extents : private __mdspan_detail::__maybe_static_array<_IndexType, size_t, dynamic_extent, _Extents...>
{
public:
  // typedefs for integral types used
  using index_type = _IndexType;
  using size_type  = make_unsigned_t<index_type>;
  using rank_type  = size_t;

  static_assert(_CCCL_TRAIT(is_integral, index_type) && !_CCCL_TRAIT(is_same, index_type, bool),
                "extents::index_type must be a signed or unsigned integer type");
  static_assert(
    __all<(__mdspan_detail::__is_representable_as<index_type>(_Extents) || (_Extents == dynamic_extent))...>::value,
    "extents ctor: arguments must be representable as index_type and nonnegative");

private:
  static constexpr rank_type __rank_ = sizeof...(_Extents);
  static constexpr rank_type __rank_dynamic_ =
    (rank_type(0) + ... + (static_cast<rank_type>(_Extents == dynamic_extent)));

  // internal storage type using __maybe_static_array
  using _Values = __mdspan_detail::__maybe_static_array<_IndexType, size_t, dynamic_extent, _Extents...>;

public:
  // [mdspan.extents.obs], observers of multidimensional index space
  [[nodiscard]] _CCCL_API static constexpr rank_type rank() noexcept
  {
    return __rank_;
  }
  [[nodiscard]] _CCCL_API static constexpr rank_type rank_dynamic() noexcept
  {
    return __rank_dynamic_;
  }

  [[nodiscard]] _CCCL_API constexpr index_type extent(rank_type __r) const noexcept
  {
    return this->__value(__r);
  }
  [[nodiscard]] _CCCL_API static constexpr size_t static_extent(rank_type __r) noexcept
  {
    return _Values::__static_value(__r);
  }

  // [mdspan.extents.cons], constructors
  _CCCL_HIDE_FROM_ABI constexpr extents() noexcept = default;

  // Construction from just dynamic or all values.
  // Precondition check is deferred to __maybe_static_array constructor
  _CCCL_TEMPLATE(class... _OtherIndexTypes)
  _CCCL_REQUIRES((sizeof...(_OtherIndexTypes) == __rank_ || sizeof...(_OtherIndexTypes) == __rank_dynamic_)
                   _CCCL_AND __mdspan_detail::__all_convertible_to_index_type<index_type, _OtherIndexTypes...>)
  _CCCL_API constexpr explicit extents(_OtherIndexTypes... __dynvals) noexcept
      : _Values(static_cast<index_type>(__dynvals)...)
  {
    // Not catching this could lead to out of bounds errors later
    // e.g. mdspan m(ptr, dextents<char, 1>(200u)); leads to an extent of -56 on m
    _CCCL_ASSERT(__mdspan_detail::__are_representable_as<index_type>(__dynvals...),
                 "extents ctor: arguments must be representable as index_type and nonnegative");
  }

  template <class _OtherIndexType>
  static constexpr bool __is_convertible_to_index_type =
    _CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type)
    && _CCCL_TRAIT(is_nothrow_constructible, index_type, const _OtherIndexType&);

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size == __rank_dynamic_) _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API constexpr extents(const array<_OtherIndexType, _Size>& __exts) noexcept
      : extents(span<const _OtherIndexType, _Size>(__exts))
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size == __rank_) _CCCL_AND(_Size != __rank_dynamic_)
                   _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API explicit constexpr extents(const array<_OtherIndexType, _Size>& __exts) noexcept
      : extents(span<const _OtherIndexType, _Size>(__exts))
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size == __rank_dynamic_) _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API constexpr extents(span<_OtherIndexType, _Size> __exts) noexcept
      : _Values(__exts)
  {
    // Not catching this could lead to out of bounds errors later
    // e.g. array a{200u}; mdspan<int, dextents<char,1>> m(ptr, extents(span<unsigned,1>(a))); leads to an extent of -56
    // on m
    _CCCL_ASSERT(__mdspan_detail::__are_representable_as<index_type>(__exts),
                 "extents ctor: arguments must be representable as index_type and nonnegative");
  }

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size != __rank_dynamic_) _CCCL_AND(_Size == __rank_)
                   _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API explicit constexpr extents(span<_OtherIndexType, _Size> __exts) noexcept
      : _Values(__exts)
  {
    // Not catching this could lead to out of bounds errors later
    // e.g. array a{200u}; mdspan<int, dextents<char,1>> m(ptr, extents(span<unsigned,1>(a))); leads to an extent of -56
    // on m
    _CCCL_ASSERT(__mdspan_detail::__are_representable_as<index_type>(__exts),
                 "extents ctor: arguments must be representable as index_type and nonnegative");
  }

private:
  // Function to construct extents storage from other extents.
  template <size_t _DynCount, size_t _Idx, class _OtherExtents, class... _DynamicValues>
  [[nodiscard]] _CCCL_API constexpr _Values __construct_vals_from_extents(
    integral_constant<size_t, _DynCount>,
    integral_constant<size_t, _Idx>,
    [[maybe_unused]] const _OtherExtents& __exts,
    _DynamicValues... __dynamic_values) noexcept
  {
    if constexpr (_Idx == __rank_)
    {
      if constexpr (_DynCount == __rank_dynamic_)
      {
        return _Values{static_cast<index_type>(__dynamic_values)...};
      }
      else
      {
        static_assert(_DynCount == __rank_dynamic_, "Constructor of invalid extents passed to extent::extent");
      }
    }
    else // _Idx < __rank_
    {
      if constexpr (static_extent(_Idx) == dynamic_extent)
      {
        return __construct_vals_from_extents(
          integral_constant<size_t, _DynCount + 1>(),
          integral_constant<size_t, _Idx + 1>(),
          __exts,
          __dynamic_values...,
          __exts.extent(_Idx));
      }
      else // static_extent(_Idx) != dynamic_extent
      {
        return __construct_vals_from_extents(
          integral_constant<size_t, _DynCount>(), integral_constant<size_t, _Idx + 1>(), __exts, __dynamic_values...);
      }
    }
    _CCCL_UNREACHABLE();
  }

  template <class _OtherIndexType, size_t... _OtherExtents>
  _CCCL_API constexpr extents(__extent_delegate_tag, const extents<_OtherIndexType, _OtherExtents...>& __other) noexcept
      : _Values(__construct_vals_from_extents(integral_constant<size_t, 0>(), integral_constant<size_t, 0>(), __other))
  {
    if constexpr (rank() != 0)
    {
      for (size_t __r = 0; __r != rank(); __r++)
      {
        if constexpr (__mdspan_detail::__potentially_narrowing<index_type, _OtherIndexType>)
        {
          // Not catching this could lead to out of bounds errors later
          // e.g. dextents<char,1>> e(dextents<unsigned,1>(200)) leads to an extent of -56 on e
          _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.extent(__r)),
                       "extents ctor: arguments must be representable as index_type and nonnegative");
        }

        // Not catching this could lead to out of bounds errors later
        // e.g. mdspan<int, extents<int, 10>> m = mdspan<int, dextents<int, 1>>(new int[5], 5);
        // Right-hand-side construction was ok, but m now thinks its range is 10 not 5
        _CCCL_ASSERT(
          (_Values::__static_value(__r) == dynamic_extent)
            || (static_cast<index_type>(__other.extent(__r)) == static_cast<index_type>(_Values::__static_value(__r))),
          "extents construction: mismatch of provided arguments with static extents.");
      }
    }
  }

public:
  // Converting constructor from other extents specializations
  template <class _OtherIndexType, size_t... _OtherExtents>
  static constexpr bool __is_explicit_conversion =
    (((_Extents != dynamic_extent) && (_OtherExtents == dynamic_extent)) || ...)
    || __mdspan_detail::__potentially_narrowing<index_type, _OtherIndexType>;

  template <size_t... _OtherExtents>
  static constexpr bool __is_matching_extents =
    ((_OtherExtents == dynamic_extent || _Extents == dynamic_extent || _OtherExtents == _Extents) && ... && true);

  _CCCL_TEMPLATE(class _OtherIndexType, size_t... _OtherExtents)
  _CCCL_REQUIRES((sizeof...(_OtherExtents) == sizeof...(_Extents)) _CCCL_AND __is_matching_extents<_OtherExtents...>
                   _CCCL_AND(!__is_explicit_conversion<_OtherIndexType, _OtherExtents...>))
  _CCCL_API constexpr extents(const extents<_OtherIndexType, _OtherExtents...>& __other) noexcept
      : extents(__extent_delegate_tag{}, __other)
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, size_t... _OtherExtents)
  _CCCL_REQUIRES((sizeof...(_OtherExtents) == sizeof...(_Extents))
                   _CCCL_AND __is_matching_extents<_OtherExtents...> _CCCL_AND
                     __is_explicit_conversion<_OtherIndexType, _OtherExtents...>)
  _CCCL_API explicit constexpr extents(const extents<_OtherIndexType, _OtherExtents...>& __other) noexcept
      : extents(__extent_delegate_tag{}, __other)
  {}

  // Comparison operator
  template <class _OtherIndexType, size_t... _OtherExtents>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const extents& __lhs, const extents<_OtherIndexType, _OtherExtents...>& __rhs) noexcept
  {
    if constexpr (rank() != sizeof...(_OtherExtents))
    {
      return false;
    }
    else if constexpr (rank() != 0)
    {
      for (rank_type __r = 0; __r != __rank_; __r++)
      {
        // avoid warning when comparing signed and unsigner integers and pick the wider of two types
        using _CommonType = common_type_t<index_type, _OtherIndexType>;
        if (static_cast<_CommonType>(__lhs.extent(__r)) != static_cast<_CommonType>(__rhs.extent(__r)))
        {
          return false;
        }
      }
      return true;
    }
    else // MSVC needs this or it complains about unreachable code in the first condition
    {
      return true;
    }
    _CCCL_UNREACHABLE();
  }

#if _CCCL_STD_VER <= 2017
  template <class _OtherIndexType, size_t... _OtherExtents>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const extents& __lhs, const extents<_OtherIndexType, _OtherExtents...>& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_STD_VER <= 2017
};

// Recursive helper classes to implement dextents alias for extents
namespace __mdspan_detail
{

template <class _IndexType, size_t _Rank, class _Extents = extents<_IndexType>>
struct __make_dextents;

template <class _IndexType, size_t _Rank, class _Extents = extents<_IndexType>>
using __make_dextents_t = typename __make_dextents<_IndexType, _Rank, _Extents>::type;

template <class _IndexType, size_t _Rank, size_t... _ExtentsPack>
struct __make_dextents<_IndexType, _Rank, extents<_IndexType, _ExtentsPack...>>
{
  using type = __make_dextents_t<_IndexType, _Rank - 1, extents<_IndexType, dynamic_extent, _ExtentsPack...>>;
};

template <class _IndexType, size_t... _ExtentsPack>
struct __make_dextents<_IndexType, 0, extents<_IndexType, _ExtentsPack...>>
{
  using type = extents<_IndexType, _ExtentsPack...>;
};

} // end namespace __mdspan_detail

// [mdspan.extents.dextents], alias template
template <class _IndexType, size_t _Rank>
using dextents = __mdspan_detail::__make_dextents_t<_IndexType, _Rank>;

template <size_t _Rank, class _IndexType = size_t>
using dims = dextents<_IndexType, _Rank>;

// nvcc cannot handle type conversions without this workaround
struct __to_dynamic_extent
{
  template <class>
  static constexpr size_t value = dynamic_extent;
};

// Deduction guide for extents
template <class... _IndexTypes>
_CCCL_HOST_DEVICE extents(_IndexTypes...) -> extents<size_t, __to_dynamic_extent::template value<_IndexTypes>...>;

namespace __mdspan_detail
{

template <class _IndexType, size_t... _ExtentsPack>
struct __is_extents<extents<_IndexType, _ExtentsPack...>> : true_type
{};

// Function to check whether a set of indices are a multidimensional
// index into extents. This is a word of power in the C++ standard
// requiring that the indices are larger than 0 and smaller than
// the respective extents.

_CCCL_TEMPLATE(class _IndexType, class _From)
_CCCL_REQUIRES(integral<_IndexType>)
[[nodiscard]] _CCCL_API constexpr bool __is_index_in_extent(_IndexType __extent, _From __value)
{
  if constexpr (integral<_From>)
  {
    if constexpr (_CCCL_TRAIT(is_signed, _From))
    {
      if (__value < 0)
      {
        return false;
      }
      using _Tp = common_type_t<_IndexType, _From>;
      return static_cast<_Tp>(__value) < static_cast<_Tp>(__extent);
    }
    else
    {
      using _Tp = common_type_t<_IndexType, _From>;
      return static_cast<_Tp>(__value) < static_cast<_Tp>(__extent);
    }
  }
  else
  {
    if constexpr (_CCCL_TRAIT(is_signed, _From))
    {
      if (static_cast<_IndexType>(__value) < 0)
      {
        return false;
      }
      return static_cast<_IndexType>(__value) < __extent;
    }
    else
    {
      return static_cast<_IndexType>(__value) < __extent;
    }
  }
  _CCCL_UNREACHABLE();
}

template <size_t... _Idxs, class _Extents, class... _From>
[[nodiscard]] _CCCL_API constexpr bool
__is_multidimensional_index_in_impl(index_sequence<_Idxs...>, const _Extents& __ext, _From... __values)
{
  return (__mdspan_detail::__is_index_in_extent(__ext.extent(_Idxs), __values) && ... && true);
}

template <class _Extents, class... _From>
[[nodiscard]] _CCCL_API constexpr bool __is_multidimensional_index_in(const _Extents& __ext, _From... __values)
{
  return __mdspan_detail::__is_multidimensional_index_in_impl(
    make_index_sequence<_Extents::rank()>(), __ext, __values...);
}

} // namespace __mdspan_detail

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_EXTENTS_H
