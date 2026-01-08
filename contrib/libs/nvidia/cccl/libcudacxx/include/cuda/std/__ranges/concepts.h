// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_CONCEPTS_H
#define _LIBCUDACXX___RANGES_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/enable_view.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/add_pointer.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()

// [range.range]

template <class _Tp>
concept range = requires(_Tp& __t) {
  _CUDA_VRANGES::begin(__t); // sometimes equality-preserving
  _CUDA_VRANGES::end(__t);
};

template <class _Tp>
concept input_range = range<_Tp> && input_iterator<iterator_t<_Tp>>;

template <class _Range>
concept borrowed_range =
  range<_Range> && (is_lvalue_reference_v<_Range> || enable_borrowed_range<remove_cvref_t<_Range>>);

// `iterator_t` defined in <__ranges/access.h>

template <range _Rp>
using sentinel_t = decltype(_CUDA_VRANGES::end(_CUDA_VSTD::declval<_Rp&>()));

template <range _Rp>
using range_difference_t = iter_difference_t<iterator_t<_Rp>>;

template <range _Rp>
using range_value_t = iter_value_t<iterator_t<_Rp>>;

template <range _Rp>
using range_reference_t = iter_reference_t<iterator_t<_Rp>>;

template <range _Rp>
using range_rvalue_reference_t = iter_rvalue_reference_t<iterator_t<_Rp>>;

template <range _Rp>
using range_common_reference_t = iter_common_reference_t<iterator_t<_Rp>>;

// [range.sized]
template <class _Tp>
concept sized_range = range<_Tp> && requires(_Tp& __t) { _CUDA_VRANGES::size(__t); };

template <sized_range _Rp>
using range_size_t = decltype(_CUDA_VRANGES::size(_CUDA_VSTD::declval<_Rp&>()));

// `disable_sized_range` defined in `<__ranges/size.h>`

// [range.view], views

// `enable_view` defined in <__ranges/enable_view.h>
// `view_base` defined in <__ranges/enable_view.h>

template <class _Tp>
concept view = range<_Tp> && movable<_Tp> && enable_view<_Tp>;

template <class _Range>
concept __simple_view = view<_Range> && range<const _Range> && same_as<iterator_t<_Range>, iterator_t<const _Range>>
                     && same_as<sentinel_t<_Range>, sentinel_t<const _Range>>;

// [range.refinements], other range refinements
template <class _Rp, class _Tp>
concept output_range = range<_Rp> && output_iterator<iterator_t<_Rp>, _Tp>;

template <class _Tp>
concept forward_range = input_range<_Tp> && forward_iterator<iterator_t<_Tp>>;

template <class _Tp>
concept bidirectional_range = forward_range<_Tp> && bidirectional_iterator<iterator_t<_Tp>>;

template <class _Tp>
concept random_access_range = bidirectional_range<_Tp> && random_access_iterator<iterator_t<_Tp>>;

template <class _Tp>
concept contiguous_range = random_access_range<_Tp> && contiguous_iterator<iterator_t<_Tp>> && requires(_Tp& __t) {
  { _CUDA_VRANGES::data(__t) } -> same_as<add_pointer_t<range_reference_t<_Tp>>>;
};

template <class _Tp>
concept common_range = range<_Tp> && same_as<iterator_t<_Tp>, sentinel_t<_Tp>>;

template <class _Tp>
inline constexpr bool __is_std_initializer_list = false;

template <class _Ep>
inline constexpr bool __is_std_initializer_list<initializer_list<_Ep>> = true;

template <class _Tp>
concept viewable_range =
  range<_Tp>
  && ((view<remove_cvref_t<_Tp>> && constructible_from<remove_cvref_t<_Tp>, _Tp>)
      || (!view<remove_cvref_t<_Tp>>
          && (is_lvalue_reference_v<_Tp>
              || (movable<remove_reference_t<_Tp>> && !__is_std_initializer_list<remove_cvref_t<_Tp>>) )));

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
// [range.range]

// clang-format off
template <class _Tp>
_CCCL_CONCEPT range =
  _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)
  (
    void(_CUDA_VRANGES::begin(__t)),
    void(_CUDA_VRANGES::end(__t))
  );

template <class _Tp>
_CCCL_CONCEPT input_range =
  _CCCL_REQUIRES_EXPR((_Tp))
  (
    requires(range<_Tp>),
    requires(input_iterator<iterator_t<_Tp>>)
  );
// clang-format on

template <class _Range>
_CCCL_CONCEPT_FRAGMENT(
  __borrowed_range_,
  requires()(requires(range<_Range>),
             requires((is_lvalue_reference_v<_Range> || enable_borrowed_range<remove_cvref_t<_Range>>) )));

template <class _Range>
_CCCL_CONCEPT borrowed_range = _CCCL_FRAGMENT(__borrowed_range_, _Range);

// `iterator_t` defined in <__ranges/access.h>

template <class _Rp>
using sentinel_t = enable_if_t<range<_Rp>, decltype(_CUDA_VRANGES::end(_CUDA_VSTD::declval<_Rp&>()))>;

template <class _Rp>
using range_difference_t = enable_if_t<range<_Rp>, iter_difference_t<iterator_t<_Rp>>>;

template <class _Rp>
using range_value_t = enable_if_t<range<_Rp>, iter_value_t<iterator_t<_Rp>>>;

template <class _Rp>
using range_reference_t = enable_if_t<range<_Rp>, iter_reference_t<iterator_t<_Rp>>>;

template <class _Rp>
using range_rvalue_reference_t = enable_if_t<range<_Rp>, iter_rvalue_reference_t<iterator_t<_Rp>>>;

template <class _Rp>
using range_common_reference_t = enable_if_t<range<_Rp>, iter_common_reference_t<iterator_t<_Rp>>>;

// [range.sized]
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__sized_range_,
                       requires(_Tp& __t)(requires(range<_Tp>), typename(decltype(_CUDA_VRANGES::size(__t)))));

template <class _Tp>
_CCCL_CONCEPT sized_range = _CCCL_FRAGMENT(__sized_range_, _Tp);

template <class _Rp>
using range_size_t = enable_if_t<sized_range<_Rp>, decltype(_CUDA_VRANGES::size(_CUDA_VSTD::declval<_Rp&>()))>;

// `disable_sized_range` defined in `<__ranges/size.h>`

// [range.view], views

// `enable_view` defined in <__ranges/enable_view.h>
// `view_base` defined in <__ranges/enable_view.h>

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__view_, requires()(requires(range<_Tp>), requires(movable<_Tp>), requires(enable_view<_Tp>)));

template <class _Tp>
_CCCL_CONCEPT view = _CCCL_FRAGMENT(__view_, _Tp);

template <class _Range>
_CCCL_CONCEPT_FRAGMENT(
  __simple_view_,
  requires()(requires(view<_Range>),
             requires(range<const _Range>),
             requires(same_as<iterator_t<_Range>, iterator_t<const _Range>>),
             requires(same_as<sentinel_t<_Range>, sentinel_t<const _Range>>)));

template <class _Range>
_CCCL_CONCEPT __simple_view = _CCCL_FRAGMENT(__simple_view_, _Range);

// [range.refinements], other range refinements
template <class _Rp, class _Tp>
_CCCL_CONCEPT_FRAGMENT(__output_range_,
                       requires()(requires(range<_Rp>), requires(output_iterator<iterator_t<_Rp>, _Tp>)));

template <class _Rp, class _Tp>
_CCCL_CONCEPT output_range = _CCCL_FRAGMENT(__output_range_, _Rp, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__forward_range_,
                       requires()(requires(input_range<_Tp>), requires(forward_iterator<iterator_t<_Tp>>)));

template <class _Tp>
_CCCL_CONCEPT forward_range = _CCCL_FRAGMENT(__forward_range_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__bidirectional_range_,
                       requires()(requires(forward_range<_Tp>), requires(bidirectional_iterator<iterator_t<_Tp>>)));

template <class _Tp>
_CCCL_CONCEPT bidirectional_range = _CCCL_FRAGMENT(__bidirectional_range_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __random_access_range_,
  requires()(requires(bidirectional_range<_Tp>), requires(random_access_iterator<iterator_t<_Tp>>)));

template <class _Tp>
_CCCL_CONCEPT random_access_range = _CCCL_FRAGMENT(__random_access_range_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __contiguous_range_,
  requires(_Tp& __t)(requires(random_access_range<_Tp>),
                     requires(contiguous_iterator<iterator_t<_Tp>>),
                     requires(same_as<decltype(_CUDA_VRANGES::data(__t)), add_pointer_t<range_reference_t<_Tp>>>)));

template <class _Tp>
_CCCL_CONCEPT contiguous_range = _CCCL_FRAGMENT(__contiguous_range_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__common_range_,
                       requires()(requires(range<_Tp>), requires(same_as<iterator_t<_Tp>, sentinel_t<_Tp>>)));

template <class _Tp>
_CCCL_CONCEPT common_range = _CCCL_FRAGMENT(__common_range_, _Tp);

template <class _Tp>
inline constexpr bool __is_std_initializer_list = false;

template <class _Ep>
inline constexpr bool __is_std_initializer_list<initializer_list<_Ep>> = true;

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __viewable_range_,
  requires()(
    requires(range<_Tp>),
    requires(((view<remove_cvref_t<_Tp>> && constructible_from<remove_cvref_t<_Tp>, _Tp>)
              || (!view<remove_cvref_t<_Tp>>
                  && (is_lvalue_reference_v<_Tp>
                      || (movable<remove_reference_t<_Tp>> && !__is_std_initializer_list<remove_cvref_t<_Tp>>) ))))));

template <class _Tp>
_CCCL_CONCEPT viewable_range = _CCCL_FRAGMENT(__viewable_range_, _Tp);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

//[container.intro.reqmts]
#if _CCCL_HAS_CONCEPTS()
template <class _Range, class _Tp>
concept __container_compatible_range = input_range<_Range> && convertible_to<range_reference_t<_Range>, _Tp>;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Range, class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __container_compatible_range_,
  requires()(requires(input_range<_Range>), requires(convertible_to<range_reference_t<_Range>, _Tp>)));

template <class _Range, class _Tp>
_CCCL_CONCEPT __container_compatible_range = _CCCL_FRAGMENT(__container_compatible_range_, _Range, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_CONCEPTS_H
