// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ITERATOR_TRAITS_H
#define _LIBCUDACXX___ITERATOR_ITERATOR_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__fwd/iterator_traits.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_primary_template.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/priority_tag.h>
#include <cuda/std/cstddef>

#if !_CCCL_COMPILER(NVRTC)
#  if _CCCL_COMPILER(MSVC)
#    include <xutility> // for ::std::input_iterator_tag
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#    include <iterator> // for ::std::input_iterator_tag
#  endif // !_CCCL_COMPILER(MSVC)

#  ifdef _GLIBCXX_DEBUG
#    include <debug/safe_iterator.h>
#  endif

#  if _CCCL_STD_VER >= 2020
#    include <cuda/std/__cccl/prologue.h>
template <class _Tp, class = void>
struct __cccl_type_is_defined : _CUDA_VSTD::false_type
{};

template <class _Tp>
struct __cccl_type_is_defined<_Tp, _CUDA_VSTD::void_t<decltype(sizeof(_Tp))>> : _CUDA_VSTD::true_type
{};

// detect whether the used STL has contiguous_iterator_tag defined
namespace std
{
struct __cccl_std_contiguous_iterator_tag_exists : __cccl_type_is_defined<struct contiguous_iterator_tag>
{};
} // namespace std

#    include <cuda/std/__cccl/epilogue.h>
#  endif // _CCCL_STD_VER >= 2020

#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Tp>
using __with_reference = _Tp&;

template <class _Tp>
concept __can_reference = requires { typename __with_reference<_Tp>; };

template <class _Tp>
concept __dereferenceable = requires(_Tp& __t) {
  { *__t } -> __can_reference; // not required to be equality-preserving
};

// [iterator.traits]
template <__dereferenceable _Tp>
using iter_reference_t = decltype(*declval<_Tp&>());

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ // vvv _CCCL_HAS_CONCEPTS() vvv

template <class _Tp>
using __with_reference = _Tp&;

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__can_reference_, requires()(typename(__with_reference<_Tp>)));

template <class _Tp>
_CCCL_CONCEPT __can_reference = _CCCL_FRAGMENT(__can_reference_, _Tp);

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wvoid-ptr-dereference")
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__dereferenceable_, requires(_Tp& __t)(requires(__can_reference<decltype(*__t)>)));
_CCCL_DIAG_POP

template <class _Tp>
_CCCL_CONCEPT __dereferenceable = _CCCL_FRAGMENT(__dereferenceable_, _Tp);

// [iterator.traits]
template <class _Tp>
using iter_reference_t = enable_if_t<__dereferenceable<_Tp>, decltype(*declval<_Tp&>())>;

#endif // _CCCL_HAS_CONCEPTS()

#if _CCCL_COMPILER(NVRTC)

struct _CCCL_TYPE_VISIBILITY_DEFAULT input_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT output_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT forward_iterator_tag : public input_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT bidirectional_iterator_tag : public forward_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT random_access_iterator_tag : public bidirectional_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT contiguous_iterator_tag : public random_access_iterator_tag
{};

#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ / vvv !_CCCL_COMPILER(NVRTC) vvv

using input_iterator_tag         = ::std::input_iterator_tag;
using output_iterator_tag        = ::std::output_iterator_tag;
using forward_iterator_tag       = ::std::forward_iterator_tag;
using bidirectional_iterator_tag = ::std::bidirectional_iterator_tag;
using random_access_iterator_tag = ::std::random_access_iterator_tag;

#  if _CCCL_STD_VER >= 2020
struct _CCCL_TYPE_VISIBILITY_DEFAULT __contiguous_iterator_tag_backfill : public ::std::random_access_iterator_tag
{};
using contiguous_iterator_tag =
  _If<::std::__cccl_std_contiguous_iterator_tag_exists::value,
      ::std::contiguous_iterator_tag,
      __contiguous_iterator_tag_backfill>;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
struct _CCCL_TYPE_VISIBILITY_DEFAULT contiguous_iterator_tag : public random_access_iterator_tag
{};
#  endif // _CCCL_STD_VER <= 2017

#endif // !_CCCL_COMPILER(NVRTC)

template <class _Iter>
struct __iter_traits_cache
{
  using type = __select_traits<remove_cvref_t<_Iter>, remove_cvref_t<_Iter>>;
};
template <class _Iter>
using _ITER_TRAITS = typename __iter_traits_cache<_Iter>::type;

#if defined(_GLIBCXX_DEBUG)
_CCCL_TEMPLATE(class _Iter, class _Ty, class _Range)
_CCCL_REQUIRES(_IsSame<_Iter, ::__gnu_debug::_Safe_iterator<_Ty*, _Range>>::value)
_CCCL_API inline auto __iter_concept_fn(::__gnu_debug::_Safe_iterator<_Ty*, _Range>, __priority_tag<3>)
  -> contiguous_iterator_tag;
#endif // _GLIBCXX_DEBUG
#if defined(__GLIBCXX__)
_CCCL_TEMPLATE(class _Iter, class _Ty, class _Range)
_CCCL_REQUIRES(_IsSame<_Iter, ::__gnu_cxx::__normal_iterator<_Ty*, _Range>>::value)
_CCCL_API inline auto __iter_concept_fn(::__gnu_cxx::__normal_iterator<_Ty*, _Range>, __priority_tag<3>)
  -> contiguous_iterator_tag;
#endif // __GLIBCXX__
#if defined(_LIBCPP_VERSION)
_CCCL_TEMPLATE(class _Iter, class _Ty)
_CCCL_REQUIRES(_IsSame<_Iter, ::std::__wrap_iter<_Ty*>>::value)
_CCCL_API inline auto __iter_concept_fn(::std::__wrap_iter<_Ty*>, __priority_tag<3>) -> contiguous_iterator_tag;
#elif defined(_MSVC_STL_VERSION) || defined(_IS_WRS)
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Array_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Array_const_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Vector_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Vector_const_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_String_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_String_const_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_String_view_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Span_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
#endif // _MSVC_STL_VERSION
_CCCL_TEMPLATE(class _Iter, class _Ty)
_CCCL_REQUIRES(_IsSame<_Iter, _Ty*>::value)
_CCCL_API inline auto __iter_concept_fn(_Ty*, __priority_tag<3>) -> contiguous_iterator_tag;

template <class _Iter>
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<2>) -> typename _ITER_TRAITS<_Iter>::iterator_concept;
template <class _Iter>
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<1>) -> typename _ITER_TRAITS<_Iter>::iterator_category;
template <class _Iter>
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<0>)
  -> enable_if_t<__is_primary_cccl_template<_Iter>::value && __is_primary_std_template<_Iter>::value,
                 random_access_iterator_tag>;

template <class _Iter>
using __iter_concept_t = decltype(_CUDA_VSTD::__iter_concept_fn<_Iter>(declval<_Iter>(), __priority_tag<3>{}));

template <class _Iter, class = void>
struct __iter_concept_cache
{};

template <class _Iter>
struct __iter_concept_cache<_Iter, void_t<__iter_concept_t<_Iter>>>
{
  using type = __iter_concept_t<_Iter>;
};

template <class _Iter>
using _ITER_CONCEPT = typename __iter_concept_cache<_Iter>::type;

template <class _Tp>
struct __has_iterator_category
{
private:
  template <class _Up>
  _CCCL_API inline static false_type __test(...);
  template <class _Up>
  _CCCL_API inline static true_type __test(typename _Up::iterator_category* = nullptr);

public:
  static const bool value = decltype(__test<_Tp>(nullptr))::value;
};

template <class _Tp>
struct __has_iterator_concept
{
private:
  template <class _Up>
  _CCCL_API inline static false_type __test(...);
  template <class _Up>
  _CCCL_API inline static true_type __test(typename _Up::iterator_concept* = nullptr);

public:
  static const bool value = decltype(__test<_Tp>(nullptr))::value;
};

#if _CCCL_HAS_CONCEPTS()

// The `cpp17-*-iterator` exposition-only concepts have very similar names to the `Cpp17*Iterator` named requirements
// from `[iterator.cpp17]`. To avoid confusion between the two, the exposition-only concepts have been banished to
// a "detail" namespace indicating they have a niche use-case.
namespace __iterator_traits_detail
{
template <class _Ip>
concept __cpp17_iterator = requires(_Ip __i) {
  { *__i } -> __can_reference;
  { ++__i } -> same_as<_Ip&>;
  { *__i++ } -> __can_reference;
} && copyable<_Ip>;

template <class _Ip>
concept __cpp17_input_iterator = __cpp17_iterator<_Ip> && equality_comparable<_Ip> && requires(_Ip __i) {
  typename incrementable_traits<_Ip>::difference_type;
  typename indirectly_readable_traits<_Ip>::value_type;
  typename common_reference_t<iter_reference_t<_Ip>&&, typename indirectly_readable_traits<_Ip>::value_type&>;
  typename common_reference_t<decltype(*__i++)&&, typename indirectly_readable_traits<_Ip>::value_type&>;
  requires signed_integral<typename incrementable_traits<_Ip>::difference_type>;
};

template <class _Ip>
concept __cpp17_forward_iterator =
  __cpp17_input_iterator<_Ip> && constructible_from<_Ip> && is_lvalue_reference_v<iter_reference_t<_Ip>>
  && same_as<remove_cvref_t<iter_reference_t<_Ip>>, typename indirectly_readable_traits<_Ip>::value_type>
  && requires(_Ip __i) {
       { __i++ } -> convertible_to<_Ip const&>;
       { *__i++ } -> same_as<iter_reference_t<_Ip>>;
     };

template <class _Ip>
concept __cpp17_bidirectional_iterator = __cpp17_forward_iterator<_Ip> && requires(_Ip __i) {
  { --__i } -> same_as<_Ip&>;
  { __i-- } -> convertible_to<_Ip const&>;
  { *__i-- } -> same_as<iter_reference_t<_Ip>>;
};

template <class _Ip>
concept __cpp17_random_access_iterator =
  __cpp17_bidirectional_iterator<_Ip> && totally_ordered<_Ip>
  && requires(_Ip __i, typename incrementable_traits<_Ip>::difference_type __n) {
       { __i += __n } -> same_as<_Ip&>;
       { __i -= __n } -> same_as<_Ip&>;
       { __i + __n } -> same_as<_Ip>;
       { __n + __i } -> same_as<_Ip>;
       { __i - __n } -> same_as<_Ip>;
       { __i - __i } -> same_as<decltype(__n)>; // NOLINT(misc-redundant-expression) ; This is llvm.org/PR54114
       { __i[__n] } -> convertible_to<iter_reference_t<_Ip>>;
     };
} // namespace __iterator_traits_detail

// We need to consider if a user has specialized std::iterator_traits
template <class _Ip>
concept __specialized_from_std = !__is_primary_std_template<remove_cvref_t<_Ip>>::value;

template <class _Ip>
concept __has_member_reference = requires { typename _Ip::reference; };

template <class _Ip>
concept __has_member_pointer = requires { typename _Ip::pointer; };

template <class _Ip>
concept __has_member_iterator_category = requires { typename _Ip::iterator_category; };

template <class _Ip>
concept __specifies_members = !__specialized_from_std<_Ip> && requires {
  typename _Ip::value_type;
  typename _Ip::difference_type;
  requires __has_member_reference<_Ip>;
  requires __has_member_iterator_category<_Ip>;
};

template <class>
struct __iterator_traits_member_pointer_or_void
{
  using type = void;
};

template <__has_member_pointer _Tp>
struct __iterator_traits_member_pointer_or_void<_Tp>
{
  using type = typename _Tp::pointer;
};

template <class _Tp>
concept __cpp17_iterator_missing_members =
  !__specialized_from_std<_Tp> && !__specifies_members<_Tp> && __iterator_traits_detail::__cpp17_iterator<_Tp>;

template <class _Tp>
concept __cpp17_input_iterator_missing_members =
  __cpp17_iterator_missing_members<_Tp> && __iterator_traits_detail::__cpp17_input_iterator<_Tp>;

// Otherwise, `pointer` names `void`.
template <class>
struct __iterator_traits_member_pointer_or_arrow_or_void
{
  using type = void;
};

// [iterator.traits]/3.2.1
// If the qualified-id `I::pointer` is valid and denotes a type, `pointer` names that type.
template <__has_member_pointer _Ip>
struct __iterator_traits_member_pointer_or_arrow_or_void<_Ip>
{
  using type = typename _Ip::pointer;
};

// Otherwise, if `decltype(declval<I&>().operator->())` is well-formed, then `pointer` names that
// type.
template <class _Ip>
  requires requires(_Ip& __i) { __i.operator->(); } && (!__has_member_pointer<_Ip>)
struct __iterator_traits_member_pointer_or_arrow_or_void<_Ip>
{
  using type = decltype(declval<_Ip&>().operator->());
};

// Otherwise, `reference` names `iter-reference-t<I>`.
template <class _Ip>
struct __iterator_traits_member_reference
{
  using type = iter_reference_t<_Ip>;
};

// [iterator.traits]/3.2.2
// If the qualified-id `I::reference` is valid and denotes a type, `reference` names that type.
template <__has_member_reference _Ip>
struct __iterator_traits_member_reference<_Ip>
{
  using type = typename _Ip::reference;
};

// [iterator.traits]/3.2.3.4
// input_iterator_tag
template <class _Ip>
struct __deduce_iterator_category
{
  using type = input_iterator_tag;
};

// [iterator.traits]/3.2.3.1
// `random_access_iterator_tag` if `I` satisfies `cpp17-random-access-iterator`, or otherwise
template <__iterator_traits_detail::__cpp17_random_access_iterator _Ip>
struct __deduce_iterator_category<_Ip>
{
  using type = random_access_iterator_tag;
};

// [iterator.traits]/3.2.3.2
// `bidirectional_iterator_tag` if `I` satisfies `cpp17-bidirectional-iterator`, or otherwise
template <__iterator_traits_detail::__cpp17_bidirectional_iterator _Ip>
  requires(!__iterator_traits_detail::__cpp17_random_access_iterator<_Ip>) // nvbug 3885350
struct __deduce_iterator_category<_Ip>
{
  using type = bidirectional_iterator_tag;
};

// [iterator.traits]/3.2.3.3
// `forward_iterator_tag` if `I` satisfies `cpp17-forward-iterator`, or otherwise
template <__iterator_traits_detail::__cpp17_forward_iterator _Ip>
  requires(!__iterator_traits_detail::__cpp17_bidirectional_iterator<_Ip>) // nvbug 3885350
struct __deduce_iterator_category<_Ip>
{
  using type = forward_iterator_tag;
};

template <class _Ip>
struct __iterator_traits_iterator_category : __deduce_iterator_category<_Ip>
{};

// [iterator.traits]/3.2.3
// If the qualified-id `I::iterator-category` is valid and denotes a type, `iterator-category` names
// that type.
template <__has_member_iterator_category _Ip>
struct __iterator_traits_iterator_category<_Ip>
{
  using type = typename _Ip::iterator_category;
};

// otherwise, it names void.
template <class>
struct __iterator_traits_difference_type
{
  using type = void;
};

// If the qualified-id `incrementable_traits<I>::difference_type` is valid and denotes a type, then
// `difference_type` names that type;
template <class _Ip>
  requires requires { typename incrementable_traits<_Ip>::difference_type; }
struct __iterator_traits_difference_type<_Ip>
{
  using type = typename incrementable_traits<_Ip>::difference_type;
};

// [iterator.traits]/3.4
// Otherwise, `iterator_traits<I>` has no members by any of the above names.
template <class>
struct __iterator_traits
{};

#  if !_CCCL_COMPILER(NVRTC)
// We need to properly accept specializations of `std::iterator_traits`
template <__specialized_from_std _Ip>
struct __iterator_traits<_Ip> : public ::std::iterator_traits<_Ip>
{};
#  endif // !_CCCL_COMPILER(NVRTC)

// [iterator.traits]/3.1
// If `I` has valid ([temp.deduct]) member types `difference-type`, `value-type`, `reference`, and
// `iterator-category`, then `iterator-traits<I>` has the following publicly accessible members:
template <__specifies_members _Ip>
struct __iterator_traits<_Ip>
{
  using iterator_category = typename _Ip::iterator_category;
  using value_type        = typename _Ip::value_type;
  using difference_type   = typename _Ip::difference_type;
  using pointer           = typename __iterator_traits_member_pointer_or_void<_Ip>::type;
  using reference         = typename _Ip::reference;
};

// [iterator.traits]/3.2
// Otherwise, if `I` satisfies the exposition-only concept `cpp17-input-iterator`,
// `iterator-traits<I>` has the following publicly accessible members:
template <__cpp17_input_iterator_missing_members _Ip>
struct __iterator_traits<_Ip>
{
  using iterator_category = typename __iterator_traits_iterator_category<_Ip>::type;
  using value_type        = typename indirectly_readable_traits<_Ip>::value_type;
  using difference_type   = typename incrementable_traits<_Ip>::difference_type;
  using pointer           = typename __iterator_traits_member_pointer_or_arrow_or_void<_Ip>::type;
  using reference         = typename __iterator_traits_member_reference<_Ip>::type;
};

// Otherwise, if `I` satisfies the exposition-only concept `cpp17-iterator`, then
// `iterator_traits<I>` has the following publicly accessible members:
template <__cpp17_iterator_missing_members _Ip>
  requires(!__cpp17_input_iterator_missing_members<_Ip>) // nvbug 3885350
struct __iterator_traits<_Ip>
{
  using iterator_category = output_iterator_tag;
  using value_type        = void;
  using difference_type   = typename __iterator_traits_difference_type<_Ip>::type;
  using pointer           = void;
  using reference         = void;
};

template <class _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT iterator_traits : __iterator_traits<_Ip>
{
  using __cccl_primary_template = iterator_traits;
};

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

// The `cpp17-*-iterator` exposition-only concepts have very similar names to the `Cpp17*Iterator` named requirements
// from `[iterator.cpp17]`. To avoid confusion between the two, the exposition-only concepts have been banished to
// a "detail" namespace indicating they have a niche use-case.
namespace __iterator_traits_detail
{
template <class _Ip>
_CCCL_CONCEPT_FRAGMENT(
  __cpp17_iterator_,
  requires(_Ip __i)(requires(__can_reference<decltype(*__i)>),
                    requires(same_as<_Ip&, decltype(++__i)>),
                    requires(__can_reference<decltype(*__i++)>),
                    requires(copyable<_Ip>)));

template <class _Ip>
_CCCL_CONCEPT __cpp17_iterator = _CCCL_FRAGMENT(__cpp17_iterator_, _Ip);

template <class _Ip>
_CCCL_CONCEPT_FRAGMENT(
  __cpp17_input_iterator_,
  requires(_Ip __i)(
    typename(common_reference_t<iter_reference_t<_Ip>&&, typename indirectly_readable_traits<_Ip>::value_type&>),
    typename(common_reference_t<decltype(*__i++)&&, typename indirectly_readable_traits<_Ip>::value_type&>),
    requires(__cpp17_iterator<_Ip>),
    requires(equality_comparable<_Ip>),
    requires(__has_member_difference_type<incrementable_traits<_Ip>>),
    requires(__has_member_value_type<indirectly_readable_traits<_Ip>>),
    requires(signed_integral<typename incrementable_traits<_Ip>::difference_type>)));

template <class _Ip>
_CCCL_CONCEPT __cpp17_input_iterator = _CCCL_FRAGMENT(__cpp17_input_iterator_, _Ip);

template <class _Ip>
_CCCL_CONCEPT_FRAGMENT(
  __cpp17_forward_iterator_,
  requires(_Ip __i)(
    requires(__cpp17_input_iterator<_Ip>),
    requires(convertible_to<decltype(__i++), _Ip const&>),
    requires(same_as<iter_reference_t<_Ip>, decltype(*__i++)>),
    requires(constructible_from<_Ip>),
    requires(_CCCL_TRAIT(is_lvalue_reference, iter_reference_t<_Ip>)),
    requires(same_as<remove_cvref_t<iter_reference_t<_Ip>>, typename indirectly_readable_traits<_Ip>::value_type>)));

template <class _Ip>
_CCCL_CONCEPT __cpp17_forward_iterator = _CCCL_FRAGMENT(__cpp17_forward_iterator_, _Ip);

template <class _Ip>
_CCCL_CONCEPT_FRAGMENT(
  __cpp17_bidirectional_iterator_,
  requires(_Ip __i)(requires(__cpp17_forward_iterator<_Ip>),
                    requires(same_as<_Ip&, decltype(--__i)>),
                    requires(convertible_to<decltype(__i--), _Ip const&>),
                    requires(same_as<iter_reference_t<_Ip>, decltype(*__i--)>)));

template <class _Ip>
_CCCL_CONCEPT __cpp17_bidirectional_iterator = _CCCL_FRAGMENT(__cpp17_bidirectional_iterator_, _Ip);

template <class _Ip>
_CCCL_CONCEPT_FRAGMENT(
  __cpp17_random_access_iterator_,
  requires(_Ip __i, typename incrementable_traits<_Ip>::difference_type __n)(
    requires(same_as<_Ip&, decltype(__i += __n)>),
    requires(same_as<_Ip&, decltype(__i -= __n)>),
    requires(same_as<_Ip, decltype(__i + __n)>),
    requires(same_as<_Ip, decltype(__n + __i)>),
    requires(same_as<_Ip, decltype(__i - __n)>),
    requires(same_as<decltype(__n), decltype(__i - __i)>),
    requires(convertible_to<decltype(__i[__n]), iter_reference_t<_Ip>>)));

template <class _Ip>
_CCCL_CONCEPT __cpp17_random_access_iterator =
  __cpp17_bidirectional_iterator<_Ip> && totally_ordered<_Ip> && _CCCL_FRAGMENT(__cpp17_random_access_iterator_, _Ip);
} // namespace __iterator_traits_detail

// We need to consider if a user has specialized std::iterator_traits
template <class _Ip>
inline constexpr bool __specialized_from_std = !__is_primary_std_template<remove_cvref_t<_Ip>>::value;

template <class, class = void>
inline constexpr bool __has_member_reference = false;

template <class _Tp>
inline constexpr bool __has_member_reference<_Tp, void_t<typename _Tp::reference>> = true;

template <class, class = void>
inline constexpr bool __has_member_pointer = false;

template <class _Tp>
inline constexpr bool __has_member_pointer<_Tp, void_t<typename _Tp::pointer>> = true;

template <class, class = void>
inline constexpr bool __has_member_iterator_category = false;

template <class _Tp>
inline constexpr bool __has_member_iterator_category<_Tp, void_t<typename _Tp::iterator_category>> = true;

template <class _Ip>
_CCCL_CONCEPT __specifies_members =
  !__specialized_from_std<_Ip> && __has_member_value_type<_Ip> && __has_member_difference_type<_Ip>
  && __has_member_reference<_Ip> && __has_member_iterator_category<_Ip>;

template <class, class = void>
struct __iterator_traits_member_pointer_or_void
{
  using type = void;
};

template <class _Tp>
struct __iterator_traits_member_pointer_or_void<_Tp, enable_if_t<__has_member_pointer<_Tp>>>
{
  using type = typename _Tp::pointer;
};

template <class _Tp>
_CCCL_CONCEPT __cpp17_iterator_missing_members =
  !__specialized_from_std<_Tp> && !__specifies_members<_Tp> && __iterator_traits_detail::__cpp17_iterator<_Tp>;

template <class _Tp>
_CCCL_CONCEPT __cpp17_input_iterator_missing_members =
  __cpp17_iterator_missing_members<_Tp> && __iterator_traits_detail::__cpp17_input_iterator<_Tp>;

// Otherwise, `pointer` names `void`.
template <class, class = void>
struct __iterator_traits_member_pointer_or_arrow_or_void
{
  using type = void;
};

// [iterator.traits]/3.2.1
// If the qualified-id `I::pointer` is valid and denotes a type, `pointer` names that type.
template <class _Ip>
struct __iterator_traits_member_pointer_or_arrow_or_void<_Ip, enable_if_t<__has_member_pointer<_Ip>>>
{
  using type = typename _Ip::pointer;
};

template <class _Ip>
_CCCL_CONCEPT_FRAGMENT(__has_operator_arrow_, requires(_Ip& __i)(__cccl_unused(__i.operator->())));

template <class _Ip>
_CCCL_CONCEPT __has_operator_arrow = _CCCL_FRAGMENT(__has_operator_arrow_, _Ip);

// Otherwise, if `decltype(declval<I&>().operator->())` is well-formed, then `pointer` names that
// type.
template <class _Ip>
struct __iterator_traits_member_pointer_or_arrow_or_void<
  _Ip,
  enable_if_t<__has_operator_arrow<_Ip> && !__has_member_pointer<_Ip>>>
{
  using type = decltype(declval<_Ip&>().operator->());
};

// Otherwise, `reference` names `iter-reference-t<I>`.
template <class _Ip, class = void>
struct __iterator_traits_member_reference
{
  using type = iter_reference_t<_Ip>;
};

// [iterator.traits]/3.2.2
// If the qualified-id `I::reference` is valid and denotes a type, `reference` names that type.
template <class _Ip>
struct __iterator_traits_member_reference<_Ip, enable_if_t<__has_member_reference<_Ip>>>
{
  using type = typename _Ip::reference;
};

// [iterator.traits]/3.2.3.4
// input_iterator_tag
template <class _Ip, class = void>
struct __deduce_iterator_category
{
  using type = input_iterator_tag;
};

// [iterator.traits]/3.2.3.1
// `random_access_iterator_tag` if `I` satisfies `cpp17-random-access-iterator`, or otherwise
template <class _Ip>
struct __deduce_iterator_category<_Ip, enable_if_t<__iterator_traits_detail::__cpp17_random_access_iterator<_Ip>>>
{
  using type = random_access_iterator_tag;
};

// [iterator.traits]/3.2.3.2
// `bidirectional_iterator_tag` if `I` satisfies `cpp17-bidirectional-iterator`, or otherwise
template <class _Ip>
struct __deduce_iterator_category<_Ip,
                                  enable_if_t<!__iterator_traits_detail::__cpp17_random_access_iterator<_Ip>
                                              && __iterator_traits_detail::__cpp17_bidirectional_iterator<_Ip>>>
{
  using type = bidirectional_iterator_tag;
};

// [iterator.traits]/3.2.3.3
// `forward_iterator_tag` if `I` satisfies `cpp17-forward-iterator`, or otherwise
template <class _Ip>
struct __deduce_iterator_category<_Ip,
                                  enable_if_t<!__iterator_traits_detail::__cpp17_bidirectional_iterator<_Ip>
                                              && __iterator_traits_detail::__cpp17_forward_iterator<_Ip>>>
{
  using type = forward_iterator_tag;
};

template <class _Ip, class = void>
struct __iterator_traits_iterator_category : __deduce_iterator_category<_Ip>
{};

// [iterator.traits]/3.2.3
// If the qualified-id `I::iterator-category` is valid and denotes a type, `iterator-category` names
// that type.
template <class _Ip>
struct __iterator_traits_iterator_category<_Ip, enable_if_t<__has_member_iterator_category<_Ip>>>
{
  using type = typename _Ip::iterator_category;
};

// otherwise, it names void.
template <class, class = void>
struct __iterator_traits_difference_type
{
  using type = void;
};

// If the qualified-id `incrementable_traits<I>::difference_type` is valid and denotes a type, then
// `difference_type` names that type;
template <class _Ip>
struct __iterator_traits_difference_type<_Ip, void_t<typename incrementable_traits<_Ip>::difference_type>>
{
  using type = typename incrementable_traits<_Ip>::difference_type;
};

// [iterator.traits]/3.4
// Otherwise, `iterator_traits<I>` has no members by any of the above names.
template <class, class = void>
struct __iterator_traits
{};

#  if !_CCCL_COMPILER(NVRTC)
template <class _Ip>
struct __iterator_traits<_Ip, enable_if_t<__specialized_from_std<_Ip>>> : public ::std::iterator_traits<_Ip>
{};
#  endif // !_CCCL_COMPILER(NVRTC)

// [iterator.traits]/3.1
// If `I` has valid ([temp.deduct]) member types `difference-type`, `value-type`, `reference`, and
// `iterator-category`, then `iterator-traits<I>` has the following publicly accessible members:
template <class _Ip>
struct __iterator_traits<_Ip, enable_if_t<__specifies_members<_Ip>>>
{
  using iterator_category = typename _Ip::iterator_category;
  using value_type        = typename _Ip::value_type;
  using difference_type   = typename _Ip::difference_type;
  using pointer           = typename __iterator_traits_member_pointer_or_void<_Ip>::type;
  using reference         = typename _Ip::reference;
};

// [iterator.traits]/3.2
// Otherwise, if `I` satisfies the exposition-only concept `cpp17-input-iterator`,
// `iterator-traits<I>` has the following publicly accessible members:
template <class _Ip>
struct __iterator_traits<_Ip, enable_if_t<!__specifies_members<_Ip> && __cpp17_input_iterator_missing_members<_Ip>>>
{
  using iterator_category = typename __iterator_traits_iterator_category<_Ip>::type;
  using value_type        = typename indirectly_readable_traits<_Ip>::value_type;
  using difference_type   = typename incrementable_traits<_Ip>::difference_type;
  using pointer           = typename __iterator_traits_member_pointer_or_arrow_or_void<_Ip>::type;
  using reference         = typename __iterator_traits_member_reference<_Ip>::type;
};

// Otherwise, if `I` satisfies the exposition-only concept `cpp17-iterator`, then
// `iterator_traits<I>` has the following publicly accessible members:
template <class _Ip>
struct __iterator_traits<_Ip,
                         enable_if_t<!__specifies_members<_Ip> && !__cpp17_input_iterator_missing_members<_Ip>
                                     && __cpp17_iterator_missing_members<_Ip>>>
{
  using iterator_category = output_iterator_tag;
  using value_type        = void;
  using difference_type   = typename __iterator_traits_difference_type<_Ip>::type;
  using pointer           = void;
  using reference         = void;
};

template <class _Ip, class>
struct _CCCL_TYPE_VISIBILITY_DEFAULT iterator_traits : __iterator_traits<_Ip>
{
  using __cccl_primary_template = iterator_traits;
};

#endif // _CCCL_HAS_CONCEPTS()

template <class _Tp>
#if _CCCL_HAS_CONCEPTS()
  requires is_object_v<_Tp>
#endif // _CCCL_HAS_CONCEPTS()
struct _CCCL_TYPE_VISIBILITY_DEFAULT iterator_traits<_Tp*>
{
  using difference_type   = ptrdiff_t;
  using value_type        = remove_cv_t<_Tp>;
  using pointer           = _Tp*;
  using reference         = typename add_lvalue_reference<_Tp>::type;
  using iterator_category = random_access_iterator_tag;
  using iterator_concept  = contiguous_iterator_tag;
};

template <class _Iter, class _Tag>
_CCCL_CONCEPT __has_iterator_category_convertible_to = _CCCL_REQUIRES_EXPR((_Iter, _Tag)) //
  (typename(typename iterator_traits<_Iter>::iterator_category),
   requires(is_convertible_v<typename iterator_traits<_Iter>::iterator_category, _Tag>));

template <class _Iter, class _Tag>
_CCCL_CONCEPT __has_iterator_concept_convertible_to = _CCCL_REQUIRES_EXPR((_Iter, _Tag)) //
  (typename(typename _Iter::iterator_concept), requires(is_convertible_v<typename _Iter::iterator_concept, _Tag>));

template <class _Iter>
inline constexpr bool __has_input_traversal =
  __has_iterator_category_convertible_to<_Iter, input_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, input_iterator_tag>;

template <class _Iter>
inline constexpr bool __has_forward_traversal =
  __has_iterator_category_convertible_to<_Iter, forward_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, forward_iterator_tag>;

template <class _Iter>
inline constexpr bool __has_bidirectional_traversal =
  __has_iterator_category_convertible_to<_Iter, bidirectional_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, bidirectional_iterator_tag>;

template <class _Iter>
inline constexpr bool __has_random_access_traversal =
  __has_iterator_category_convertible_to<_Iter, random_access_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, random_access_iterator_tag>;

// __has_contiguous_traversal determines if an iterator is known by
// libc++ to be contiguous, either because it advertises itself as such
// (in C++20) or because it is a pointer type or a known trivial wrapper
// around a (possibly fancy) pointer type, such as __wrap_iter<T*>.
// Such iterators receive special "contiguous" optimizations in
// std::copy and std::sort.
//
template <class _Iter>
inline constexpr bool __has_contiguous_traversal =
  __has_iterator_category_convertible_to<_Iter, contiguous_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, contiguous_iterator_tag>;

// Any native pointer which is an iterator is also a contiguous iterator.
template <class _Tp>
inline constexpr bool __has_contiguous_traversal<_Tp*> = true;

template <class _InputIterator>
using __iter_value_type = typename iterator_traits<_InputIterator>::value_type;

template <class _InputIterator>
using __iter_key_type = typename remove_const<typename iterator_traits<_InputIterator>::value_type::first_type>::type;

template <class _InputIterator>
using __iter_mapped_type = typename iterator_traits<_InputIterator>::value_type::second_type;

template <class _InputIterator>
using __iter_to_alloc_type =
  pair<typename add_const<typename iterator_traits<_InputIterator>::value_type::first_type>::type,
       typename iterator_traits<_InputIterator>::value_type::second_type>;

template <class _Iter>
using __iterator_category_type = typename iterator_traits<_Iter>::iterator_category;

template <class _Iter>
using __iterator_pointer_type = typename iterator_traits<_Iter>::pointer;

template <class _Iter>
using __iter_diff_t = typename iterator_traits<_Iter>::difference_type;

template <class _InputIterator>
using __iter_value_type = typename iterator_traits<_InputIterator>::value_type;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_ITERATOR_TRAITS_H
