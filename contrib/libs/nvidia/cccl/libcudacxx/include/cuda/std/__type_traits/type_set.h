//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_TYPE_SET_H
#define _LIBCUDACXX___TYPE_TRAITS_TYPE_SET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class...>
struct __type_list;

template <class _Set, class... _Ty>
struct __type_set_contains : __fold_and<_CCCL_TRAIT(is_base_of, type_identity<_Ty>, _Set)...>
{};

template <class _Set, class... _Ty>
inline constexpr bool __type_set_contains_v = __fold_and_v<is_base_of_v<type_identity<_Ty>, _Set>...>;

namespace __set
{
template <class... _Ts>
struct __tupl;

template <>
struct __tupl<>
{
  template <class _Ty>
  using __maybe_insert _CCCL_NODEBUG_ALIAS = __tupl<_Ty>;

  _CCCL_API static constexpr size_t __size() noexcept
  {
    return 0;
  }
};

template <class _Ty, class... _Ts>
struct __tupl<_Ty, _Ts...>
    : type_identity<_Ty>
    , __tupl<_Ts...>
{
  template <class _Uy>
  using __maybe_insert _CCCL_NODEBUG_ALIAS =
    _If<_CCCL_TRAIT(__type_set_contains, __tupl, _Uy), __tupl, __tupl<_Uy, _Ty, _Ts...>>;

  _CCCL_API static constexpr size_t __size() noexcept
  {
    return sizeof...(_Ts) + 1;
  }
};

template <bool _Empty>
struct __bulk_insert
{
  template <class _Set, class...>
  using __call _CCCL_NODEBUG_ALIAS = _Set;
};

template <>
struct __bulk_insert<false>
{
#if _CCCL_COMPILER(MSVC, <, 19, 20)
  template <class _Set, class _Ty, class... _Us>
  _CCCL_API inline static auto __insert_fn(__type_list<_Ty, _Us...>*) ->
    typename __bulk_insert<sizeof...(_Us) == 0>::template __call<typename _Set::template __maybe_insert<_Ty>, _Us...>;

  template <class _Set, class... _Us>
  using __call _CCCL_NODEBUG_ALIAS = decltype(__insert_fn<_Set>(static_cast<__type_list<_Us...>*>(nullptr)));
#else
  template <class _Set, class _Ty, class... _Us>
  using __call _CCCL_NODEBUG_ALIAS =
    typename __bulk_insert<sizeof...(_Us) == 0>::template __call<typename _Set::template __maybe_insert<_Ty>, _Us...>;
#endif
};
} // namespace __set

// When comparing sets for equality, use conjunction<> to short-circuit the set
// comparison if the sizes are different.
template <class _ExpectedSet, class... _Ts>
using __type_set_eq =
  conjunction<bool_constant<sizeof...(_Ts) == _ExpectedSet::__size()>, __type_set_contains<_ExpectedSet, _Ts...>>;

template <class _ExpectedSet, class... _Ts>
inline constexpr bool __type_set_eq_v = __type_set_eq<_ExpectedSet, _Ts...>::value;

template <class... _Ts>
using __type_set = __set::__tupl<_Ts...>;

template <class _Set, class... _Ts>
using __type_set_insert = typename __set::__bulk_insert<sizeof...(_Ts) == 0>::template __call<_Set, _Ts...>;

template <class... _Ts>
using __make_type_set = __type_set_insert<__type_set<>, _Ts...>;

template <class _Ty, class... _Ts>
struct __is_included_in : __fold_or<_CCCL_TRAIT(is_same, _Ty, _Ts)...>
{};

template <class _Ty, class... _Ts>
inline constexpr bool __is_included_in_v = __fold_or_v<is_same_v<_Ty, _Ts>...>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_TYPE_SET_H
