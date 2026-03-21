//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_COMMON_TYPE_H
#define _LIBCUDACXX___TYPE_TRAITS_COMMON_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT common_type;

template <class... _Tp>
using common_type_t _CCCL_NODEBUG_ALIAS = typename common_type<_Tp...>::type;

// Let COND_RES(X, Y) be:
template <class _Tp, class _Up>
using __cond_type = decltype(false ? declval<_Tp>() : declval<_Up>());

// We need to ensure that extended floating point types like __half and __nv bfloat16 have a common type with real
// floating point types
template <class _Tp, class _Up, class = void>
struct __common_type_extended_floating_point
{};

#if !defined(__CUDA_NO_HALF_CONVERSIONS__) && !defined(__CUDA_NO_HALF_OPERATORS__) \
  && !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__) && !defined(__CUDA_NO_BFLOAT16_OPERATORS__)
template <class _Tp, class _Up>
struct __common_type_extended_floating_point<_Tp,
                                             _Up,
                                             enable_if_t<_CCCL_TRAIT(__is_extended_floating_point, remove_cvref_t<_Tp>)
                                                         && _CCCL_TRAIT(is_arithmetic, remove_cvref_t<_Up>)>>
{
  using type = common_type_t<__copy_cvref_t<_Tp, float>, _Up>;
};

template <class _Tp, class _Up>
struct __common_type_extended_floating_point<
  _Tp,
  _Up,
  enable_if_t<_CCCL_TRAIT(is_arithmetic, remove_cvref_t<_Tp>)
              && _CCCL_TRAIT(__is_extended_floating_point, remove_cvref_t<_Up>)>>
{
  using type = common_type_t<_Tp, __copy_cvref_t<_Up, float>>;
};
#endif // extended floating point as arithmetic type

template <class _Tp, class _Up, class = void>
struct __common_type3 : __common_type_extended_floating_point<_Tp, _Up>
{};

#if _CCCL_STD_VER >= 2020
// sub-bullet 4 - "if COND_RES(CREF(D1), CREF(D2)) denotes a type..."
template <class _Tp, class _Up>
struct __common_type3<_Tp, _Up, void_t<__cond_type<const _Tp&, const _Up&>>>
{
  using type = remove_cvref_t<__cond_type<const _Tp&, const _Up&>>;
};
#endif // _CCCL_STD_VER >= 2020

template <class _Tp, class _Up, class = void>
struct __common_type2_imp : __common_type3<_Tp, _Up>
{};

// MSVC has a bug in its declval handling, where it happily accepts __cond_type<_Tp, _Up>, even though both
// branches have diverging return types, this happens for extended floating point types
template <class _Tp, class _Up>
using __msvc_declval_workaround =
#if _CCCL_COMPILER(MSVC)
  enable_if_t<_CCCL_TRAIT(is_same, __cond_type<_Tp, _Up>, __cond_type<_Up, _Tp>)>;
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  void;
#endif // !_CCCL_COMPILER(MSVC)

// sub-bullet 3 - "if decay_t<decltype(false ? declval<D1>() : declval<D2>())> ..."
template <class _Tp, class _Up>
struct __common_type2_imp<_Tp, _Up, void_t<__cond_type<_Tp, _Up>, __msvc_declval_workaround<_Tp, _Up>>>
{
  using type _CCCL_NODEBUG_ALIAS = decay_t<__cond_type<_Tp, _Up>>;
};

template <class, class = void>
struct __common_type_impl
{};

template <class... _Tp>
struct __common_types;

template <class _Tp, class _Up>
struct __common_type_impl<__common_types<_Tp, _Up>, void_t<common_type_t<_Tp, _Up>>>
{
  using type = common_type_t<_Tp, _Up>;
};

template <class _Tp, class _Up, class _Vp, class... _Rest>
struct __common_type_impl<__common_types<_Tp, _Up, _Vp, _Rest...>, void_t<common_type_t<_Tp, _Up>>>
    : __common_type_impl<__common_types<common_type_t<_Tp, _Up>, _Vp, _Rest...>>
{};

// bullet 1 - sizeof...(Tp) == 0

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT common_type<>
{};

// bullet 2 - sizeof...(Tp) == 1

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT common_type<_Tp> : public common_type<_Tp, _Tp>
{};

// bullet 3 - sizeof...(Tp) == 2

// sub-bullet 1 - "If is_same_v<T1, D1> is false or ..."
template <class _Tp, class _Up, class _D1 = decay_t<_Tp>, class _D2 = decay_t<_Up>>
struct __common_type2 : common_type<_D1, _D2>
{};

template <class _Tp, class _Up>
struct __common_type2<_Tp, _Up, _Tp, _Up> : __common_type2_imp<_Tp, _Up>
{};

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT common_type<_Tp, _Up> : __common_type2<_Tp, _Up>
{};

// bullet 4 - sizeof...(Tp) > 2

template <class _Tp, class _Up, class _Vp, class... _Rest>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
common_type<_Tp, _Up, _Vp, _Rest...> : __common_type_impl<__common_types<_Tp, _Up, _Vp, _Rest...>>
{};

template <class... _Tp>
using common_type_t _CCCL_NODEBUG_ALIAS = typename common_type<_Tp...>::type;

template <class, class, class = void>
inline constexpr bool __has_common_type = false;

template <class _Tp, class _Up>
inline constexpr bool __has_common_type<_Tp, _Up, void_t<common_type_t<_Tp, _Up>>> = true;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_COMMON_TYPE_H
