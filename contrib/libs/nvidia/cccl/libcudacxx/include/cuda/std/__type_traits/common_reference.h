//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_COMMON_REFERENCE_H
#define _LIBCUDACXX___TYPE_TRAITS_COMMON_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/copy_cv.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NV_DIAG_SUPPRESS(1384) // warning: pointer converted to bool

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// common_reference

// Let COND_RES(X, Y) be:
#if _CCCL_COMPILER(MSVC) // Workaround for DevCom-1627396
template <class _Tp>
_Tp __returns_exactly() noexcept; // not defined

template <class _Xp, class _Yp>
using __cond_res_if_right = decltype(false ? __returns_exactly<_Xp>() : __returns_exactly<_Yp>());

template <class _Tp, class _Up, class = void>
struct __cond_res_workaround
{};

template <class _Tp, class _Up>
struct __cond_res_workaround<_Tp, _Up, void_t<__cond_res_if_right<_Tp, _Up>>>
{
  using _RTp = remove_cvref_t<_Tp>;
  using type =
    conditional_t<is_same_v<_RTp, remove_cvref_t<_Up>> && (is_scalar_v<_RTp> || is_array_v<_RTp>)
                    && ((is_lvalue_reference_v<_Tp> && is_rvalue_reference_v<_Up>)
                        || (is_rvalue_reference_v<_Tp> && is_lvalue_reference_v<_Up>) ),
                  decay_t<__copy_cv_t<remove_reference_t<_Tp>, remove_reference_t<_Up>>>,
                  __cond_res_if_right<_Tp, _Up>>;
};

template <class _Xp, class _Yp>
using __cond_res = typename __cond_res_workaround<_Xp, _Yp>::type;
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
template <class _Xp, class _Yp>
using __cond_res = decltype(false ? _CUDA_VSTD::declval<_Xp (&)()>()() : _CUDA_VSTD::declval<_Yp (&)()>()());
#endif // !_CCCL_COMPILER(MSVC)

// Let `XREF(A)` denote a unary alias template `T` such that `T<U>` denotes the same type as `U`
// with the addition of `A`'s cv and reference qualifiers, for a non-reference cv-unqualified type
// `U`.
// [Note: `XREF(A)` is `__xref<A>::template __call`]
template <class _Tp>
using __xref = __apply_cvref_fn<_Tp>;

// Given types A and B, let X be remove_reference_t<A>, let Y be remove_reference_t<B>,
// and let COMMON-REF(A, B) be:
template <class _Ap, class _Bp, class = void>
struct __common_ref;

template <class _Xp, class _Yp>
using __common_ref_t _CCCL_NODEBUG_ALIAS = typename __common_ref<_Xp, _Yp>::__type;

template <class _Xp, class _Yp>
using __cv_cond_res = __cond_res<__copy_cv_t<_Xp, _Yp>&, __copy_cv_t<_Yp, _Xp>&>;

//    If A and B are both lvalue reference types, COMMON-REF(A, B) is
//    COND-RES(COPYCV(X, Y)&, COPYCV(Y, X)&) if that type exists and is a reference type.
template <class _Ap, class _Bp>
struct __common_ref<_Ap&, _Bp&, enable_if_t<_CCCL_TRAIT(is_reference, __cv_cond_res<_Ap, _Bp>)>>
{
  using __type = __cv_cond_res<_Ap, _Bp>;
};

//    Otherwise, let C be remove_reference_t<COMMON-REF(X&, Y&)>&&. ...
template <class _Xp, class _Yp>
using __common_ref_C = remove_reference_t<__common_ref_t<_Xp&, _Yp&>>&&;

//    .... If A and B are both rvalue reference types, C is well-formed, and
//    is_convertible_v<A, C> && is_convertible_v<B, C> is true, then COMMON-REF(A, B) is C.
template <class _Ap, class _Bp, class = void>
struct __common_ref_rr
{};

template <class _Ap, class _Bp>
struct __common_ref_rr<_Ap&&,
                       _Bp&&,
                       enable_if_t<_CCCL_TRAIT(is_convertible, _Ap&&, __common_ref_C<_Ap, _Bp>)
                                   && _CCCL_TRAIT(is_convertible, _Bp&&, __common_ref_C<_Ap, _Bp>)>>
{
  using __type = __common_ref_C<_Ap, _Bp>;
};

template <class _Ap, class _Bp>
struct __common_ref<_Ap&&, _Bp&&> : __common_ref_rr<_Ap&&, _Bp&&>
{};

//    Otherwise, let D be COMMON-REF(const X&, Y&). ...
template <class _Tp, class _Up>
using __common_ref_D = __common_ref_t<const _Tp&, _Up&>;

//    ... If A is an rvalue reference and B is an lvalue reference and D is well-formed and
//    is_convertible_v<A, D> is true, then COMMON-REF(A, B) is D.
template <class _Ap, class _Bp, class = void>
struct __common_ref_lr
{};

template <class _Ap, class _Bp>
struct __common_ref_lr<_Ap&&, _Bp&, enable_if_t<_CCCL_TRAIT(is_convertible, _Ap&&, __common_ref_D<_Ap, _Bp>)>>
{
  using __type = __common_ref_D<_Ap, _Bp>;
};

template <class _Ap, class _Bp>
struct __common_ref<_Ap&&, _Bp&> : __common_ref_lr<_Ap&&, _Bp&>
{};

//    Otherwise, if A is an lvalue reference and B is an rvalue reference, then
//    COMMON-REF(A, B) is COMMON-REF(B, A).
template <class _Ap, class _Bp>
struct __common_ref<_Ap&, _Bp&&> : __common_ref_lr<_Bp&&, _Ap&>
{};

//    Otherwise, COMMON-REF(A, B) is ill-formed.
template <class _Ap, class _Bp, class>
struct __common_ref
{};

// Note C: For the common_reference trait applied to a parameter pack [...]

template <class...>
struct common_reference;

template <class... _Types>
using common_reference_t _CCCL_NODEBUG_ALIAS = typename common_reference<_Types...>::type;

template <class, class, class = void>
inline constexpr bool __has_common_reference = false;

template <class _Tp, class _Up>
inline constexpr bool __has_common_reference<_Tp, _Up, void_t<common_reference_t<_Tp, _Up>>> = true;

// bullet 1 - sizeof...(T) == 0
template <>
struct common_reference<>
{};

// bullet 2 - sizeof...(T) == 1
template <class _Tp>
struct common_reference<_Tp>
{
  using type = _Tp;
};

// bullet 3 - sizeof...(T) == 2
template <class _Tp, class _Up, class = void>
struct __common_reference_sub_bullet3;
template <class _Tp, class _Up, class = void>
struct __common_reference_sub_bullet2 : __common_reference_sub_bullet3<_Tp, _Up>
{};
template <class _Tp, class _Up, class = void>
struct __common_reference_sub_bullet1 : __common_reference_sub_bullet2<_Tp, _Up>
{};

// sub-bullet 1 - If T1 and T2 are reference types and COMMON-REF(T1, T2) is well-formed, then
// the member typedef `type` denotes that type.
template <class _Tp, class _Up>
struct common_reference<_Tp, _Up> : __common_reference_sub_bullet1<_Tp, _Up>
{};

template <class _Tp, class _Up>
struct __common_reference_sub_bullet1<
  _Tp,
  _Up,
  void_t<__common_ref_t<_Tp, _Up>, enable_if_t<_CCCL_TRAIT(is_reference, _Tp) && _CCCL_TRAIT(is_reference, _Up)>>>
{
  using type = __common_ref_t<_Tp, _Up>;
};

// sub-bullet 2 - Otherwise, if basic_common_reference<remove_cvref_t<T1>, remove_cvref_t<T2>, XREF(T1), XREF(T2)>::type
// is well-formed, then the member typedef `type` denotes that type.
template <class, class, template <class> class, template <class> class>
struct basic_common_reference
{};

template <class _Tp, class _Up>
using __basic_common_reference_t _CCCL_NODEBUG_ALIAS =
  typename basic_common_reference<remove_cvref_t<_Tp>,
                                  remove_cvref_t<_Up>,
                                  __xref<_Tp>::template __call,
                                  __xref<_Up>::template __call>::type;

template <class _Tp, class _Up>
struct __common_reference_sub_bullet2<_Tp, _Up, void_t<__basic_common_reference_t<_Tp, _Up>>>
{
  using type = __basic_common_reference_t<_Tp, _Up>;
};

// sub-bullet 3 - Otherwise, if COND-RES(T1, T2) is well-formed,
// then the member typedef `type` denotes that type.
template <class _Tp, class _Up>
struct __common_reference_sub_bullet3<_Tp, _Up, void_t<__cond_res<_Tp, _Up>>>
{
  using type = __cond_res<_Tp, _Up>;
};

// sub-bullet 4 & 5 - Otherwise, if common_type_t<T1, T2> is well-formed,
//                    then the member typedef `type` denotes that type.
//                  - Otherwise, there shall be no member `type`.
template <class _Tp, class _Up, class>
struct __common_reference_sub_bullet3 : common_type<_Tp, _Up>
{};

// bullet 4 - If there is such a type `C`, the member typedef type shall denote the same type, if
//            any, as `common_reference_t<C, Rest...>`.
template <class _Tp, class _Up, class _Vp, class... _Rest>
struct common_reference<_Tp, _Up, _Vp, void_t<common_reference_t<_Tp, _Up>>, _Rest...>
    : common_reference<common_reference_t<_Tp, _Up>, _Vp, _Rest...>
{};

// bullet 5 - Otherwise, there shall be no member `type`.
template <class...>
struct common_reference
{};

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_END_NV_DIAG_SUPPRESS()

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_COMMON_REFERENCE_H
