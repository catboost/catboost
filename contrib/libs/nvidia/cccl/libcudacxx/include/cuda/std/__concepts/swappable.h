//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_SWAPPABLE_H
#define _LIBCUDACXX___CONCEPTS_SWAPPABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__concepts/common_reference_with.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_COMPILER(MSVC)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(461) // nonstandard cast to array type ignored
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

// [concept.swappable]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__swap)

template <class _Tp>
void swap(_Tp&, _Tp&) = delete;

#if _CCCL_HAS_CONCEPTS()
template <class _Tp, class _Up>
concept __unqualified_swappable_with =
  (__class_or_enum<remove_cvref_t<_Tp>> || __class_or_enum<remove_cvref_t<_Up>>)
  && requires(_Tp&& __t, _Up&& __u) { swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u)); };

template <class _Tp>
concept __exchangeable =
  !__unqualified_swappable_with<_Tp&, _Tp&> && move_constructible<_Tp> && assignable_from<_Tp&, _Tp>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __unqualified_swappable_with_,
  requires(_Tp&& __t, _Up&& __u)((swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u)))));

template <class _Tp, class _Up>
_CCCL_CONCEPT __unqualified_swappable_with = _CCCL_FRAGMENT(__unqualified_swappable_with_, _Tp, _Up);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__exchangeable_,
                       requires()(requires(!__unqualified_swappable_with<_Tp&, _Tp&>),
                                  requires(move_constructible<_Tp>),
                                  requires(assignable_from<_Tp&, _Tp>)));

template <class _Tp>
_CCCL_CONCEPT __exchangeable = _CCCL_FRAGMENT(__exchangeable_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

#if _CCCL_HAS_CONCEPTS() && !_CCCL_COMPILER(NVHPC) // nvbug4051640
struct __fn;

_CCCL_BEGIN_NV_DIAG_SUPPRESS(2642)
template <class _Tp, class _Up, size_t _Size>
concept __swappable_arrays =
  !__unqualified_swappable_with<_Tp (&)[_Size], _Up (&)[_Size]> && extent_v<_Tp> == extent_v<_Up>
  && requires(_Tp (&__t)[_Size], _Up (&__u)[_Size], const __fn& __swap) { __swap(__t[0], __u[0]); };
_CCCL_END_NV_DIAG_SUPPRESS()

#else // ^^^ _CCCL_HAS_CONCEPTS() && !_CCCL_COMPILER(NVHPC) ^^^ / vvv !_CCCL_HAS_CONCEPTS() || _CCCL_COMPILER(NVHPC) vvv
template <class _Tp, class _Up, size_t _Size, class = void>
inline constexpr bool __swappable_arrays = false;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^ || _CCCL_COMPILER(NVHPC)

template <class _Tp, class _Up, class = void>
inline constexpr bool __noexcept_swappable_arrays = false;

struct __fn
{
  // 2.1   `S` is `(void)swap(E1, E2)`* if `E1` or `E2` has class or enumeration type and...
  // *The name `swap` is used here unqualified.
  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(__unqualified_swappable_with<_Tp, _Up>)
  _CCCL_API constexpr void operator()(_Tp&& __t, _Up&& __u) const
    noexcept(noexcept(swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u))))
  {
    swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u));
  }

  // 2.2   Otherwise, if `E1` and `E2` are lvalues of array types with equal extent and...
  _CCCL_TEMPLATE(class _Tp, class _Up, size_t _Size)
  _CCCL_REQUIRES(__swappable_arrays<_Tp, _Up, _Size>)
  _CCCL_API constexpr void operator()(_Tp (&__t)[_Size], _Up (&__u)[_Size]) const
    noexcept(__noexcept_swappable_arrays<_Tp, _Up>)
  {
    // TODO(cjdb): replace with `_CUDA_VRANGES::swap_ranges`.
    for (size_t __i = 0; __i < _Size; ++__i)
    {
      (*this)(__t[__i], __u[__i]);
    }
  }

  // 2.3   Otherwise, if `E1` and `E2` are lvalues of the same type `T` that models...
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__exchangeable<_Tp>)
  _CCCL_API constexpr void operator()(_Tp& __x, _Tp& __y) const
    noexcept(_CCCL_TRAIT(is_nothrow_move_constructible, _Tp) && _CCCL_TRAIT(is_nothrow_move_assignable, _Tp))
  {
    __y = _CUDA_VSTD::exchange(__x, _CUDA_VSTD::move(__y));
  }
};

#if !_CCCL_HAS_CONCEPTS() || _CCCL_COMPILER(NVHPC)
template <class _Tp, class _Up, class _Size>
_CCCL_CONCEPT_FRAGMENT(
  __swappable_arrays_,
  requires(_Tp (&__t)[_Size::value], _Up (&__u)[_Size::value], const __fn& __swap)(
    requires(!__unqualified_swappable_with<_Tp (&)[_Size::value], _Up (&)[_Size::value]>),
    requires(_CCCL_TRAIT(extent, _Tp) == _CCCL_TRAIT(extent, _Up)),
    (__swap(__t[0], __u[0]))));

template <class _Tp, class _Up, size_t _Size>
inline constexpr bool __swappable_arrays<_Tp, _Up, _Size, void_t<type_identity_t<_Tp>>> =
  _CCCL_FRAGMENT(__swappable_arrays_, _Tp, _Up, _CUDA_VSTD::integral_constant<size_t, _Size>);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^ || _CCCL_COMPILER(NVHPC)

template <class _Tp, class _Up>
inline constexpr bool __noexcept_swappable_arrays<_Tp, _Up, void_t<type_identity_t<_Tp>>> =
  noexcept(__swap::__fn{}(_CUDA_VSTD::declval<_Tp&>(), _CUDA_VSTD::declval<_Up&>()));

_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto swap = __swap::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept swappable = requires(_Tp& __a, _Tp& __b) { _CUDA_VRANGES::swap(__a, __b); };

template <class _Tp, class _Up>
concept swappable_with = common_reference_with<_Tp, _Up> && requires(_Tp&& __t, _Up&& __u) {
  _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Tp>(__t));
  _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Up>(__u));
  _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u));
  _CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Tp>(__t));
};
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__swappable_, requires(_Tp& __a, _Tp& __b)((_CUDA_VRANGES::swap(__a, __b))));

template <class _Tp>
_CCCL_CONCEPT swappable = _CCCL_FRAGMENT(__swappable_, _Tp);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __swappable_with_,
  requires(_Tp&& __t, _Up&& __u)(
    requires(common_reference_with<_Tp, _Up>),
    (_CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Tp>(__t))),
    (_CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Up>(__u))),
    (_CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Tp>(__t), _CUDA_VSTD::forward<_Up>(__u))),
    (_CUDA_VRANGES::swap(_CUDA_VSTD::forward<_Up>(__u), _CUDA_VSTD::forward<_Tp>(__t)))));

template <class _Tp, class _Up>
_CCCL_CONCEPT swappable_with = _CCCL_FRAGMENT(__swappable_with_, _Tp, _Up);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#if _CCCL_COMPILER(MSVC)
_CCCL_END_NV_DIAG_SUPPRESS() // nonstandard cast to array type ignored
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_SWAPPABLE_H
