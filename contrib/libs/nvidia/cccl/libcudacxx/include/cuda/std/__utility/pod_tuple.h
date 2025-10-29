//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___UTILITY_POD_TUPLE_H
#define __CUDA_STD___UTILITY_POD_TUPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/undefined.h>

/**
 * @file pod_tuple.h
 * @brief Provides a lightweight implementation of a tuple-like structure that can be
 * aggregate-initialized. It can be used to return a tuple of immovable types from a function.
 * It is guaranteed to be a structural type up to 8 elements.
 *
 * This header defines the `__tuple` template and related utilities for creating and
 * manipulating tuples with compile-time optimizations.
 *
 * @details
 * The `__tuple` structure is designed to minimize template instantiations and improve
 * compile-time performance by unrolling tuples of sizes 1-8. It also provides utilities
 * for accessing tuple elements and applying callable objects to tuple contents.
 *
 * Key features:
 * - Lightweight tuple implementation that can be aggregate initialized.
 * - Tuple elements can be direct-initialized.
 * - Compile-time optimizations for small tuples (sizes 1-8).
 * - Support for callable application via `__apply`.
 * - Utilities for accessing tuple elements using `__get`.
 */

// Unroll tuples of size 1-8 to bring down the number of template instantiations and to
// permit __tuple to be used to initialize a structured binding without resorting to the
// heavy-weight std::tuple protocol. This code was generated with the following macros,
// which can be found here: https://godbolt.org/z/v6aac9v7E

/*
#define _CCCL_TUPLE_DEFINE_TPARAM(_Idx)  , class _CCCL_PP_CAT(_Tp, _Idx)
#define _CCCL_TUPLE_TPARAM(_Idx)         , _CCCL_PP_CAT(_Tp, _Idx)
#define _CCCL_TUPLE_DEFINE_ELEMENT(_Idx) _CCCL_NO_UNIQUE_ADDRESS _CCCL_PP_CAT(_Tp, _Idx) _CCCL_PP_CAT(__val, _Idx);
#define _CCCL_TUPLE_MBR(_Idx)            , static_cast<_Self&&>(__self)._CCCL_PP_CAT(__val, _Idx)

#define _CCCL_DEFINE_TUPLE(_SizeSub1)                                                                  \
  template <class _Tp0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_TPARAM, 1)>                       \
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_TPARAM, 1)> \
  {                                                                                                    \
    _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;                                                               \
    _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_ELEMENT, 1)                                          \
                                                                                                       \
    template <class _Fn, class _Self, class... _Us>                                                    \
    _CCCL_TRIVIAL_API static constexpr auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us)         \
    _CCCL_ARROW(static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)...,                                  \
                                         static_cast<_Self&&>(__self).__val0                           \
                                         _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_MBR, 1)))              \
  };

_CCCL_PP_REPEAT_REVERSE(_CCCL_TUPL_UNROLL_LIMIT, _CCCL_DEFINE_TUPLE)
*/

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunknown-warning-option") // "unknown warning group '-Wc++26-extensions'"
_CCCL_DIAG_SUPPRESS_CLANG("-Wc++26-extensions") // "pack indexing is a C++26 extension"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_COMPILER(GCC) || _CCCL_COMPILER(NVHPC)
// GCC (as of v14) does not implement the resolution of CWG1835
// https://cplusplus.github.io/CWG/issues/1835.html
// See: https://godbolt.org/z/TzxrhK6ea
#  define _CCCL_NO_CWG1835
#endif

#ifdef _CCCL_NO_CWG1835
#  define _CCCL_CWG1835_TEMPLATE
#else
#  define _CCCL_CWG1835_TEMPLATE template
#endif

#define _CCCL_ARROW(...)                                   \
  noexcept(noexcept(__VA_ARGS__))->decltype((__VA_ARGS__)) \
  {                                                        \
    return __VA_ARGS__;                                    \
  }

template <size_t _Index, class _Ty>
struct __box
{
  _CCCL_NO_UNIQUE_ADDRESS _Ty __value;
};

template <class _Index, class... _Ts>
struct __tupl_base;

template <size_t... _Index, class... _Ts>
struct _CCCL_DECLSPEC_EMPTY_BASES __tupl_base<index_sequence<_Index...>, _Ts...> : __box<_Index, _Ts>...
{
  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us)
    _CCCL_ARROW(static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self)._CCCL_CWG1835_TEMPLATE __box<_Index, _Ts>::__value...))
};

template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_DECLSPEC_EMPTY_BASES __tuple //
    : __tupl_base<index_sequence_for<_Ts...>, _Ts...>
{};

template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5, class _Tp6, class _Tp7>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5, _Tp6, _Tp7>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;
  _CCCL_NO_UNIQUE_ADDRESS _Tp5 __val5;
  _CCCL_NO_UNIQUE_ADDRESS _Tp6 __val6;
  _CCCL_NO_UNIQUE_ADDRESS _Tp7 __val7;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(noexcept(static_cast<_Fn&&>(__fn)(
    static_cast<_Us&&>(__us)...,
    static_cast<_Self&&>(__self).__val0,
    static_cast<_Self&&>(__self).__val1,
    static_cast<_Self&&>(__self).__val2,
    static_cast<_Self&&>(__self).__val3,
    static_cast<_Self&&>(__self).__val4,
    static_cast<_Self&&>(__self).__val5,
    static_cast<_Self&&>(__self).__val6,
    static_cast<_Self&&>(__self).__val7)))
    -> decltype((static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4,
      static_cast<_Self&&>(__self).__val5,
      static_cast<_Self&&>(__self).__val6,
      static_cast<_Self&&>(__self).__val7)))
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4,
      static_cast<_Self&&>(__self).__val5,
      static_cast<_Self&&>(__self).__val6,
      static_cast<_Self&&>(__self).__val7);
  }
};
template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5, class _Tp6>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5, _Tp6>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;
  _CCCL_NO_UNIQUE_ADDRESS _Tp5 __val5;
  _CCCL_NO_UNIQUE_ADDRESS _Tp6 __val6;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(noexcept(static_cast<_Fn&&>(__fn)(
    static_cast<_Us&&>(__us)...,
    static_cast<_Self&&>(__self).__val0,
    static_cast<_Self&&>(__self).__val1,
    static_cast<_Self&&>(__self).__val2,
    static_cast<_Self&&>(__self).__val3,
    static_cast<_Self&&>(__self).__val4,
    static_cast<_Self&&>(__self).__val5,
    static_cast<_Self&&>(__self).__val6)))
    -> decltype((static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4,
      static_cast<_Self&&>(__self).__val5,
      static_cast<_Self&&>(__self).__val6)))
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4,
      static_cast<_Self&&>(__self).__val5,
      static_cast<_Self&&>(__self).__val6);
  }
};
template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;
  _CCCL_NO_UNIQUE_ADDRESS _Tp5 __val5;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(noexcept(static_cast<_Fn&&>(__fn)(
    static_cast<_Us&&>(__us)...,
    static_cast<_Self&&>(__self).__val0,
    static_cast<_Self&&>(__self).__val1,
    static_cast<_Self&&>(__self).__val2,
    static_cast<_Self&&>(__self).__val3,
    static_cast<_Self&&>(__self).__val4,
    static_cast<_Self&&>(__self).__val5)))
    -> decltype((static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4,
      static_cast<_Self&&>(__self).__val5)))
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4,
      static_cast<_Self&&>(__self).__val5);
  }
};
template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(noexcept(static_cast<_Fn&&>(__fn)(
    static_cast<_Us&&>(__us)...,
    static_cast<_Self&&>(__self).__val0,
    static_cast<_Self&&>(__self).__val1,
    static_cast<_Self&&>(__self).__val2,
    static_cast<_Self&&>(__self).__val3,
    static_cast<_Self&&>(__self).__val4)))
    -> decltype((static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4)))
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3,
      static_cast<_Self&&>(__self).__val4);
  }
};
template <class _Tp0, class _Tp1, class _Tp2, class _Tp3>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(noexcept(static_cast<_Fn&&>(__fn)(
    static_cast<_Us&&>(__us)...,
    static_cast<_Self&&>(__self).__val0,
    static_cast<_Self&&>(__self).__val1,
    static_cast<_Self&&>(__self).__val2,
    static_cast<_Self&&>(__self).__val3)))
    -> decltype((static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3)))
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2,
      static_cast<_Self&&>(__self).__val3);
  }
};
template <class _Tp0, class _Tp1, class _Tp2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(noexcept(static_cast<_Fn&&>(__fn)(
    static_cast<_Us&&>(__us)...,
    static_cast<_Self&&>(__self).__val0,
    static_cast<_Self&&>(__self).__val1,
    static_cast<_Self&&>(__self).__val2)))
    -> decltype((static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2)))
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__val0,
      static_cast<_Self&&>(__self).__val1,
      static_cast<_Self&&>(__self).__val2);
  }
};
template <class _Tp0, class _Tp1>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(noexcept(static_cast<_Fn&&>(__fn)(
    static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__val0, static_cast<_Self&&>(__self).__val1)))
    -> decltype((static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__val0, static_cast<_Self&&>(__self).__val1)))
  {
    return static_cast<_Fn&&>(
      __fn)(static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__val0, static_cast<_Self&&>(__self).__val1);
  }
};
template <class _Tp0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    noexcept(static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__val0)))
    -> decltype((static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__val0)))
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__val0);
  }
};

template <class... _Ts>
_CCCL_HOST_DEVICE __tuple(_Ts...) -> __tuple<_Ts...>;

//
// __apply(fn, tuple, extra...)
//
_CCCL_EXEC_CHECK_DISABLE
template <class _Fn, class _Tuple, class... _Us>
_CCCL_TRIVIAL_API constexpr auto __apply(_Fn&& __fn, _Tuple&& __tuple, _Us&&... __us)
  _CCCL_ARROW(__tuple.__apply(static_cast<_Fn&&>(__fn), static_cast<_Tuple&&>(__tuple), static_cast<_Us&&>(__us)...))

    template <class _Fn, class _Tuple, class... _Us>
    using __apply_result_t _CCCL_NODEBUG_ALIAS =
      decltype(declval<_Tuple>().__apply(declval<_Fn>(), declval<_Tuple>(), declval<_Us>()...));

template <class _Fn, class _Tuple, class... _Us>
_CCCL_CONCEPT __applicable = _CUDA_VSTD::_IsValidExpansion<__apply_result_t, _Fn, _Tuple, _Us...>::value;

template <class _Fn, class _Tuple, class... _Us>
using __nothrow_applicable_detail_t =
  _CUDA_VSTD::enable_if_t<noexcept(declval<_Tuple>().__apply(declval<_Fn>(), declval<_Tuple>(), declval<_Us>()...))>;

template <class _Fn, class _Tuple, class... _Us>
_CCCL_CONCEPT __nothrow_applicable =
  _CUDA_VSTD::_IsValidExpansion<__nothrow_applicable_detail_t, _Fn, _Tuple, _Us...>::value;

//
// __get<I>(tupl)
//
namespace __detail
{
#if _CCCL_HAS_PACK_INDEXING()

template <size_t _Idx>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __get_fn
{
  template <class... _Ts>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Ts&&... __ts) const noexcept -> decltype(auto)
  {
    return static_cast<decltype(__ts...[_Idx])&&>(__ts...[_Idx]);
  }
};

#else // ^^^ _CCCL_HAS_PACK_INDEXING() ^^^ / vvv !_CCCL_HAS_PACK_INDEXING() vvv

template <size_t>
using __ignore_t = _CUDA_VSTD::__ignore_t;

template <size_t _Idx, class = _CUDA_VSTD::make_index_sequence<_Idx>>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __get_fn;

template <size_t _Idx, size_t... _Is>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __get_fn<_Idx, _CUDA_VSTD::index_sequence<_Is...>>
{
  template <class _Ty, class... _Rest>
  _CCCL_TRIVIAL_API constexpr auto operator()(__ignore_t<_Is>..., _Ty&& __ty, _Rest&&...) noexcept -> _Ty&&
  {
    return static_cast<_Ty&&>(__ty);
  }
};

#endif // !_CCCL_HAS_PACK_INDEXING()
} // namespace __detail

template <size_t _Idx, class _Tuple>
_CCCL_TRIVIAL_API constexpr auto __get(_Tuple&& __tuple) noexcept -> decltype(auto)
{
  return __tuple.__apply(__detail::__get_fn<_Idx>{}, static_cast<_Tuple&&>(__tuple));
}

//
// __decayed_tuple<Ts...>
//
template <class... _Ts>
using __decayed_tuple _CCCL_NODEBUG_ALIAS = __tuple<decay_t<_Ts>...>;

//
// __pair
//
template <class _First, class _Second>
struct __pair
{
  _CCCL_NO_UNIQUE_ADDRESS _First first;
  _CCCL_NO_UNIQUE_ADDRESS _Second second;
};

template <class _First, class _Second>
_CCCL_HOST_DEVICE __pair(_First, _Second) -> __pair<_First, _Second>;

//
// __tuple_size_v
//
template <class _Tuple>
extern __undefined<_Tuple> __tuple_size_v;

template <class... _Ts>
inline constexpr size_t __tuple_size_v<__tuple<_Ts...>> = sizeof...(_Ts);

template <class... _Ts>
inline constexpr size_t __tuple_size_v<const __tuple<_Ts...>> = sizeof...(_Ts);

template <class... _Ts>
inline constexpr size_t __tuple_size_v<__tuple<_Ts...>&> = sizeof...(_Ts);

template <class... _Ts>
inline constexpr size_t __tuple_size_v<const __tuple<_Ts...>&> = sizeof...(_Ts);

//
// __tuple_element_t
//
template <class _Tp>
_CCCL_API auto __remove_rvalue_ref(_Tp&&) noexcept -> _Tp;

template <size_t _Index, class _Tuple>
using __tuple_element_t _CCCL_NODEBUG_ALIAS =
  decltype(_CUDA_VSTD::__remove_rvalue_ref(_CUDA_VSTD::__get<_Index>(declval<_Tuple>())));

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA_STD___UTILITY_POD_TUPLE_H
