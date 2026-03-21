//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H
#define _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/undefined.h>

//! \file type_list.h
//! This file defines a type-list type and some fundamental algorithms on type
//! lists.
//!
//! It also defines a "meta-callable" protocol for parameterizing the type list
//! algorithms, and some higher-order meta-callables. A meta-callable is a class
//! type with a nested \c __call template alias.
//!
//! For the purpose of this file, a "trait type" is a class type with a nested
//! type alias named \c type.

#if !defined(_CCCL_META_UNROLL_LIMIT)
#  if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVHPC)
#    define _CCCL_META_UNROLL_LIMIT 10
#  else
#    define _CCCL_META_UNROLL_LIMIT 16
#  endif
#endif

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

template <class... _Ts>
struct __type_list;

template <class _Ty>
using __type = typename _Ty::type;

//! \brief Evaluate a meta-callable with the given arguments
template <class _Fn, class... _Ts>
using __type_call = typename _Fn::template __call<_Ts...>;

//! \brief Evaluate a unary meta-callable with the given argument
template <class _Fn, class _Ty>
using __type_call1 = typename _Fn::template __call<_Ty>;

//! \brief Evaluate a binary meta-callable with the given arguments
template <class _Fn, class _Ty, class _Uy>
using __type_call2 = typename _Fn::template __call<_Ty, _Uy>;

namespace __detail
{
template <size_t _DependentValue>
struct __type_call_indirect_fn
{
  template <template <class...> class _Fn, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = _Fn<_Ts...>;
};
} // namespace __detail

//! \brief Evaluate a meta-callable with the given arguments, with an indirection
//! to avoid the dreaded "pack expansion argument for non-pack parameter" error.
template <class _Fn, class... _Ts>
using __type_call_indirect = //
  typename __detail::__type_call_indirect_fn<sizeof(
    __type_list<_Fn, _Ts...>*)>::template __call<_Fn::template __call, _Ts...>;

//! \brief Turns a class or alias template into a meta-callable
template <template <class...> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = _Fn<_Ts...>;
};

//! \brief Turns a unary class or alias template into a meta-callable
template <template <class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote1
{
  template <class _Ty>
  using __call _CCCL_NODEBUG_ALIAS = _Fn<_Ty>;
};

//! \brief Turns a binary class or alias template into a meta-callable
template <template <class, class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote2
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = _Fn<_Ty, _Uy>;
};

//! \brief Turns a trait class template \c _Fn into a meta-callable \c
//! __type_quote_trait such that \c __type_quote_trait<_Fn>::__call<_Ts...> is
//! equivalent to \c _Fn<_Ts...>::type.
template <template <class...> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type<_Fn<_Ts...>>;
};

//! \brief Turns a unary trait class template \c _Fn into a meta-callable
//! \c __type_quote_trait1 such that \c __type_quote_trait1<_Fn>::__call<_Ty> is
//! equivalent to \c _Fn<_Ty>::type.
template <template <class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait1
{
  template <class _Ty>
  using __call _CCCL_NODEBUG_ALIAS = __type<_Fn<_Ty>>;
};

//! \brief Turns a binary trait class template \c _Fn into a meta-callable
//! \c __type_quote_trait2 such that \c __type_quote_trait2<_Fn>::__call<_Ty,_Uy>
//! is equivalent to \c _Fn<_Ty,_Uy>::type.
template <template <class, class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait2
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = __type<_Fn<_Ty, _Uy>>;
};

//! \brief Adds an indirection to a meta-callable to avoid the dreaded "pack
//! expansion argument for non-pack parameter" error.
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_indirect
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = typename __detail::__type_call_indirect_fn<sizeof(
    __type_list<_Ts...>*)>::template __call<_Fn::template __call, _Ts...>;
};

//! \brief Turns a class or alias template into a meta-callable with an
//! indirection to avoid the dreaded "pack expansion argument for non-pack
//! parameter" error.
template <template <class...> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_indirect_quote
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS =
    typename __detail::__type_call_indirect_fn<sizeof(__type_list<_Ts...>*)>::template __call<_Fn, _Ts...>;
};

//! \brief A meta-callable that composes two meta-callables
template <class _Fn1, class _Fn2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_compose
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_call1<_Fn1, __type_call<_Fn2, _Ts...>>;
};

template <template <class...> class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_front_quote
{
  template <class... _Us>
  using __call _CCCL_NODEBUG_ALIAS = _Fn<_Ts..., _Us...>;
};

//! \brief A meta-callable that binds the front arguments to a meta-callable
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_front : __type_bind_front_quote<_Fn::template __call, _Ts...>
{};

template <template <class...> class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_back_quote
{
  template <class... _Us>
  using __call _CCCL_NODEBUG_ALIAS = _Fn<_Us..., _Ts...>;
};

//! \brief A meta-callable that binds the back arguments to a meta-callable
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_back : __type_bind_back_quote<_Fn::template __call, _Ts...>
{};

//! \brief A meta-callable that always evaluates to \c _Ty.
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_always
{
  template <class...>
  using __call _CCCL_NODEBUG_ALIAS = _Ty;
};

//! \brief A unary meta-callable that returns its argument unmodified.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_self
{
  template <class _Ty>
  using __call _CCCL_NODEBUG_ALIAS = _Ty;
};

template <class _Ty>
using __type_self_t _CCCL_NODEBUG_ALIAS = _Ty;

//! \brief Perform a logical AND operation on a list of Boolean types.
//!
//! \note The AND operation is not short-circuiting.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_strict_and
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __fold_and<_Ts::value...>;
};

//! \brief Perform a logical OR operation on a list of Boolean types.
//!
//! \note The OR operation is not short-circuiting.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_strict_or
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __fold_or<_Ts::value...>;
};

//! \brief Perform a logical NOT operation on a Boolean type.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_not
{
  template <class _Ty>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<(!_Ty::value)>;
};

//! \brief Test whether two integral constants are equal.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_equal
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<(_Ty::value == _Uy::value)>;
};

//! \brief Test whether two integral constants are not equal.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_not_equal
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<(_Ty::value != _Uy::value)>;
};

//! \brief Test whether one integral constant is less than another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_less
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<(_Ty::value < _Uy::value)>;
};

//! \brief Test whether one integral constant is less than or equal to another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_less_equal
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<(_Ty::value <= _Uy::value)>;
};

//! \brief Test whether one integral constant is greater than another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_greater
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<(_Ty::value > _Uy::value)>;
};

//! \brief Test whether one integral constant is greater than or equal to another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_greater_equal
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<(_Ty::value >= _Uy::value)>;
};

//! \brief A functional adaptor that negates a unary predicate
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_negate1
{
  template <class _Ty>
  using __call _CCCL_NODEBUG_ALIAS = __type_call1<__type_not, __type_call1<_Fn, _Ty>>;
};

//! \brief A functional adaptor that negates a binary predicate
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_negate2
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = __type_call1<__type_not, __type_call2<_Fn, _Ty, _Uy>>;
};

//! \brief A type list
template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_list
{
  static constexpr size_t const __size = sizeof...(_Ts);

  // A type_list behaves like a meta-callable
  // that takes a meta-callable and applies the
  // elements of the list to it.
  template <class _Fn, class... _Us>
  using __call _CCCL_NODEBUG_ALIAS = __type_call<_Fn, _Ts..., _Us...>;
};

// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
template <class... _Ts>
constexpr size_t const __type_list<_Ts...>::__size;

//! \brief A pointer to a type list, often used as a function argument.
template <class... _Ts>
using __type_list_ptr = __type_list<_Ts...>*;

//! \brief Return the size of a type list.
template <class _List>
using __type_list_size = integral_constant<size_t, _List::__size>;

//! \brief Given a type list and a list of types, append the types to the list.
template <class _List, class... _Ts>
using __type_push_back = __type_call<_List, __type_quote<__type_list>, _Ts...>;

//! \brief Given a type list and a list of types, prepend the types to the list.
template <class _List, class... _Ts>
using __type_push_front = __type_call1<_List, __type_bind_front_quote<__type_list, _Ts...>>;

namespace __detail
{
template <template <class...> class _Fn, class... _Ts>
_CCCL_API inline auto __as_type_list_fn(__undefined<_Fn<_Ts...>>*) //
  -> __type_list<_Ts...>;

template <template <class _Ty, _Ty...> class _Fn, class _Ty, _Ty... _Us>
_CCCL_API inline auto __as_type_list_fn(__undefined<_Fn<_Ty, _Us...>>*) //
  -> __type_list<integral_constant<_Ty, _Us>...>;

template <class _Ret, class... _Args>
_CCCL_API inline auto __as_type_list_fn(__undefined<_Ret(_Args...)>*) //
  -> __type_list<_Ret, _Args...>;
} // namespace __detail

//! \brief Given a type that is can be interpreted as a type list, return its
//! type list interpretation. Types that can be interpreted as a type
//! list are of the following forms:
//!
//! \li `C<Ts...>`, for any class template `C` and types `Ts...`.
//! \li `C<T, T... Vs>`, for any class template `C`, type `T` and values `Vs...`.
//!     The resulting type is `_Fn<integral_constant<T, Vs>...>`.
//! \li `R(As...)`, for any function type `R(As...)`. The resulting type is
//!     `_Fn<R, As...>`.
template <class _List>
using __as_type_list = decltype(__detail::__as_type_list_fn(static_cast<__undefined<_List>*>(nullptr)));

//! \brief Given a type that can be interpreted as a type list and a
//! meta-callable, invoke the meta-callable with the types in the list.
//! \c __as_type_list is used to convert the \c _List type to a specialization
//! of \c __type_list.
//!
//! \sa __as_type_list
template <class _Fn, class _List>
using __type_apply = __type_call<__as_type_list<_List>, _Fn>;

//! \brief Given a type that can be interpreted as a type list and a class or
//! alias template, instantiate the template with the types in the list.
//! \c __as_type_list is used to convert the \c _List type to a specialization
//! of \c __type_list.
//!
//! \sa __as_type_list
template <template <class...> class _Fn, class _List>
using __type_apply_q = __type_call<__as_type_list<_List>, __type_quote<_Fn>>;

namespace __detail
{
// Only the following precise formulation works with nvcc < 12.2
template <class _Fn, class... _Ts, template <class...> class _Fn2 = _Fn::template __call, class = _Fn2<_Ts...>>
_CCCL_API inline auto __type_callable_fn(__type_list<_Fn, _Ts...>*) -> true_type;

_CCCL_API inline auto __type_callable_fn(void*) -> false_type;
} // namespace __detail

//! \brief Test whether a meta-callable is callable with a given set of
//! arguments.
//!
//! Given a meta-callable and a list of arguments, return \c true_type if
//! the meta-callable can be called with the arguments, and \c false_type
//! otherwise.
template <class _Fn, class... _Ts>
using __type_callable = decltype(__detail::__type_callable_fn(static_cast<__type_list<_Fn, _Ts...>*>(nullptr)));

namespace __detail
{
template <bool _IsCallable>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_defer_fn
{
  template <class, class...>
  using __call _CCCL_NODEBUG_ALIAS = __type_defer_fn;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_defer_fn<true>
{
  template <class _Fn, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = type_identity<__type_call<_Fn, _Ts...>>;
};
} // namespace __detail

//! \brief Defer the evaluation of a meta-callable with a list of arguments.
//!
//! Given a meta-callable and a list of arguments, return a trait type \c T
//! where \c T::type is the result of evaluating the meta-callable with the
//! arguments, or if the meta-callable is not callable with the arguments, a
//! class type without a nested \c ::type type alias.
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
__type_defer : __type_call<__detail::__type_defer_fn<__type_callable<_Fn, _Ts...>::value>, _Fn, _Ts...>
{};

//! \brief Defer the instantiation of a template with a list of arguments.
//!
//! Given a variadic template and a list of arguments, return a trait type \c T
//! where \c T::type is the result of instantiating the template with the
//! arguments, or if the template cannot be instantiated with the arguments, a
//! class type without a nested \c ::type type alias.
template <template <class...> class _Fn, class... _Ts>
using __type_defer_quote = __type_defer<__type_quote<_Fn>, _Ts...>;

//! \brief A composition of two meta-callables that will attempt to call the
//! first, and if that fails, call the second.
//!
//! Compose two meta-callables, \c _TryFn and \c _CatchFn, into a new
//! meta-callable. Given arguments \c _Ts... , \c __type_try_catch will try to
//! evaluate `__type_call<_TryFn, _Ts...>`. If this type is well-formed, it will
//! be the result of the composition. Otherwise, the result will be
//! `__type_call<_CatchFn, _Ts...>`.
template <class _TryFn, class _CatchFn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_try_catch
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_call<_If<__type_callable<_TryFn, _Ts...>::value, _TryFn, _CatchFn>, _Ts...>;
};

// Implementation for indexing into a list of types:
#  if _CCCL_HAS_PACK_INDEXING()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wc++26-extensions")

template <size_t _Ip, class... _Ts>
using __type_index_c = _Ts...[_Ip];

template <class _Ip, class... _Ts>
using __type_index = _Ts...[_Ip::value];

_CCCL_DIAG_POP

// Versions of nvcc prior to 12.0 have trouble with pack expansion into
// __type_pack_element in an alias template, so we use the fall-back
// implementation instead.
#  elif defined(_CCCL_BUILTIN_TYPE_PACK_ELEMENT)

namespace __detail
{
// On some versions of gcc, __type_pack_element cannot be mangled so
// hide it behind a named template.
template <size_t _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_index_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_TYPE_PACK_ELEMENT(_Ip, _Ts...);
};
} // namespace __detail

template <size_t _Ip, class... _Ts>
using __type_index_c = __type_call<__detail::__type_index_fn<_Ip>, _Ts...>;

template <class _Ip, class... _Ts>
using __type_index = __type_call<__detail::__type_index_fn<_Ip::value>, _Ts...>;

#  else // ^^^ _CCCL_BUILTIN_TYPE_PACK_ELEMENT ^^^ / vvv !_CCCL_BUILTIN_TYPE_PACK_ELEMENT vvv

// Fallback implementation for __type_index uses multiple inheritance. See:
// https://ldionne.com/2015/11/29/efficient-parameter-pack-indexing/

namespace __detail
{
template <size_t _Ip, class _Ty>
struct __type_tuple_elem
{
  using type _CCCL_NODEBUG_ALIAS = _Ty;
};

template <class _Is, class... _Ts>
struct __type_tupl;

template <size_t... _Is, class... _Ts>
struct __type_tupl<index_sequence<_Is...>, _Ts...> : __type_tuple_elem<_Is, _Ts>...
{};

template <class... _Ts>
using __type_tuple = __type_tupl<make_index_sequence<sizeof...(_Ts)>, _Ts...>;

template <size_t _Ip, class _Ty>
_CCCL_API inline __type_tuple_elem<_Ip, _Ty> __type_tuple_get(__type_tuple_elem<_Ip, _Ty>);

template <size_t _Ip, class... _Ts>
using __type_tuple_element_t _CCCL_NODEBUG_ALIAS =
  __type<decltype(__detail::__type_tuple_get<_Ip>(__type_tuple<_Ts...>{}))>;

struct __type_index_large_size_fn
{
  template <class _Ip, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_tuple_element_t<_Ip::value, _Ts...>;
};

template <size_t _Ip>
struct __type_index_small_size_fn;

#    define _M0(X) class,
#    define _M1(_N)                                             \
      template <>                                               \
      struct __type_index_small_size_fn<_N>                     \
      {                                                         \
        template <_CCCL_PP_REPEAT(_N, _M0) class _Ty, class...> \
        using __call _CCCL_NODEBUG_ALIAS = _Ty;                 \
      };

_CCCL_PP_REPEAT_REVERSE(_CCCL_META_UNROLL_LIMIT, _M1)

#    undef _M0
#    undef _M1

template <bool _IsSmall>
struct __type_index_select_fn : __type_index_large_size_fn // Default for larger indices
{};

template <>
struct __type_index_select_fn<true> // Fast implementation for smaller indices
{
  template <class _Ip, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_call_indirect<__type_index_small_size_fn<_Ip::value>, _Ts...>;
};
} // namespace __detail

#    if !_CCCL_COMPILER(MSVC)

template <class _Ip, class... _Ts>
using __type_index = __type_call<__detail::__type_index_select_fn<(_Ip::value < _CCCL_META_UNROLL_LIMIT)>, _Ip, _Ts...>;

#    else // ^^^ !_CCCL_COMPILER(MSVC) ^^^ / vvv _CCCL_COMPILER(MSVC) vvv

// Simplify the implementation for MSVC, which has trouble with the above
template <class _Ip, class... _Ts>
using __type_index = __detail::__type_index_large_size_fn::__call<_Ip, _Ts...>;

#    endif // !_CCCL_COMPILER(MSVC)

template <size_t _Ip, class... _Ts>
using __type_index_c = __type_index<integral_constant<size_t, _Ip>, _Ts...>;

#  endif // !_CCCL_BUILTIN_TYPE_PACK_ELEMENT

namespace __detail
{
template <size_t _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_at_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_index_c<_Ip, _Ts...>;
};
} // namespace __detail

//! \brief Given a type list and an index, return the type at that index.
template <size_t _Ip, class _List>
using __type_at_c = __type_call1<_List, __detail::__type_at_fn<_Ip>>;

//! \brief Given a type list and an index, return the type at that index.
template <class _Ip, class _List>
using __type_at = __type_call1<_List, __detail::__type_at_fn<_Ip::value>>;

//! \brief Given a type list return the type at the front of the list.
template <class _List>
using __type_front = __type_at_c<0, _List>;

//! \brief Given a type list return the type at the back of the list.
template <class _List>
using __type_back = __type_at_c<_List::__size - 1, _List>;

//! \brief A pair of types
template <class _First, class _Second>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_pair
{
  using __first _CCCL_NODEBUG_ALIAS  = _First;
  using __second _CCCL_NODEBUG_ALIAS = _Second;
};

//! \brief Retrieve the first of a pair of types
//! \pre \c _Pair is a specialization of \c __type_pair
template <class _Pair>
using __type_pair_first _CCCL_NODEBUG_ALIAS = typename _Pair::__first;

//! \brief Retrieve the second of a pair of types
//! \pre \c _Pair is a specialization of \c __type_pair
template <class _Pair>
using __type_pair_second _CCCL_NODEBUG_ALIAS = typename _Pair::__second;

//! \see __type_switch
template <class _Value>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_default
{
  template <class>
  using __rebind _CCCL_NODEBUG_ALIAS = __type_default;

  using type _CCCL_NODEBUG_ALIAS = _Value;
};

#  if defined(_CCCL_DOXYGEN_INVOKED)

//! \see __type_switch
template <auto _Label, class _Value>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_case
{
  template <class _OtherInt>
  using __rebind _CCCL_NODEBUG_ALIAS = __type_case<static_cast<_OtherInt>(_Label), _Value>;

  using type = _Value;
};

#  else // ^^^ DOXYGEN ^^^ / vvv !DOXYGEN vvv

template <class _Label, class _Value>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_case_
{
  template <class _OtherInt>
  using __rebind _CCCL_NODEBUG_ALIAS = __type_case_<integral_constant<_OtherInt, _Label::value>, _Value>;

  using type = _Value;
};

template <auto _Label, class _Value>
using __type_case _CCCL_NODEBUG_ALIAS = __type_case_<integral_constant<decltype(_Label), _Label>, _Value>;

#  endif // !DOXYGEN

namespace __detail
{
template <auto _Label, class _Value>
_CCCL_API inline auto __type_switch_fn(__type_case<_Label, _Value>*, int) -> __type_case<_Label, _Value>;

template <auto _Label, class _Value>
_CCCL_API inline auto __type_switch_fn(__type_default<_Value>*, long) -> __type_default<_Value>;
} // namespace __detail

//! \see __type_switch
template <class _Type, class... _Cases>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_DECLSPEC_EMPTY_BASES __type_switch_fn : _Cases::template __rebind<_Type>...
{
  template <class _Label>
  using __call _CCCL_NODEBUG_ALIAS =
    __type<decltype(__detail::__type_switch_fn<_Label::value>(static_cast<__type_switch_fn*>(nullptr), 0))>;
};

//! \brief Given an integral constant \c _Label and a pack of "cases"
//! consisting of one or more specializations of \c __type_case and zero or
//! one specializations of \c __type_default, `__type_switch<_Label, _Cases...>`
//! returns the value associated with the first case whose label matches the
//! given label. If no such case exists, the value associated with the default
//! case is returned. If no default case exists, the type is ill-formed.
//!
//! \p Example:
//! \code
//! using result = __type_switch<2,
//!                              __type_case<1, char>,
//!                              __type_case<2, double>,
//!                              __type_default<float>>;
//! static_assert(is_same_v<result, double>);
//! \endcode
template <auto _Label, class... _Cases>
using __type_switch _CCCL_NODEBUG_ALIAS =
  __type_call<__type_switch_fn<decltype(_Label), _Cases...>, integral_constant<decltype(_Label), _Label>>;

namespace __detail
{
#  if _CCCL_COMPILER(MSVC, <, 19, 38)
// A workaround for https://developercommunity.visualstudio.com/t/fatal-error-C1001:-Internal-compiler-err/10405847
struct __type_concat_fn
{
  template <class... _Lists>
  struct __trait
  {};

  template <class... _Ts, class... _Us, class... _Lists>
  struct __trait<__type_list<_Ts...>, __type_list<_Us...>, _Lists...> : __trait<__type_list<_Ts..., _Us...>, _Lists...>
  {};

  template <class... _Ts>
  struct __trait<__type_list<_Ts...>>
  {
    using type = __type_list<_Ts...>;
  };

  template <>
  struct __trait<>
  {
    using type = __type_list<>;
  };

  template <class... _Lists>
  using __call _CCCL_NODEBUG_ALIAS = __type<__trait<_Lists...>>;
};
#  else // ^^^ _CCCL_COMPILER(MSVC, <, 19, 38) ^^^ / vvv _CCCL_COMPILER(MSVC, >=, 19, 38) vvv
template <size_t _Count>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_concat_fn
{
  using __next_t _CCCL_NODEBUG_ALIAS = __type_maybe_concat_fn<(_Count < 8 ? 0 : _Count - 8)>;

  template <class... _Ts,
            class... _As,
            class... _Bs,
            class... _Cs,
            class... _Ds,
            class... _Es,
            class... _Fs,
            class... _Gs,
            class... _Hs,
            class... _Tail>
  _CCCL_API inline static auto __fn(
    __type_list_ptr<_Ts...>, // state
    __type_list_ptr<_As...>, // 1
    __type_list_ptr<_Bs...>, // 2
    __type_list_ptr<_Cs...>, // 3
    __type_list_ptr<_Ds...>, // 4
    __type_list_ptr<_Es...>, // 5
    __type_list_ptr<_Fs...>, // 6
    __type_list_ptr<_Gs...>, // 7
    __type_list_ptr<_Hs...>, // 8
    _Tail*... __tail) // rest
    -> decltype(__next_t::__fn(
      __type_list_ptr<_Ts..., _As..., _Bs..., _Cs..., _Ds..., _Es..., _Fs..., _Gs..., _Hs...>{nullptr},
      __tail...,
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr}));
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_concat_fn<0>
{
  template <class... _Ts>
  _CCCL_API inline static auto __fn(__type_list_ptr<_Ts...>, ...) -> __type_list<_Ts...>;
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_concat_fn
{
  template <class... _Lists>
  using __call _CCCL_NODEBUG_ALIAS = decltype(__type_maybe_concat_fn<sizeof...(_Lists)>::__fn(
    __type_list_ptr<>{nullptr},
    static_cast<_Lists*>(nullptr)...,
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr}));
};
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 38)
} // namespace __detail

//! \brief Concatenate a list of type lists into a single type list.
//!
//! When passed no type lists, \c __type_concat returns an empty type list.
template <class... _Lists>
using __type_concat = __type_call<__detail::__type_concat_fn, _Lists...>;

//! \brief Given a list of type lists, concatenate all the lists into one.
//!
//! When passed an empty type list, \c __type_flatten returns an empty type list.
template <class _ListOfLists>
using __type_flatten = __type_call1<_ListOfLists, __type_quote<__type_concat>>;

namespace __detail
{
template <bool _IsEmpty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn;

template <bool _Found>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_find_if_found
{
  template <class _Fn, class _Head, class... _Tail>
  using __call _CCCL_NODEBUG_ALIAS = __type_list<_Head, _Tail...>;
};

template <>
struct __type_find_if_found<false>
{
  template <class _Fn, class _Head, class... _Tail>
  using __call _CCCL_NODEBUG_ALIAS = __type_call<__type_maybe_find_if_fn<sizeof...(_Tail) == 0>, _Fn, _Tail...>;
};

template <bool _IsEmpty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn // Type list is not empty
{
  template <class _Fn, class _Head, class... _Tail>
  using __call _CCCL_NODEBUG_ALIAS =
    __type_call<__type_find_if_found<__type_call1<_Fn, _Head>::value>, _Fn, _Head, _Tail...>;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn<true> // Type list is empty
{
  template <class, class... _None>
  using __call _CCCL_NODEBUG_ALIAS = __type_list<>;
};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_find_if_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_call<__type_maybe_find_if_fn<sizeof...(_Ts) == 0>, _Fn, _Ts...>;
};

template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_same_as
{
  template <class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = bool_constant<_CCCL_TRAIT(is_same, _Ty, _Uy)>;
};
} // namespace __detail

//! \brief Given a type list and a predicate, find the first type in the list
//! that satisfies the predicate. It returns a type list containing the first
//! type that satisfies the predicate and all the types after it.
//!
//! If no type in the list satisfies the predicate, \c __type_find_if returns an
//! empty type list.
template <class _List, class _Fn>
using __type_find_if = __type_call1<_List, __detail::__type_find_if_fn<_Fn>>;

//! \brief Given a type list and type, find the first occurrence of the type in
//! the list. It returns a type list containing the type and all the types after
//! it.
//!
//! If the type is not in the list, \c __type_find returns an empty type list.
template <class _List, class _Ty>
using __type_find = __type_find_if<_List, __detail::__type_same_as<_Ty>>;

namespace __detail
{
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_transform_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_list<__type_call1<_Fn, _Ts>...>;
};
} // namespace __detail

//! \brief Given a type list and a unary meta-callable, apply the meta-callable
//! to each type in the list. It returns a new type list containing the results.
template <class _List, class _Fn>
using __type_transform = __type_call1<_List, __detail::__type_transform_fn<_Fn>>;

namespace __detail
{
template <size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_right_fn;

template <size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_fn;

#  define _M0(_N) , class _CCCL_PP_CAT(_, _N)
#  define _M1(_N) typename _Fn::template __call <
#  define _M2(_N) , _CCCL_PP_CAT(_, _N)
#  define _M3(_N) _M2(_N) >
#  define _M4(_N) _M2(_CCCL_PP_DEC(_N)) >

#  define _LIBCUDACXX_TYPE_LIST_FOLD_RIGHT(_N)                                                     \
    template <>                                                                                    \
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_right_fn<_N>                                  \
    {                                                                                              \
      template <class _Fn, class _State _CCCL_PP_REPEAT(_N, _M0)>                                  \
      using __call _CCCL_NODEBUG_ALIAS = _CCCL_PP_REPEAT(_N, _M1) _State _CCCL_PP_REPEAT(_N, _M3); \
    };

_CCCL_PP_REPEAT_REVERSE(_CCCL_PP_INC(_CCCL_META_UNROLL_LIMIT), _LIBCUDACXX_TYPE_LIST_FOLD_RIGHT)

template <size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_right_fn
{
  template <class _Fn, class _State _CCCL_PP_REPEAT(_CCCL_META_UNROLL_LIMIT, _M0), class... _Rest>
  using __call _CCCL_NODEBUG_ALIAS = __type_call_indirect<
    __type_fold_right_fn<_Np - _CCCL_META_UNROLL_LIMIT>,
    _Fn,
    __type_call<__type_fold_right_fn<_CCCL_META_UNROLL_LIMIT>, _Fn, _State _CCCL_PP_REPEAT(_CCCL_META_UNROLL_LIMIT, _M2)>,
    _Rest...>;
};

template <class _Init, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_right_select_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_call_indirect<__type_fold_right_fn<sizeof...(_Ts)>, _Fn, _Init, _Ts...>;
};

#  define _LIBCUDACXX_TYPE_FOLD_LEFT(_N)                                                                             \
    template <>                                                                                                      \
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_fn<_N>                                                     \
    {                                                                                                                \
      template <class _Fn, class _State _CCCL_PP_REPEAT(_N, _M0)>                                                    \
      using __call _CCCL_NODEBUG_ALIAS = _CCCL_PP_REPEAT(_N, _M1) _State _CCCL_PP_REPEAT(_N, _M4, _N, _CCCL_PP_DEC); \
    };

_CCCL_PP_REPEAT_REVERSE(_CCCL_PP_INC(_CCCL_META_UNROLL_LIMIT), _LIBCUDACXX_TYPE_FOLD_LEFT)

template <size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_fn
{
  template <class _Fn, class _State _CCCL_PP_REPEAT(_CCCL_META_UNROLL_LIMIT, _M0), class... _Rest>
  using __call _CCCL_NODEBUG_ALIAS =
    __type_call<__type_fold_left_fn<_CCCL_META_UNROLL_LIMIT>,
                _Fn,
                __type_call_indirect<__type_fold_left_fn<_Np - _CCCL_META_UNROLL_LIMIT>, _Fn, _State, _Rest...> //
                  _CCCL_PP_REPEAT(_CCCL_META_UNROLL_LIMIT, _M2, 0, _CCCL_PP_INC)>;
};

template <class _Init, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_select_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_call_indirect<__type_fold_left_fn<sizeof...(_Ts)>, _Fn, _Init, _Ts...>;
};

#  undef _LIBCUDACXX_TYPE_FOLD_LEFT
#  undef _LIBCUDACXX_TYPE_FOLD_RIGHT

#  undef _M4
#  undef _M3
#  undef _M2
#  undef _M1
#  undef _M0
} // namespace __detail

//! \brief Fold a type list from the right with a binary meta-callable and an
//! initial state.
template <class _List, class _Init, class _Fn>
using __type_fold_right = __type_call1<_List, __detail::__type_fold_right_select_fn<_Init, _Fn>>;

//! \brief Fold a type list from the left with a binary meta-callable and an
//! initial state.
template <class _List, class _Init, class _Fn>
using __type_fold_left = __type_call1<_List, __detail::__type_fold_left_select_fn<_Init, _Fn>>;

namespace __detail
{
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_remove_fn
{
  template <class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = _If<_CCCL_TRAIT(is_same, _Ty, _Uy), __type_list<>, __type_list<_Uy>>;
};
} // namespace __detail

//! \brief Remove all occurrences of a type from a type list
template <class _List, class _Ty>
using __type_remove = __type_flatten<__type_transform<_List, __detail::__type_remove_fn<_Ty>>>;

namespace __detail
{
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_remove_if_fn
{
  template <class _Uy>
  using __call _CCCL_NODEBUG_ALIAS = _If<__type_call1<_Fn, _Uy>::value, __type_list<>, __type_list<_Uy>>;
};
} // namespace __detail

//! \brief Remove all types satisfying a unary predicate from a type list
template <class _List, class _Fn>
using __type_remove_if = __type_flatten<__type_transform<_List, __detail::__type_remove_if_fn<_Fn>>>;

//! \brief Remove all types not satisfying a unary predicate from a type list
template <class _List, class _Fn>
using __type_copy_if = __type_flatten<__type_transform<_List, __detail::__type_remove_if_fn<__type_negate1<_Fn>>>>;

//! \brief Remove all duplicate types from a type list
template <class _List>
using __type_unique = __as_type_list<__type_call<_List, __type_quote<__make_type_set>>>;

namespace __detail
{
// _State is the list of type lists being built by the __type_fold_left
template <class _State, class _List>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_cartesian_product_fn
{
  template <class _Ty>
  struct __lambda0
  {
    template <class _List2>
    using __lambda1 _CCCL_NODEBUG_ALIAS = __type_list<__type_push_front<_List2, _Ty>>;

    using type _CCCL_NODEBUG_ALIAS = __type_flatten<__type_transform<_State, __type_quote1<__lambda1>>>;
  };

  using type _CCCL_NODEBUG_ALIAS = __type_flatten<__type_transform<_List, __type_quote_trait1<__lambda0>>>;
};
} // namespace __detail
/// \endcond

//! \brief Given a list of lists \c _Lists, return a new list of lists that is
//! the cartesian product.
//!
//! \par Complexity
//! `O(N * M)`, where `N` is the size of the outer list, and
//! `M` is the size of the inner lists.
template <class... _Lists>
using __type_cartesian_product =
  __type_fold_left<__type_list<_Lists...>,
                   __type_list<__type_list<>>,
                   __type_quote_trait2<__detail::__type_cartesian_product_fn>>;

// A unary meta-callable for converting a type to its size in bytes
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_sizeof
{
  template <class _Ty>
  using __call _CCCL_NODEBUG_ALIAS = integral_constant<size_t, sizeof(_Ty)>;
};

//! \brief A list of compile-time values, and a meta-callable that accepts a
//! meta-callable and evaluates it with the values, each value wrapped in an
//! integral constant wrapper.
template <class _Ty, _Ty... _Values>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_value_list : __type_list<integral_constant<_Ty, _Values>...>
{
  using __type _CCCL_NODEBUG_ALIAS = _Ty;
};

namespace __detail
{
template <class _Ty, _Ty _Start, _Ty _Stride, _Ty... _Is>
_CCCL_API inline auto __type_iota_fn(integer_sequence<_Ty, _Is...>*)
  -> __type_value_list<_Ty, _Ty(_Start + (_Is * _Stride))...>;
} // namespace __detail

//! \brief Return an \c __type_value_list of size \c _Size starting at \c _Start
//! and incrementing by \c _Stride.
template <class _Ty, _Ty _Start, _Ty _Size, _Ty _Stride = _Ty(1)>
using __type_iota =
  decltype(__detail::__type_iota_fn<_Ty, _Start, _Stride>(static_cast<make_integer_sequence<_Ty, _Size>*>(nullptr)));

#endif // _CCCL_DOXYGEN_INVOKED

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H
