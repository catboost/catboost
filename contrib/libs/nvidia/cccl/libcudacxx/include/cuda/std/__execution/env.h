//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___EXECUTION_ENV_H
#define __CUDA_STD___EXECUTION_ENV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__functional/reference_wrapper.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/pod_tuple.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <functional> // IWYU pragma: keep for ::std::reference_wrapper
#endif // !_CCCL_COMPILER(NVRTC)

//! @file env.h
//! @brief Provides utilities for querying and managing environments, an unordered
//! collection of key/value pairs.
//!
//! This header defines templates and structures for querying properties of environments
//! and managing them in a composable way. It includes support for querying properties of
//! individual environments, combining multiple environments, and handling reference
//! wrappers.
//!
//! An environment is a collection of key/value pairs where the key is an empty tag type.
//! Environment objects have a `query` member function that accepts a key and returns the
//! associated value.
//!
//! @details
//! The key components in this file include:
//! - `__query_result_t`: A type alias for the result of querying a query from an
//!   environment.
//! - `__queryable_with`: A concept to check if a query can be property from an
//!   environment.
//! - `__nothrow_queryable_with`: A concept to check if a property can be queried without
//!   potentially throwing.
//! - `prop`: Builds an environment from a key/value pair.
//! - `env`: Builds an environment from a variadic number of environments. It allows
//!   querying properties from the first environment that satisfies a given query type.
//! - `get_env_t`: A callable object for retrieving the environment associated with an
//!   object.
//! - `env_of_t`: A type alias for the environment type of an object.
//!
//! @namespace cuda::std::execution
//! The primary namespace for all components defined in this file.
//!
//! @concept __queryable_with
//! Checks if a query `_Query` can be queried from an environment `_Env`.
//! @tparam _Env The type of the environment.
//! @tparam _Query The type of the property to be queried.
//!
//! @concept __nothrow_queryable_with Checks if a query `_Query` can be queried from an
//! environment `_Env` without potentially throwing.
//! @tparam _Env The type of the environment.
//! @tparam _Query The type of the property to be queried.
//!
//! @struct prop
//! @tparam _Query The type of the property to be queried.
//! @tparam _Value The type of the value associated with the query.
//! A simple environment with a single value associated with a query.
//!
//! @struct env
//! @tparam _Envs The types of the sub-environments contained within this environment.
//! Represents a composable execution environment that can query properties from its
//! sub-environments.
//!
//! @struct get_env_t
//! A callable object for retrieving the environment associated with an object.
//!

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_EXECUTION

namespace __detail
{
template <class _Env, class _Query>
_CCCL_API auto __query_result_() -> decltype(declval<_Env>().query(_Query()));

#if _CCCL_HAS_EXCEPTIONS()
template <class _Env, class _Query>
using __nothrow_queryable_with_t _CCCL_NODEBUG_ALIAS = enable_if_t<noexcept(declval<_Env>().query(_Query{}))>;
#endif // _CCCL_HAS_EXCEPTIONS()

template <class _Ty>
extern _Ty __unwrap_ref;

#if !_CCCL_COMPILER(NVRTC)
template <class _Ty>
extern _Ty& __unwrap_ref<::std::reference_wrapper<_Ty>>;
#endif // !_CCCL_COMPILER(NVRTC)

template <class _Ty>
extern _Ty& __unwrap_ref<reference_wrapper<_Ty>>;

inline constexpr size_t __npos = static_cast<size_t>(-1);

[[nodiscard]] _CCCL_API constexpr auto __find_pos(bool const* const __begin, bool const* const __end) noexcept -> size_t
{
  for (bool const* __where = __begin; __where != __end; ++__where)
  {
    if (*__where)
    {
      return static_cast<size_t>(__where - __begin);
    }
  }
  return __npos;
}
} // namespace __detail

template <class _Env, class _Query>
using __query_result_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__query_result_<_Env, _Query>());

template <class _Env, class _Query>
_CCCL_CONCEPT __queryable_with = _IsValidExpansion<__query_result_t, _Env, _Query>::value;

#if _CCCL_HAS_EXCEPTIONS()

template <class _Env, class _Query>
_CCCL_CONCEPT __nothrow_queryable_with = _IsValidExpansion<__detail::__nothrow_queryable_with_t, _Env, _Query>::value;

#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv

template <class _Env, class _Query>
_CCCL_CONCEPT __nothrow_queryable_with = true;

#endif // !_CCCL_HAS_EXCEPTIONS()

template <class _Ty>
using __unwrap_reference_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__unwrap_ref<_Ty>);

template <class _Query, class _DefaultFn = void>
struct __basic_query : __basic_query<_Query>
{
  static constexpr bool __is_nothrow = noexcept(_DefaultFn{}());

  using __basic_query<_Query>::operator();

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(__ignore_t) const noexcept(__is_nothrow)
    -> decltype(_DefaultFn{}())
  {
    static_assert(is_base_of_v<__basic_query, _Query>, "_Query must be derived from __basic_query<_Query>");
    return _DefaultFn{}();
  }
};

template <class _Query>
struct __basic_query<_Query, void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Env&& __env) const
    noexcept(__nothrow_queryable_with<_Env, _Query>) -> __query_result_t<_Env, _Query>
  {
    static_assert(is_base_of_v<__basic_query, _Query>, "_Query must be derived from __basic_query<_Query>");
    return __env.query(_Query{});
  }
};

// nvvm/bin/cicc segfaults when `prop` uses [[no_unique_address]]
#if (_CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() || defined(_CCCL_DOXYGEN_INVOKED)) && !_CCCL_CUDA_COMPILER(NVCC) \
  && !_CCCL_CUDA_COMPILER(NVRTC)

//! @brief A template structure representing a query with a query and a value.
//!
//! @tparam _Query The type of the query associated with the query.
//! @tparam _Value The type of the value associated with the query.
//!
//! This structure provides a mechanism to associate a query with a value
//! and allows querying the value using the query type.
//!
//! Members:
//! - _Query __query: The query object.
//! - _Value __value: The value associated with the query.
//!
//! Member Functions:
//! - constexpr auto query(_Query) const noexcept -> const _Value&:
//!   Returns the value associated with the query.
template <class _Query, class _Value>
struct _CCCL_TYPE_VISIBILITY_DEFAULT prop
{
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(_Query) const noexcept -> const _Value&
  {
    return __value;
  }

  _CCCL_NO_UNIQUE_ADDRESS _Query __ignore;
  _CCCL_NO_UNIQUE_ADDRESS _Value __value;
};

#else // ^^^ _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() ^^^ / vvv !_CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() vvv

template <class _Query, class _Value>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_DECLSPEC_EMPTY_BASES prop : _Query
{
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(_Query) const noexcept -> const _Value&
  {
    return __value;
  }

  _Value __value;
};

#endif // !_CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS()

template <class _Query, class _Value>
_CCCL_HOST_DEVICE prop(_Query, _Value) -> prop<_Query, _Value>;

//! @brief A variadic template structure representing an environment.
//!
//! This structure encapsulates a tuple of environments and provides functionality to query
//! the first environment that satisfies a given query type.
//!
//! @tparam _Envs... Variadic template parameter pack representing the types of
//! environments.
template <class... _Envs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT env
{
  //!
  //! @brief Retrieves the first environment that satisfies the given query type.
  //!
  //! This static constexpr function checks each environment in the tuple to find
  //! the first one that is queryable with the specified query type. If such an
  //! environment is found, it is returned.
  //!
  //! @tparam _Query The type of the query to be performed.
  //! @param __self A constant reference to the current `env` instance.
  //! @return The first environment in the tuple that satisfies the query type.
  //! @note If no environment satisfies the query, the behavior is undefined.
  template <class _Query>
  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr decltype(auto) __get_1st(const env& __self) noexcept
  {
    // NOLINTNEXTLINE (modernize-avoid-c-arrays)
    constexpr bool __flags[] = {__queryable_with<_Envs, _Query>..., false};
    constexpr size_t __idx   = __detail::__find_pos(__flags, __flags + sizeof...(_Envs));
    if constexpr (__idx != __detail::__npos)
    {
      return _CUDA_VSTD::__get<__idx>(__self.__envs_);
    }
  }

  //! @brief Alias for the type of the first environment that satisfies the query type.
  //!
  //! This alias resolves to the type of the first environment in the tuple that
  //! satisfies the specified query type.
  //!
  //! @tparam _Query The type of the query to be performed.
  template <class _Query>
  using __1st_env_t _CCCL_NODEBUG_ALIAS = decltype(env::__get_1st<_Query>(declval<const env&>()));

  //! @brief Queries the first environment that satisfies the given query type.
  //!
  //! This function invokes the `query` method on the first environment in the tuple
  //! that satisfies the specified query type, passing the query object as an argument.
  //!
  //! @tparam _Query The type of the query to be performed.
  //! @param __query The query object to be passed to the environment's `query` method.
  //! @return The result of the `query` method on the first environment that satisfies the query type.
  //! @throws noexcept If the query operation is noexcept for the resolved environment and query type.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__queryable_with<__1st_env_t<_Query>, _Query>)
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable_with<__1st_env_t<_Query>, _Query>) -> __query_result_t<__1st_env_t<_Query>, _Query>
  {
    return env::__get_1st<_Query>(*this).query(__query);
  }

  __tuple<_Envs...> __envs_;
};

template <class... _Envs>
_CCCL_HOST_DEVICE env(_Envs...) -> env<__unwrap_reference_t<_Envs>...>;

#ifndef _CCCL_DOXYGEN_INVOKED

// Partial specialization for no env because NVCC segfaults trying to compile `__tuple<>`
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT env<>
{};

// Partial specialization for two environments so that the syntax `env(env0, env1)` is
// valid. That is, `env` can use CTAD with a parentesized list of arguments.
template <class _Env0, class _Env1>
struct _CCCL_TYPE_VISIBILITY_DEFAULT env<_Env0, _Env1>
{
  _Env0 __env0_;
  _Env1 __env1_;

  template <class _Query>
  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr decltype(auto) __get_1st(const env& __self) noexcept
  {
    if constexpr (__queryable_with<_Env0, _Query>)
    {
      return (__self.__env0_);
    }
    else if constexpr (__queryable_with<_Env1, _Query>)
    {
      return (__self.__env1_);
    }
  }

  template <class _Query>
  using __1st_env_t _CCCL_NODEBUG_ALIAS = decltype(env::__get_1st<_Query>(declval<const env&>()));

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__queryable_with<__1st_env_t<_Query>, _Query>)
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable_with<__1st_env_t<_Query>, _Query>) -> __query_result_t<__1st_env_t<_Query>, _Query>
  {
    return env::__get_1st<_Query>(*this).query(__query);
  }
};
#endif // _CCCL_DOXYGEN_INVOKED

//! @brief Provides utilities for extracting the execution environment from an object.
//!
//! This code defines a utility for obtaining the execution environment of a given object
//! through its `get_env` member function. It also provides a fallback for types that do
//! not have a `get_env` member function.
//!
//! @tparam _Ty The type of the object from which the environment is to be extracted.
//!
//! @details
//! - The first `operator()` is a callable operator that extracts the environment from an
//!   object of type `_Ty` by calling its `get_env` method. It ensures that the `get_env`
//!   method is `noexcept` and returns the deduced environment type.
//! - The second `operator()` is less preferred than the first. It accepts any argument and
//!   returns a default-constructed `env<>`.
//!
//! @throws None. Both `operator()` overloads are marked as `noexcept`.
struct get_env_t
{
  template <class _Ty>
  using __env_of _CCCL_NODEBUG_ALIAS = decltype(declval<_Ty>().get_env());

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Ty>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(const _Ty& __ty) const noexcept -> __env_of<const _Ty&>
  {
    static_assert(noexcept(__ty.get_env()));
    return __ty.get_env();
  }

  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(__ignore_t) const noexcept -> env<>
  {
    return {};
  }
};

_CCCL_GLOBAL_CONSTANT get_env_t get_env{};

template <class _Ty>
using env_of_t _CCCL_NODEBUG_ALIAS = decltype(get_env(declval<_Ty>()));

//////////////////////////////////////////////////////////////////////////////////////////
// forwarding_query_t
_CCCL_GLOBAL_CONSTANT struct forwarding_query_t
{
  template <class _Tag>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tag) const noexcept -> bool
  {
    if constexpr (__queryable_with<_Tag, forwarding_query_t>)
    {
      static_assert(noexcept(_Tag().query(*this)));
      return _Tag().query(*this);
    }
    return _CUDA_VSTD::derived_from<_Tag, forwarding_query_t>;
  }
} forwarding_query{};

template <class _Tag>
_CCCL_CONCEPT __forwarding_query = forwarding_query(_Tag{});

//////////////////////////////////////////////////////////////////////////////////////////
// __query_or
namespace __detail
{
// query an environment, or return a default value if the query is not supported
struct __query_or_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env, class _Query, class _Default)
  _CCCL_REQUIRES(__queryable_with<_Env, _Query>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env, _Query, _Default&&) const
    noexcept(__nothrow_queryable_with<_Env, _Query>) -> __query_result_t<_Env, _Query>
  {
    return __env.query(_Query{});
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Default>
  [[nodiscard]] _CCCL_API constexpr auto
  operator()(_CUDA_VSTD::__ignore_t, _CUDA_VSTD::__ignore_t, _Default&& __default) const
    noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Default>) -> _Default
  {
    return static_cast<_Default&&>(__default);
  }
};
} // namespace __detail

_CCCL_GLOBAL_CONSTANT __detail::__query_or_t __query_or{};

template <class _Env, class _Query, class _Default>
using __query_result_or_t _CCCL_NODEBUG_ALIAS =
  decltype(__query_or(_CUDA_VSTD::declval<_Env>(), _CUDA_VSTD::declval<_Query>(), _CUDA_VSTD::declval<_Default>()));

_LIBCUDACXX_END_NAMESPACE_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA_STD___EXECUTION_ENV_H
