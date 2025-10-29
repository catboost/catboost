// Copyright (c)      2018 NVIDIA Corporation
//                         (Bryce Adelstein Lelbach <brycelelbach@gmail.com>)
// Copyright (c) 2013-2018 Eric Niebler (`THRUST_RETURNS`, etc)
// Copyright (c) 2016-2018 Casey Carter (`THRUST_RETURNS`, etc)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/preprocessor.h>

#include <cuda/std/type_traits>
#include <cuda/std/utility>

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_FWD(x)
/// \brief Performs universal forwarding of a universal reference.
///
#define THRUST_FWD(x) ::cuda::std::forward<decltype(x)>(x)

/// \def THRUST_RETURNS(...)
/// \brief Expands to a function definition that returns the expression
///        \c __VA_ARGS__.
///
#define THRUST_RETURNS(...)       \
  noexcept(noexcept(__VA_ARGS__)) \
  {                               \
    return (__VA_ARGS__);         \
  }                               \
  /**/

/// \def THRUST_DECLTYPE_RETURNS(...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression \c __VA_ARGS__.
///
// Trailing return types seem to confuse Doxygen, and cause it to interpret
// parts of the function's body as new function signatures.
#if defined(_CCCL_DOXYGEN_INVOKED)
#  define THRUST_DECLTYPE_RETURNS(...) \
    {                                  \
      return (__VA_ARGS__);            \
    }                                  \
    /**/
#else
#  define THRUST_DECLTYPE_RETURNS(...)                     \
    noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) \
    {                                                      \
      return (__VA_ARGS__);                                \
    }                                                      \
    /**/
#endif
