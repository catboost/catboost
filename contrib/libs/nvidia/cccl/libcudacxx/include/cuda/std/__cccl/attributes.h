//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_ATTRIBUTES_H
#define __CCCL_ATTRIBUTES_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/diagnostic.h>
#include <cuda/std/__cccl/dialect.h>

#include <cuda/std/__cccl/prologue.h>

#ifdef __has_attribute
#  define _CCCL_HAS_ATTRIBUTE(__x) __has_attribute(__x)
#else // ^^^ __has_attribute ^^^ / vvv !__has_attribute vvv
#  define _CCCL_HAS_ATTRIBUTE(__x) 0
#endif // !__has_attribute

#ifdef __has_cpp_attribute
#  define _CCCL_HAS_CPP_ATTRIBUTE(__x) __has_cpp_attribute(__x)
#else // ^^^ __has_cpp_attribute ^^^ / vvv !__has_cpp_attribute vvv
#  define _CCCL_HAS_CPP_ATTRIBUTE(__x) 0
#endif // !__has_cpp_attribute

#ifdef __has_declspec_attribute
#  define _CCCL_HAS_DECLSPEC_ATTRIBUTE(__x) __has_declspec_attribute(__x)
#else // ^^^ __has_declspec_attribute ^^^ / vvv !__has_declspec_attribute vvv
#  define _CCCL_HAS_DECLSPEC_ATTRIBUTE(__x) 0
#endif // !__has_declspec_attribute

// MSVC needs extra help with empty base classes
#if _CCCL_COMPILER(MSVC) || _CCCL_HAS_DECLSPEC_ATTRIBUTE(empty_bases)
#  define _CCCL_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_DECLSPEC_EMPTY_BASES
#endif // !_CCCL_COMPILER(MSVC)

#if _CCCL_HAS_ATTRIBUTE(__nodebug__)
#  define _CCCL_NODEBUG __attribute__((__nodebug__))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(__nodebug__) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(__nodebug__) vvv
#  define _CCCL_NODEBUG
#endif // !_CCCL_HAS_ATTRIBUTE(__nodebug__)

// Debuggers do not step into functions marked with __attribute__((__artificial__)). This
// is useful for small wrapper functions that just dispatch to other functions and that
// are inlined into the caller.
#if _CCCL_HAS_ATTRIBUTE(__artificial__) && !_CCCL_HAS_CUDA_COMPILER()
#  define _CCCL_ARTIFICIAL __attribute__((__artificial__))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(__artificial__) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(__artificial__) vvv
#  define _CCCL_ARTIFICIAL
#endif // !_CCCL_HAS_ATTRIBUTE(__artificial__)

// The nodebug attribute flattens aliases down to the actual type rather typename meow<T>::type
#if _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_NODEBUG_ALIAS _CCCL_NODEBUG
#else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG) vvv
#  define _CCCL_NODEBUG_ALIAS
#endif // !_CCCL_CUDA_COMPILER(CLANG)

// _CCCL_ASSUME

#if _CCCL_HAS_CPP_ATTRIBUTE(assume)
#  define _CCCL_ASSUME(...) [[assume(__VA_ARGS__)]]
#elif _CCCL_CUDA_COMPILER(NVCC) && _CCCL_COMPILER(NVHPC)
#  define _CCCL_ASSUME(...) \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__builtin_assume(__VA_ARGS__);), (_CCCL_BUILTIN_ASSUME(__VA_ARGS__);))
#else
#  define _CCCL_ASSUME(...) _CCCL_BUILTIN_ASSUME(__VA_ARGS__)
#endif

// _CCCL_CONST

#if _CCCL_HAS_CPP_ATTRIBUTE(__gnu__::__const__)
#  define _CCCL_CONST [[__gnu__::__const__]]
#else // ^^^ has gnu::const ^^^ / vvv no gnu::const vvv
#  define _CCCL_CONST _CCCL_PURE
#endif // ^^^ no gnu::const ^^^

// _CCCL_DIAGNOSE_IF

#if _CCCL_HAS_ATTRIBUTE(__diagnose_if__)
#  define _CCCL_DIAGNOSE_IF(_COND, _MSG, _TYPE) __attribute__((__diagnose_if__(_COND, _MSG, _TYPE)))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(diagnose_if) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(diagnose_if) vvv
#  define _CCCL_DIAGNOSE_IF(_COND, _MSG, _TYPE)
#endif // !_CCCL_HAS_ATTRIBUTE(diagnose_if)

// _CCCL_INTRINSIC

// MSVC provides a way to mark functions as intrinsic provided the function's body consists of a single
// return statement of a cast expression (e.g., move(x) or forward<T>(u)).
#if _CCCL_COMPILER(MSVC) && _CCCL_HAS_CPP_ATTRIBUTE(msvc::intrinsic)
#  define _CCCL_INTRINSIC [[msvc::intrinsic]]
#else
#  define _CCCL_INTRINSIC
#endif

// _CCCL_PURE

#if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 5)
#  define _CCCL_PURE __nv_pure__
#elif _CCCL_HAS_CPP_ATTRIBUTE(__gnu__::__pure__)
#  define _CCCL_PURE [[__gnu__::__pure__]]
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_PURE __declspec(noalias)
#else
#  define _CCCL_PURE
#endif

// _CCCL_NO_CFI

#if !_CCCL_COMPILER(GCC)
#  define _CCCL_NO_CFI _CCCL_NO_SANITIZE("cfi")
#else
#  define _CCCL_NO_CFI
#endif

// _CCCL_NO_SANITIZE

#if _CCCL_HAS_ATTRIBUTE(__no_sanitize__)
#  define _CCCL_NO_SANITIZE(_STR) __attribute__((__no_sanitize__(_STR)))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(no_sanitize) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(no_sanitize) vvv
#  define _CCCL_NO_SANITIZE(_STR)
#endif // !_CCCL_HAS_ATTRIBUTE(no_sanitize)

// _CCCL_NO_SPECIALIZATIONS

#if _CCCL_HAS_CPP_ATTRIBUTE(clang::__no_specializations__)
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)   [[clang::__no_specializations__(_MSG)]]
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 1
#elif _CCCL_HAS_CPP_ATTRIBUTE(msvc::no_specializations)
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)   [[msvc::no_specializations(_MSG)]]
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 1
#else // ^^^ has attribute no_specializations ^^^ / vvv hasn't attribute no_specializations vvv
#  define _CCCL_NO_SPECIALIZATIONS_BECAUSE(_MSG)
#  define _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS() 0
#endif // ^^^ hasn't attribute no_specializations ^^^

#define _CCCL_NO_SPECIALIZATIONS \
  _CCCL_NO_SPECIALIZATIONS_BECAUSE("Users are not allowed to specialize this cccl entity")

// _CCCL_NO_UNIQUE_ADDRESS

#if _CCCL_COMPILER(MSVC) || _CCCL_HAS_CPP_ATTRIBUTE(no_unique_address) < 201803L
// MSVC implementation has lead to multiple issues with silent runtime corruption when passing data into kernels
#  define _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() 0
#  define _CCCL_NO_UNIQUE_ADDRESS
#elif _CCCL_HAS_CPP_ATTRIBUTE(no_unique_address)
#  define _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() 1
#  define _CCCL_NO_UNIQUE_ADDRESS                 [[no_unique_address]]
#else
#  define _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() 0
#  define _CCCL_NO_UNIQUE_ADDRESS
#endif

// Passing objects with nested [[no_unique_address]] to kernels leads to data corruption.
// This is caused by cudafe++ not honoring [[no_unique_address]] when compiling for C++17
// with clang as the host compiler. See nvbug 5265027 for more details.
#if _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() && _CCCL_COMPILER(CLANG) && _CCCL_STD_VER < 2020 \
  && _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  undef _CCCL_NO_UNIQUE_ADDRESS
#  define _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() 0
#  define _CCCL_NO_UNIQUE_ADDRESS
#endif // _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() && _CCCL_COMPILER(CLANG)

// _CCCL_PREFERRED_NAME

#if _CCCL_HAS_ATTRIBUTE(__preferred_name__)
#  define _CCCL_PREFERRED_NAME(x) __attribute__((__preferred_name__(x)))
#else
#  define _CCCL_PREFERRED_NAME(x)
#endif

#if _CCCL_HAS_ATTRIBUTE(__require_constant_initialization__)
#  define _CCCL_REQUIRE_CONSTANT_INITIALIZATION __attribute__((__require_constant_initialization__))
#else
#  define _CCCL_REQUIRE_CONSTANT_INITIALIZATION
#endif

// _CCCL_RESTRICT

#if _CCCL_COMPILER(MSVC) // vvv _CCCL_COMPILER(MSVC) vvv
#  define _CCCL_RESTRICT __restrict
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_RESTRICT __restrict__
#endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

#include <cuda/std/__cccl/epilogue.h>

#endif // __CCCL_ATTRIBUTES_H
