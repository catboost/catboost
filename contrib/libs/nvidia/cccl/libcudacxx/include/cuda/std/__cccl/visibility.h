//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_VISIBILITY_H
#define __CCCL_VISIBILITY_H

#ifndef _CUDA__CCCL_CONFIG
#  error "<__cccl/visibility.h> should only be included in from <cuda/__cccl_config>"
#endif // _CUDA__CCCL_CONFIG

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

// We want to ensure that all warning emitting from this header are suppressed
#if defined(_CCCL_FORCE_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/attributes.h>
#include <cuda/std/__cccl/execution_space.h>

// For unknown reasons, nvc++ need to selectively disable this warning
// We do not want to use our usual macro because that would have push / pop semantics
#if _CCCL_COMPILER(NVHPC)
#  pragma nv_diag_suppress 1407
#endif // _CCCL_COMPILER(NVHPC)

// Enable us to hide kernels
#if _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_VISIBILITY_HIDDEN
#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ / vvv _CCCL_COMPILER(NVRTC) vvv
#  define _CCCL_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_COMPILER(MSVC)
#  define _CCCL_VISIBILITY_DEFAULT __declspec(dllimport)
#elif _CCCL_COMPILER(NVRTC) // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv _CCCL_COMPILER(NVRTC) vvv
#  define _CCCL_VISIBILITY_DEFAULT
#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ / vvv !_CCCL_COMPILER(NVRTC) vvv
#  define _CCCL_VISIBILITY_DEFAULT __attribute__((__visibility__("default")))
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_TYPE_VISIBILITY_DEFAULT
#elif _CCCL_HAS_ATTRIBUTE(__type_visibility__)
#  define _CCCL_TYPE_VISIBILITY_DEFAULT __attribute__((__type_visibility__("default")))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(__type_visibility__) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(__type_visibility__) vvv
#  define _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_VISIBILITY_DEFAULT
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_COMPILER(MSVC)
#  define _CCCL_FORCEINLINE __forceinline
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv _CCCL_COMPILER(MSVC) vvv
#  define _CCCL_FORCEINLINE __inline__ __attribute__((__always_inline__))
#endif // !_CCCL_COMPILER(MSVC)

#if _CCCL_HAS_ATTRIBUTE(__exclude_from_explicit_instantiation__)
#  define _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((__exclude_from_explicit_instantiation__))
#else // ^^^ exclude_from_explicit_instantiation ^^^ / vvv !exclude_from_explicit_instantiation vvv
// NVCC complains mightily about being unable to inline functions if we use _CCCL_FORCEINLINE here
#  define _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#endif // !exclude_from_explicit_instantiation

#if _CCCL_COMPILER(NVHPC) // NVHPC has issues with visibility attributes on symbols with internal linkage
#  define _CCCL_HIDE_FROM_ABI inline
#else // ^^^ _CCCL_COMPILER(NVHPC) ^^^ / vvv !_CCCL_COMPILER(NVHPC) vvv
#  define _CCCL_HIDE_FROM_ABI _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION inline
#endif // !_CCCL_COMPILER(NVHPC)

#if !defined(CCCL_DETAIL_KERNEL_ATTRIBUTES)
#  define CCCL_DETAIL_KERNEL_ATTRIBUTES __global__ _CCCL_VISIBILITY_HIDDEN
#endif // !CCCL_DETAIL_KERNEL_ATTRIBUTES

// _CCCL_HIDE_FROM_ABI and _CCCL_FORCEINLINE cannot be used together because they both try
// to add `inline` to the function declaration. The following macros slice the function
// attributes differently to avoid this problem:
// - `_CCCL_API` declares the function host/device and hides the symbol from the ABI
// - `_CCCL_TRIVIAL_API` does the same while also inlining and hiding the function from
//   debuggers
#if _CCCL_COMPILER(NVHPC) // NVHPC has issues with visibility attributes on symbols with internal linkage
#  define _CCCL_API        _CCCL_HOST_DEVICE
#  define _CCCL_HOST_API   _CCCL_HOST
#  define _CCCL_DEVICE_API _CCCL_DEVICE
#else // ^^^ _CCCL_COMPILER(NVHPC) ^^^ / vvv !_CCCL_COMPILER(NVHPC) vvv
#  define _CCCL_API        _CCCL_HOST_DEVICE _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#  define _CCCL_HOST_API   _CCCL_HOST _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#  define _CCCL_DEVICE_API _CCCL_DEVICE _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#endif // !_CCCL_COMPILER(NVHPC)

// _CCCL_TRIVIAL_API force-inlines a function, marks its visibility as hidden, and causes
// debuggers to skip it. This is useful for trivial internal functions that do dispatching
// or other plumbing work. It is particularly useful in the definition of customization
// point objects.
#define _CCCL_TRIVIAL_API        _CCCL_API _CCCL_ARTIFICIAL _CCCL_NODEBUG inline
#define _CCCL_TRIVIAL_HOST_API   _CCCL_HOST_API _CCCL_ARTIFICIAL _CCCL_NODEBUG inline
#define _CCCL_TRIVIAL_DEVICE_API _CCCL_DEVICE_API _CCCL_ARTIFICIAL _CCCL_NODEBUG inline

// Some functions have their addresses appear in public types (e.g., in
// `cuda::__overrides_for` specializations). If the function is declared
// `__attribute__((visibility("hidden")))`, and if the address appears, say, in the type
// of a member of a class that is declared `__attribute__((visibility("default")))`, GCC
// complains bitterly. So we avoid declaring those functions `hidden`. Instead of the
// typical `_CCCL_API` macro, we use `_CCCL_PUBLIC_API` for those functions.
#if _CCCL_COMPILER(MSVC)
#  define _CCCL_PUBLIC_API        _CCCL_HOST_DEVICE
#  define _CCCL_PUBLIC_HOST_API   _CCCL_HOST
#  define _CCCL_PUBLIC_DEVICE_API _CCCL_DEVICE
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_PUBLIC_API        _CCCL_HOST_DEVICE _CCCL_VISIBILITY_DEFAULT
#  define _CCCL_PUBLIC_HOST_API   _CCCL_HOST _CCCL_VISIBILITY_DEFAULT
#  define _CCCL_PUBLIC_DEVICE_API _CCCL_DEVICE _CCCL_VISIBILITY_DEFAULT
#endif // !_CCCL_COMPILER(MSVC)

//! _LIBCUDACXX_HIDE_FROM_ABI is for backwards compatibility for external projects.
//! _CCCL_API and its variants are the preferred way to declare functions
//! that should be hidden from the ABI.
//! Defined here to suppress any warnings from the definition
#define _LIBCUDACXX_HIDE_FROM_ABI _CCCL_API inline

#endif // __CCCL_VISIBILITY_H
