//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// NO include guards here (this file is included multiple times)

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/diagnostic.h>

#if !defined(_CCCL_PROLOGUE_INCLUDED)
#  error "cccl internal error: <cuda/std/__cccl/prologue.h> must be included before <cuda/std/__cccl/epilogue.h>"
#endif
#undef _CCCL_PROLOGUE_INCLUDED

// warnings pop
_CCCL_DIAG_POP

// __declspec modifiers

#if defined(align)
#  error \
    "cccl internal error: macro `align` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_align)
#  pragma pop_macro("align")
#  undef _CCCL_POP_MACRO_align
#endif

#if defined(allocate)
#  error \
    "cccl internal error: macro `allocate` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_allocate)
#  pragma pop_macro("allocate")
#  undef _CCCL_POP_MACRO_allocate
#endif

#if defined(allocator)
#  error \
    "cccl internal error: macro `allocator` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_allocator)
#  pragma pop_macro("allocator")
#  undef _CCCL_POP_MACRO_allocator
#endif

#if defined(appdomain)
#  error \
    "cccl internal error: macro `appdomain` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_appdomain)
#  pragma pop_macro("appdomain")
#  undef _CCCL_POP_MACRO_appdomain
#endif

#if defined(code_seg)
#  error \
    "cccl internal error: macro `code_seg` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_code_seg)
#  pragma pop_macro("code_seg")
#  undef _CCCL_POP_MACRO_code_seg
#endif

#if defined(deprecated)
#  error \
    "cccl internal error: macro `deprecated` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_deprecated)
#  pragma pop_macro("deprecated")
#  undef _CCCL_POP_MACRO_deprecated
#endif

#if defined(dllimport)
#  error \
    "cccl internal error: macro `dllimport` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_dllimport)
#  pragma pop_macro("dllimport")
#  undef _CCCL_POP_MACRO_dllimport
#endif

#if defined(dllexport)
#  error \
    "cccl internal error: macro `dllexport` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_dllexport)
#  pragma pop_macro("dllexport")
#  undef _CCCL_POP_MACRO_dllexport
#endif

#if defined(empty_bases)
#  error \
    "cccl internal error: macro `empty_bases` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_empty_bases)
#  pragma pop_macro("empty_bases")
#  undef _CCCL_POP_MACRO_empty_bases
#endif

#if defined(hybrid_patchable)
#  error \
    "cccl internal error: macro `hybrid_patchable` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_hybrid_patchable)
#  pragma pop_macro("hybrid_patchable")
#  undef _CCCL_POP_MACRO_hybrid_patchable
#endif

#if defined(jitintrinsic)
#  error \
    "cccl internal error: macro `jitintrinsic` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_jitintrinsic)
#  pragma pop_macro("jitintrinsic")
#  undef _CCCL_POP_MACRO_jitintrinsic
#endif

#if defined(naked)
#  error \
    "cccl internal error: macro `naked` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_naked)
#  pragma pop_macro("naked")
#  undef _CCCL_POP_MACRO_naked
#endif

#if defined(noalias)
#  error \
    "cccl internal error: macro `noalias` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_noalias)
#  pragma pop_macro("noalias")
#  undef _CCCL_POP_MACRO_noalias
#endif

#if defined(noinline)
#  error \
    "cccl internal error: macro `noinline` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_noinline)
#  pragma pop_macro("noinline")
#  undef _CCCL_POP_MACRO_noinline
#endif

#if defined(noreturn)
#  error \
    "cccl internal error: macro `noreturn` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_noreturn)
#  pragma pop_macro("noreturn")
#  undef _CCCL_POP_MACRO_noreturn
#endif

#if defined(nothrow)
#  error \
    "cccl internal error: macro `nothrow` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_nothrow)
#  pragma pop_macro("nothrow")
#  undef _CCCL_POP_MACRO_nothrow
#endif

#if defined(novtable)
#  error \
    "cccl internal error: macro `novtable` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_novtable)
#  pragma pop_macro("novtable")
#  undef _CCCL_POP_MACRO_novtable
#endif

#if defined(no_sanitize_address)
#  error \
    "cccl internal error: macro `no_sanitize_address` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_no_sanitize_address)
#  pragma pop_macro("no_sanitize_address")
#  undef _CCCL_POP_MACRO_no_sanitize_address
#endif

#if defined(process)
#  error \
    "cccl internal error: macro `process` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_process)
#  pragma pop_macro("process")
#  undef _CCCL_POP_MACRO_process
#endif

#if defined(property)
#  error \
    "cccl internal error: macro `property` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_property)
#  pragma pop_macro("property")
#  undef _CCCL_POP_MACRO_property
#endif

#if defined(restrict)
#  error \
    "cccl internal error: macro `restrict` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_restrict)
#  pragma pop_macro("restrict")
#  undef _CCCL_POP_MACRO_restrict
#endif

#if defined(safebuffers)
#  error \
    "cccl internal error: macro `safebuffers` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_safebuffers)
#  pragma pop_macro("safebuffers")
#  undef _CCCL_POP_MACRO_safebuffers
#endif

#if defined(selectany)
#  error \
    "cccl internal error: macro `selectany` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_selectany)
#  pragma pop_macro("selectany")
#  undef _CCCL_POP_MACRO_selectany
#endif

#if defined(spectre)
#  error \
    "cccl internal error: macro `spectre` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_spectre)
#  pragma pop_macro("spectre")
#  undef _CCCL_POP_MACRO_spectre
#endif

#if defined(thread)
#  error \
    "cccl internal error: macro `thread` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_thread)
#  pragma pop_macro("thread")
#  undef _CCCL_POP_MACRO_thread
#endif

#if defined(uuid)
#  error \
    "cccl internal error: macro `uuid` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_uuid)
#  pragma pop_macro("uuid")
#  undef _CCCL_POP_MACRO_uuid
#endif

// [[msvc::attribute]] attributes

#if defined(msvc)
#  error \
    "cccl internal error: macro `msvc` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_msvc)
#  pragma pop_macro("msvc")
#  undef _CCCL_POP_MACRO_msvc
#endif

#if defined(flatten)
#  error \
    "cccl internal error: macro `flatten` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_flatten)
#  pragma pop_macro("flatten")
#  undef _CCCL_POP_MACRO_flatten
#endif

#if defined(forceinline)
#  error \
    "cccl internal error: macro `forceinline` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_forceinline)
#  pragma pop_macro("forceinline")
#  undef _CCCL_POP_MACRO_forceinline
#endif

#if defined(forceinline_calls)
#  error \
    "cccl internal error: macro `forceinline_calls` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_forceinline_calls)
#  pragma pop_macro("forceinline_calls")
#  undef _CCCL_POP_MACRO_forceinline_calls
#endif

#if defined(intrinsic)
#  error \
    "cccl internal error: macro `intrinsic` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_intrinsic)
#  pragma pop_macro("intrinsic")
#  undef _CCCL_POP_MACRO_intrinsic
#endif

#if defined(noinline)
#  error \
    "cccl internal error: macro `noinline` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_noinline)
#  pragma pop_macro("noinline")
#  undef _CCCL_POP_MACRO_noinline
#endif

#if defined(noinline_calls)
#  error \
    "cccl internal error: macro `noinline_calls` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_noinline_calls)
#  pragma pop_macro("noinline_calls")
#  undef _CCCL_POP_MACRO_noinline_calls
#endif

#if defined(no_tls_guard)
#  error \
    "cccl internal error: macro `no_tls_guard` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_no_tls_guard)
#  pragma pop_macro("no_tls_guard")
#  undef _CCCL_POP_MACRO_no_tls_guard
#endif

// Windows nasty macros

#if defined(min)
#  error \
    "cccl internal error: macro `min` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_min)
#  pragma pop_macro("min")
#  undef _CCCL_POP_MACRO_min
#endif

#if defined(max)
#  error \
    "cccl internal error: macro `max` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_max)
#  pragma pop_macro("max")
#  undef _CCCL_POP_MACRO_max
#endif

#if defined(interface)
#  error \
    "cccl internal error: macro `interface` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_interface)
#  pragma pop_macro("interface")
#  undef _CCCL_POP_MACRO_interface
#endif

#if defined(__valid)
#  error \
    "cccl internal error: macro `__valid` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO___valid)
#  pragma pop_macro("__valid")
#  undef _CCCL_POP_MACRO___valid
#endif

// other macros

#if defined(clang)
#  error \
    "cccl internal error: macro `clang` was redefined between <cuda/std/__cccl/prologue.h> and <cuda/std/__cccl/epilogue.h>"
#elif defined(_CCCL_POP_MACRO_clang)
#  pragma pop_macro("clang")
#  undef _CCCL_POP_MACRO_clang
#endif

// NO include guards here (this file is included multiple times)
