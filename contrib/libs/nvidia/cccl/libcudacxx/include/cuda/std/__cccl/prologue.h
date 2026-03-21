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

#if defined(_CCCL_PROLOGUE_INCLUDED)
#  error \
    "cccl internal error: <cuda/std/__cccl/epilogue.h> must be included before next <cuda/std/__cccl/prologue.h> is reincluded"
#endif
#define _CCCL_PROLOGUE_INCLUDED() 1

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/diagnostic.h>

// __declspec modifiers

#if defined(align)
#  pragma push_macro("align")
#  undef align
#  define _CCCL_POP_MACRO_align
#endif // defined(align)

#if defined(allocate)
#  pragma push_macro("allocate")
#  undef allocate
#  define _CCCL_POP_MACRO_allocate
#endif // defined(allocate)

#if defined(allocator)
#  pragma push_macro("allocator")
#  undef allocator
#  define _CCCL_POP_MACRO_allocator
#endif // defined(allocator)

#if defined(appdomain)
#  pragma push_macro("appdomain")
#  undef appdomain
#  define _CCCL_POP_MACRO_appdomain
#endif // defined(appdomain)

#if defined(code_seg)
#  pragma push_macro("code_seg")
#  undef code_seg
#  define _CCCL_POP_MACRO_code_seg
#endif // defined(code_seg)

#if defined(deprecated)
#  pragma push_macro("deprecated")
#  undef deprecated
#  define _CCCL_POP_MACRO_deprecated
#endif // defined(deprecated)

#if defined(dllimport)
#  pragma push_macro("dllimport")
#  undef dllimport
#  define _CCCL_POP_MACRO_dllimport
#endif // defined(dllimport)

#if defined(dllexport)
#  pragma push_macro("dllexport")
#  undef dllexport
#  define _CCCL_POP_MACRO_dllexport
#endif // defined(dllexport)

#if defined(empty_bases)
#  pragma push_macro("empty_bases")
#  undef empty_bases
#  define _CCCL_POP_MACRO_empty_bases
#endif // defined(empty_bases)

#if defined(hybrid_patchable)
#  pragma push_macro("hybrid_patchable")
#  undef hybrid_patchable
#  define _CCCL_POP_MACRO_hybrid_patchable
#endif // defined(hybrid_patchable)

#if defined(jitintrinsic)
#  pragma push_macro("jitintrinsic")
#  undef jitintrinsic
#  define _CCCL_POP_MACRO_jitintrinsic
#endif // defined(jitintrinsic)

#if defined(naked)
#  pragma push_macro("naked")
#  undef naked
#  define _CCCL_POP_MACRO_naked
#endif // defined(naked)

#if defined(noalias)
#  pragma push_macro("noalias")
#  undef noalias
#  define _CCCL_POP_MACRO_noalias
#endif // defined(noalias)

#if defined(noinline)
#  pragma push_macro("noinline")
#  undef noinline
#  define _CCCL_POP_MACRO_noinline
#endif // defined(noinline)

#if defined(noreturn)
#  pragma push_macro("noreturn")
#  undef noreturn
#  define _CCCL_POP_MACRO_noreturn
#endif // defined(noreturn)

#if defined(nothrow)
#  pragma push_macro("nothrow")
#  undef nothrow
#  define _CCCL_POP_MACRO_nothrow
#endif // defined(nothrow)

#if defined(novtable)
#  pragma push_macro("novtable")
#  undef novtable
#  define _CCCL_POP_MACRO_novtable
#endif // defined(novtable)

#if defined(no_sanitize_address)
#  pragma push_macro("no_sanitize_address")
#  undef no_sanitize_address
#  define _CCCL_POP_MACRO_no_sanitize_address
#endif // defined(no_sanitize_address)

#if defined(process)
#  pragma push_macro("process")
#  undef process
#  define _CCCL_POP_MACRO_process
#endif // defined(process)

#if defined(property)
#  pragma push_macro("property")
#  undef property
#  define _CCCL_POP_MACRO_property
#endif // defined(property)

#if defined(restrict)
#  pragma push_macro("restrict")
#  undef restrict
#  define _CCCL_POP_MACRO_restrict
#endif // defined(restrict)

#if defined(safebuffers)
#  pragma push_macro("safebuffers")
#  undef safebuffers
#  define _CCCL_POP_MACRO_safebuffers
#endif // defined(safebuffers)

#if defined(selectany)
#  pragma push_macro("selectany")
#  undef selectany
#  define _CCCL_POP_MACRO_selectany
#endif // defined(selectany)

#if defined(spectre)
#  pragma push_macro("spectre")
#  undef spectre
#  define _CCCL_POP_MACRO_spectre
#endif // defined(spectre)

#if defined(thread)
#  pragma push_macro("thread")
#  undef thread
#  define _CCCL_POP_MACRO_thread
#endif // defined(thread)

#if defined(uuid)
#  pragma push_macro("uuid")
#  undef uuid
#  define _CCCL_POP_MACRO_uuid
#endif // defined(uuid)

// [[msvc::attribute]] attributes

#if defined(msvc)
#  pragma push_macro("msvc")
#  undef msvc
#  define _CCCL_POP_MACRO_msvc
#endif // defined(msvc)

#if defined(flatten)
#  pragma push_macro("flatten")
#  undef flatten
#  define _CCCL_POP_MACRO_flatten
#endif // defined(flatten)

#if defined(forceinline)
#  pragma push_macro("forceinline")
#  undef forceinline
#  define _CCCL_POP_MACRO_forceinline
#endif // defined(forceinline)

#if defined(forceinline_calls)
#  pragma push_macro("forceinline_calls")
#  undef forceinline_calls
#  define _CCCL_POP_MACRO_forceinline_calls
#endif // defined(forceinline_calls)

#if defined(intrinsic)
#  pragma push_macro("intrinsic")
#  undef intrinsic
#  define _CCCL_POP_MACRO_intrinsic
#endif // defined(intrinsic)

#if defined(noinline)
#  pragma push_macro("noinline")
#  undef noinline
#  define _CCCL_POP_MACRO_noinline
#endif // defined(noinline)

#if defined(noinline_calls)
#  pragma push_macro("noinline_calls")
#  undef noinline_calls
#  define _CCCL_POP_MACRO_noinline_calls
#endif // defined(noinline_calls)

#if defined(no_tls_guard)
#  pragma push_macro("no_tls_guard")
#  undef no_tls_guard
#  define _CCCL_POP_MACRO_no_tls_guard
#endif // defined(no_tls_guard)

// Windows nasty macros

#if defined(min)
#  pragma push_macro("min")
#  undef min
#  define _CCCL_POP_MACRO_min
#endif // defined(min)

#if defined(max)
#  pragma push_macro("max")
#  undef max
#  define _CCCL_POP_MACRO_max
#endif // defined(max)

#if defined(interface)
#  pragma push_macro("interface")
#  undef interface
#  define _CCCL_POP_MACRO_interface
#endif // defined(interface)

#if defined(__valid)
#  pragma push_macro("__valid")
#  undef __valid
#  define _CCCL_POP_MACRO___valid
#endif // defined(__valid)

// other macros

#if defined(clang)
#  pragma push_macro("clang")
#  undef clang
#  define _CCCL_POP_MACRO_clang
#endif // defined(clang)

_CCCL_DIAG_PUSH

// disable some msvc warnings
// https://github.com/microsoft/STL/blob/master/stl/inc/yvals_core.h#L353
// warning C4100: 'quack': unreferenced formal parameter
// warning C4127: conditional expression is constant
// warning C4180: qualifier applied to function type has no meaning; ignored
// warning C4197: 'purr': top-level volatile in cast is ignored
// warning C4324: 'roar': structure was padded due to alignment specifier
// warning C4455: literal suffix identifiers that do not start with an underscore are reserved
// warning C4503: 'hum': decorated name length exceeded, name was truncated
// warning C4522: 'woof' : multiple assignment operators specified
// warning C4668: 'meow' is not defined as a preprocessor macro, replacing with '0' for '#if/#elif'
// warning C4800: 'boo': forcing value to bool 'true' or 'false' (performance warning)
// warning C4996: 'meow': was declared deprecated
_CCCL_DIAG_SUPPRESS_MSVC(4100 4127 4180 4197 4296 4324 4455 4503 4522 4668 4800 4996)

// NO include guards here (this file is included multiple times)
