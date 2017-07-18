// -*- C++ -*-
//===--------------------------- stdbool.h --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP_STDBOOL_H
#define _LIBCPP_STDBOOL_H


/*
    stdbool.h synopsis

Macros:

    __bool_true_false_are_defined

*/

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#ifdef _LIBCPP_COMPILER_MSVC
#include _LIBCPP_MSVC_INCLUDE(stdbool.h)
#else
#include_next <stdbool.h>
#endif

#ifdef __cplusplus
#undef bool
#undef true
#undef false
#undef __bool_true_false_are_defined
#define __bool_true_false_are_defined 1
#endif

#endif  // _LIBCPP_STDBOOL_H
