// -*- C++ -*-
//===-------------------- support/android/locale_android.cpp ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ctype.h>

// fix names for clang50 toolchain ctype.h
#if defined(__clang__) && __clang_major__ >= 5
#define _U _CTYPE_U
#define _L _CTYPE_L
#define _N _CTYPE_D
#define _S _CTYPE_S
#define _P _CTYPE_P
#define _C _CTYPE_C
#define _X _CTYPE_X
#define _B _CTYPE_B
#endif

// Bionic exports the non-standard _ctype_ array in <ctype.h>,
// unfortunately, cannot be used directly for libc++ because it doesn't
// have a proper bit-flag for blank characters.
//
// Note that the header does define a _B flag (as 0x80), but it
// is only set on the space (32) character, and used to implement
// isprint() properly. The implementation of isblank() relies on
// direct comparisons with 9 and 32 instead.
//
// The following is a local copy of the Bionic _ctype_ array that has
// been modified in the following way:
//
//   - It stores 16-bit unsigned values, instead of 8-bit char ones.
//
//   - Bit flag _BLANK (0x100) is used to indicate blank characters.
//     It is only set for indices 9 (TAB) and 32 (SPACE).
//
//   - Support signed char properly for indexing.

// Used to tag blank characters, this doesn't appear in <ctype.h> nor
// the original Bionic _ctype_ array.
#define _BLANK  0x100

// NOTE: A standalone forward declaration is required to ensure that this
// variable is properly exported with a C name. In other words, this does
// _not_ work:
//
//  extern "C" {
//  const char* const _ctype_android = ...;
//  }
//
extern "C" const unsigned short* const _ctype_android;

static const unsigned short ctype_android_tab[256+128] = {
       /* -128..-1 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 80 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 88 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 90 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 98 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* A0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* A8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* B0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* B8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* C0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* C8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* D0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* D8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* E0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* E8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* F0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* F8 */
       /* 0..127 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C,
        _C,     _C|_S|_BLANK, _C|_S,  _C|_S,  _C|_S,  _C|_S,  _C,     _C,
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C,
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C,
  _S|_B|_BLANK, _P,           _P,     _P,     _P,     _P,     _P,     _P,
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P,
        _N,     _N,           _N,     _N,     _N,     _N,     _N,     _N,
        _N,     _N,           _P,     _P,     _P,     _P,     _P,     _P,
        _P,     _U|_X,        _U|_X,  _U|_X,  _U|_X,  _U|_X,  _U|_X,  _U,
        _U,     _U,           _U,     _U,     _U,     _U,     _U,     _U,
        _U,     _U,           _U,     _U,     _U,     _U,     _U,     _U,
        _U,     _U,           _U,     _P,     _P,     _P,     _P,     _P,
        _P,     _L|_X,        _L|_X,  _L|_X,  _L|_X,  _L|_X,  _L|_X,  _L,
        _L,     _L,           _L,     _L,     _L,     _L,     _L,     _L,
        _L,     _L,           _L,     _L,     _L,     _L,     _L,     _L,
        /* determine printability based on the IS0 8859 8-bit standard */
        _L,     _L,           _L,     _P,     _P,     _P,     _P,     _C,
        /* 128..255, same as -128..127 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 80 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 88 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 90 */
        _C,     _C,           _C,     _C,     _C,     _C,     _C,     _C, /* 98 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* A0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* A8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* B0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* B8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* C0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* C8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* D0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* D8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* E0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* E8 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* F0 */
        _P,     _P,           _P,     _P,     _P,     _P,     _P,     _P, /* F8 */
};

const unsigned short* const _ctype_android = ctype_android_tab + 128;
