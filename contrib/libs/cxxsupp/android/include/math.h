/*
 * Copyright (C) 2013 The Android Open Source Project
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#ifndef NDK_ANDROID_SUPPORT_MATH_H
#define NDK_ANDROID_SUPPORT_MATH_H

#include_next <math.h>

#if !defined(__LP64__)

#ifdef __cplusplus
extern "C" {
#endif

// TODO(digit): Check that this is not needed for Clang.
typedef double      double_t;
typedef float       float_t;

// Missing long double functions. Note that 'long double' is the same
// as 'double' on 32-bit Android, so this will define stubs.
#define LLVM_LIBCXX_LONG_DOUBLE_FUNCTIONS

long double     acosl(long double);
long double     asinl(long double);
long double     atanl(long double);
long double     atan2l(long double x, long double y);
long double     cosl(long double);
long double     coshl(long double);
long double     expl(long double);
long double     fmodl(long double, long double);
long double     powl(long double, long double);
long double     sinl(long double);
long double     sinhl(long double);
long double     sqrtl(long double);
long double     tanl(long double);
long double     tanhl(long double);
long double     acoshl(long double);
long double     asinhl(long double);
long double     atanhl(long double);
long double     cbrtl(long double);
long double     erfl(long double);
long double     erfcl(long double);
long double     expm1l(long double);
long double     hypotl(long double, long double);
long double     lgammal(long double);
long long int   llrintl(long double);
long double     logl(long double);
long double     log1pl(long double);
long double     log2l(long double);
long double     logbl(long double);
long double     log10l(long double);
long double     nanl(const char*);
long double     nearbyintl(long double);
long double     remainderl(long double, long double);
long double     remquol(long double, long double, int*);
long double     rintl(long double);
long int        lrintl(long double);
long double     tgammal(long double);
long double     modfl(long double, long double*);
long double     exp2l(long double);

float           tgammaf(float);
double          nan(const char*);
float           nanf(const char*);

float           log2f(float);
double          log2(double);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // !__LP64__

#endif  /* NDK_ANDROID_SUPPORT_MATH_H */
