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

#include <math.h>
#include <assert.h>

/* Any 64-bit Android math library should provide these functions,
 * so these wrappers are only needed for 32-bit systems.
 */
#ifndef __LP64__

/*
 * On 32-bit Android, long double and double are identical, hence
 * nexttoward is the same as nextafter.
 */

__attribute__((weak)) double nexttoward(double d, long double td) {
  return nextafter(d, (double)td);
}

__attribute__((weak)) float nexttowardf(float f, long double td) {
  return nextafterf(f, (float)td);
}

__attribute__((weak)) long double nexttowardl(long double ld, long double td) {
  return nextafter((double)ld, (double)td);
}

__attribute__((weak)) long double acosl(long double x) { return acos((double)x); }
__attribute__((weak)) long double asinl(long double x) { return asin((double)x); }
__attribute__((weak)) long double atanl(long double x) { return atan((double)x); }
__attribute__((weak)) long double atan2l(long double x, long double y) { return atan2((double)x, (double)y); }
__attribute__((weak)) long double cosl(long double x) { return cos((double)x); }
__attribute__((weak)) long double coshl(long double x) { return cosh((double)x); }
__attribute__((weak)) long double expl(long double x) { return exp((double)x); }
__attribute__((weak)) long double fmodl(long double x, long double y) { return fmod((double)x, (double)y); }
__attribute__((weak)) long double powl(long double x, long double y) { return pow((double)x, (double)y); }
__attribute__((weak)) long double sinl(long double x) { return sin((double)x); }
__attribute__((weak)) long double sinhl(long double x) { return sinh((double)x); }
__attribute__((weak)) long double sqrtl(long double x) { return sqrt((double)x); }
__attribute__((weak)) long double tanl(long double x) { return tan((double)x); }
__attribute__((weak)) long double tanhl(long double x) { return tanh((double)x); }
__attribute__((weak)) long double acoshl(long double x) { return acosh((double)x); }
__attribute__((weak)) long double asinhl(long double x) { return asinh((double)x); }
__attribute__((weak)) long double atanhl(long double x) { return atanh((double)x); }
__attribute__((weak)) long double cbrtl(long double x) { return cbrt((double)x); }
__attribute__((weak)) long double erfl(long double x) { return erf((double)x); }
__attribute__((weak)) long double erfcl(long double x) { return erfc((double)x); }
__attribute__((weak)) long double expm1l(long double x) { return expm1((double)x); }
__attribute__((weak)) long double hypotl(long double x, long double y) { return hypot((double)x, (double)y); }
__attribute__((weak)) long double lgammal(long double x) { return lgamma((double)x); }
__attribute__((weak)) long long int llrintl(long double x) { return llrint((double)x); }
__attribute__((weak)) long double logl(long double x) { return log((double)x); }
__attribute__((weak)) long double log1pl(long double x) { return log1p((double)x); }
__attribute__((weak)) long double log2l(long double x) { return log2((double)x); }
__attribute__((weak)) long double logbl(long double x) { return logb((double)x); }
__attribute__((weak)) long double log10l(long double x) { return log10((double)x); }
__attribute__((weak)) long double nanl(const char* s) { return nan(s); }
__attribute__((weak)) long double nearbyintl(long double x) { return nearbyint((double)x); }
__attribute__((weak)) long double remainderl(long double x, long double y) { return remainder((double)x, (double)y); }
__attribute__((weak)) long double remquol(long double x, long double y, int* i) { return remquo((double)x, (double)y, i); }
__attribute__((weak)) long double rintl(long double x) { return rint((double)x); }
__attribute__((weak)) long int lrintl(long double x) { return lrint((double)x); }
__attribute__((weak)) long double tgammal(long double x) { return tgamma((double)x); }
__attribute__((weak)) long double modfl(long double x, long double* y) { return modf((double)x, (double *)y); }
__attribute__((weak)) long double exp2l(long double x) { return exp2((double)x); }

#endif  // !__LP64__
