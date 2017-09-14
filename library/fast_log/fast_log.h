#pragma once

#include <util/system/types.h>
#include <util/system/yassert.h>

/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/
/* The above copyright message relates only to funcions FastLog2f(),
 * FastLogf(), FastestLog2f() and FastestLogf() from below.
 */

static inline bool LogInputCheck(float value) noexcept {
    return value == value //not a NaN
        && value >= 0;    //Within allowable range
}

/**
 * @returns     Base 2 logarithm of the value.
 *              This is fast, inline and quite accurate algorithm.
 *              Accuracy: ~1.e-5
 *              Speed: ~3x over logf()
 *              Source: https://code.google.com/archive/p/fastapprox/
 *              Description: http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
 */
static inline float FastLog2f(float value) noexcept {
    Y_ASSERT(LogInputCheck(value));
    union {
        float f;
        ui32 i;
    } vx = {value};
    union {
        ui32 i;
        float f;
    } mx = {(vx.i & 0x007FFFFF) | 0x3f000000};
    float y = vx.i;
    y *= 1.1920928955078125e-7f;

    return y - 124.22551499f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
}

/**
 * @returns     Base e logarithm of the value.
 *              This is fast, inline and quite accurate algorithm.
 *              Accuracy: ~1.e-5
 *              Speed: ~3x over logf()
 */
static inline float FastLogf(float value) noexcept {
    return 0.69314718f * FastLog2f(value);
}

/**
 * @returns     Base 2 logarithm of the value.
 *              This is a fast implementation, which implies lower accuracy.
 *              Accuracy: ~1.e-3
 *              Speed: ~6x over logf()
 *              Source: http://www.flipcode.com/archives/Fast_log_Function.shtml
 */
static inline float FasterLog2f(float value) noexcept {
    Y_ASSERT(LogInputCheck(value));
    union {
        float f;
        int i;
    } vx = {value};
    const int log2 = ((vx.i >> 23) & 255) - 128;

    vx.i &= ~(255 << 23);
    vx.i += 127 << 23;

    return ((-1.0f / 3) * vx.f + 2) * vx.f - 2.0f / 3 + log2;
}

/**
 * @returns     Base e logarithm of the value.
 *              This is a fast implementation, which implies lower accuracy.
 *              Accuracy: ~1.e-3
 *              Speed: ~6x over logf()
 */
static inline float FasterLogf(float value) noexcept {
    return FasterLog2f(value) * 0.69314718f;
}

/**
 * @returns     Base 2 logarithm of the value.
 *              This is a very fast implementation for a notable sake of accuracy.
 *              Accuracy: ~1.e-2
 *              Speed: ~12x over logf()
 *              Source: https://code.google.com/archive/p/fastapprox/
 *              Description: http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
 */
static inline float FastestLog2f(float value) noexcept {
    Y_ASSERT(LogInputCheck(value));
    union {
        float f;
        ui32 i;
    } vx = {value};
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 126.94269504f;
}

/**
 * @returns     Base e logarithm of the value.
 *              This is a very fast implementation for a notable sake of accuracy.
 *              Accuracy: ~1.e-2
 *              Speed: ~12x over logf()
 *              Source: https://code.google.com/archive/p/fastapprox/
 *              Description: http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
 */
static inline float FastestLogf(float value) noexcept {
    Y_ASSERT(LogInputCheck(value));
    union {
        float f;
        ui32 i;
    } vx = {value};
    float y = vx.i;
    y *= 8.2629582881927490e-8f;
    return y - 87.989971088f;
}
