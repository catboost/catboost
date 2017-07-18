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
#ifndef NDK_ANDROID_SUPPORT_STDIO_H
#define NDK_ANDROID_SUPPORT_STDIO_H

#if defined(__LP64__)

#include_next <stdio.h>

#else

// This is to avoid a compiler error when the putc() macro definition
// in <stdio.h> follows a putc() function definition which is apparently
// not compatible with it.
#define _POSIX_THREADS 1
#include_next <stdio.h>

#include <stdarg.h>
#include <wchar.h>
#include <xlocale.h>

#ifdef __cplusplus
extern "C" {
#endif

int asprintf_l(char**, locale_t, const char*, ...);
int sprintf_l(char*, locale_t, const char*, ...);
int snprintf_l(char*, size_t, locale_t, const char*, ...);
int sscanf_l(const char*, locale_t, const char*, ...);

int vfwscanf(FILE* __restrict__, const wchar_t* __restrict__, va_list);
int vswscanf(const wchar_t *__restrict__, const wchar_t * __restrict__, va_list);
int vwscanf(const wchar_t *__restrict__, va_list);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // !__LP64__

#endif  // NDK_ANDROID_SUPPORT_STDIO_H
