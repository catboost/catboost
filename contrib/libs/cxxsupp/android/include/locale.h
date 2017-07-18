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
#ifndef NDK_ANDROID_SUPPORT_LOCALE_H
#define NDK_ANDROID_SUPPORT_LOCALE_H

#if defined(__LP64__)

#include_next <locale.h>

#else

#define lconv  __libc_lconv
#define localeconv  __libc_localeconv
#include_next <locale.h>
#undef lconv
#undef localeconv
#include <xlocale.h>

#if !defined(LC_CTYPE)
/* Define all LC_XXX to itself.  Sounds silly but libc++ expects it's defined, not in enum */
#define LC_CTYPE           LC_CTYPE
#define LC_NUMERIC         LC_NUMERIC
#define LC_TIME            LC_TIME
#define LC_COLLATE         LC_COLLATE
#define LC_MONETARY        LC_MONETARY
#define LC_MESSAGES        LC_MESSAGES
#define LC_ALL             LC_ALL
#define LC_PAPER           LC_PAPER
#define LC_NAME            LC_NAME
#define LC_ADDRESS         LC_ADDRESS
#define LC_TELEPHONE       LC_TELEPHONE
#define LC_MEASUREMENT     LC_MEASUREMENT
#define LC_IDENTIFICATION  LC_IDENTIFICATION
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LC_CTYPE_MASK  (1 << LC_CTYPE)
#define LC_NUMERIC_MASK (1 << LC_NUMERIC)
#define LC_TIME_MASK (1 << LC_TIME)
#define LC_COLLATE_MASK (1 << LC_COLLATE)
#define LC_MONETARY_MASK (1 << LC_MONETARY)
#define LC_MESSAGES_MASK (1 << LC_MESSAGES)
#define LC_PAPER_MASK (1 << LC_PAPER)
#define LC_NAME_MASK (1 << LC_NAME)
#define LC_ADDRESS_MASK (1 << LC_ADDRESS)
#define LC_TELEPHONE_MASK (1 << LC_TELEPHONE)
#define LC_MEASUREMENT_MASK (1 << LC_MEASUREMENT)
#define LC_IDENTIFICATION_MASK (1 << LC_IDENTIFICATION)

#undef LC_ALL_MASK
#define LC_ALL_MASK (LC_CTYPE_MASK \
                     | LC_NUMERIC_MASK \
                     | LC_TIME_MASK \
                     | LC_COLLATE_MASK \
                     | LC_MONETARY_MASK \
                     | LC_MESSAGES_MASK \
                     | LC_PAPER_MASK \
                     | LC_NAME_MASK \
                     | LC_ADDRESS_MASK \
                     | LC_TELEPHONE_MASK \
                     | LC_MEASUREMENT_MASK \
                     | LC_IDENTIFICATION_MASK \
                     )

extern locale_t newlocale(int, const char*, locale_t);
extern locale_t uselocale(locale_t);
extern void     freelocale(locale_t);

#define LC_GLOBAL_LOCALE    ((locale_t) -1L)

struct lconv {
    char* decimal_point;       /* Decimal point character */
    char* thousands_sep;       /* Thousands separator */
    char* grouping;            /* Grouping */
    char* int_curr_symbol;
    char* currency_symbol;
    char* mon_decimal_point;
    char* mon_thousands_sep;
    char* mon_grouping;
    char* positive_sign;
    char* negative_sign;
    char  int_frac_digits;
    char  frac_digits;
    char  p_cs_precedes;
    char  p_sep_by_space;
    char  n_cs_precedes;
    char  n_sep_by_space;
    char  p_sign_posn;
    char  n_sign_posn;
    /* ISO-C99 */
    char  int_p_cs_precedes;
    char  int_p_sep_by_space;
    char  int_n_cs_precedes;
    char  int_n_sep_by_space;
    char  int_p_sign_posn;
    char  int_n_sign_posn;
};

struct lconv* localeconv(void);

#if 0
// Used to implement the std::ctype<char> specialization.
extern const char * const __ctype_c_mask_table;
// TODO(ajwong): Make this based on some exported bionic constant.
const int __ctype_c_mask_table_size = 256;
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // !__LP64__

#endif  // NDK_ANDROID_SUPPORT_LOCALE_H
