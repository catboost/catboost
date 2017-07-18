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
#ifndef NDK_ANDROID_SUPPORT_LANGINFO_H
#define NDK_ANDROID_SUPPORT_LANGINFO_H

// __LP64__

#include <nl_types.h>

#define _NL_ITEM(category,index)  (((category) << 10) | (index))

#define _NL_ITEM_CATEGORY(nl)  ((nl) >> 10)
#define _NL_ITEM_INDEX(nl)     ((nl) & 0x3ff)

#define CODESET _NL_ITEM(LC_CTYPE, 0)

/* Abbreviated days of the week */
#define ABDAY_1 _NL_ITEM(LC_TIME,1)
#define ABDAY_2 _NL_ITEM(LC_TIME,2)
#define ABDAY_3 _NL_ITEM(LC_TIME,3)
#define ABDAY_4 _NL_ITEM(LC_TIME,4)
#define ABDAY_5 _NL_ITEM(LC_TIME,5)
#define ABDAY_6 _NL_ITEM(LC_TIME,6)
#define ABDAY_7 _NL_ITEM(LC_TIME,7)

/* Long names of the week */
#define DAY_1   _NL_ITEM(LC_TIME,11)
#define DAY_2   _NL_ITEM(LC_TIME,12)
#define DAY_3   _NL_ITEM(LC_TIME,13)
#define DAY_4   _NL_ITEM(LC_TIME,14)
#define DAY_5   _NL_ITEM(LC_TIME,15)
#define DAY_6   _NL_ITEM(LC_TIME,16)
#define DAY_7   _NL_ITEM(LC_TIME,17)

/* Abbreviated month names */
#define ABMON_1  _NL_ITEM(LC_TIME,21)
#define ABMON_2  _NL_ITEM(LC_TIME,22)
#define ABMON_3  _NL_ITEM(LC_TIME,23)
#define ABMON_4  _NL_ITEM(LC_TIME,24)
#define ABMON_5  _NL_ITEM(LC_TIME,25)
#define ABMON_6  _NL_ITEM(LC_TIME,26)
#define ABMON_7  _NL_ITEM(LC_TIME,27)
#define ABMON_8  _NL_ITEM(LC_TIME,28)
#define ABMON_9  _NL_ITEM(LC_TIME,29)
#define ABMON_10 _NL_ITEM(LC_TIME,30)
#define ABMON_11 _NL_ITEM(LC_TIME,31)
#define ABMON_12 _NL_ITEM(LC_TIME,32)

/* Long month names */
#define MON_1    _NL_ITEM(LC_TIME,41)
#define MON_2    _NL_ITEM(LC_TIME,42)
#define MON_3    _NL_ITEM(LC_TIME,43)
#define MON_4    _NL_ITEM(LC_TIME,44)
#define MON_5    _NL_ITEM(LC_TIME,45)
#define MON_6    _NL_ITEM(LC_TIME,46)
#define MON_7    _NL_ITEM(LC_TIME,47)
#define MON_8    _NL_ITEM(LC_TIME,48)
#define MON_9    _NL_ITEM(LC_TIME,49)
#define MON_10   _NL_ITEM(LC_TIME,50)
#define MON_11   _NL_ITEM(LC_TIME,51)
#define MON_12   _NL_ITEM(LC_TIME,52)

#define AM_STR      _NL_ITEM(LC_TIME,53)
#define PM_STR      _NL_ITEM(LC_TIME,54)
#define D_T_FMT     _NL_ITEM(LC_TIME,55)
#define D_FMT       _NL_ITEM(LC_TIME,56)
#define T_FMT       _NL_ITEM(LC_TIME,57)
#define T_FMT_AMPM  _NL_ITEM(LC_TIME,58)
#define ERA         _NL_ITEM(LC_TIME,59)
#define ERA_D_FMT   _NL_ITEM(LC_TIME,60)
#define ERA_D_T_FMT _NL_ITEM(LC_TIME,61)
#define ERA_T_FMT   _NL_ITEM(LC_TIME,62)
#define ALT_DIGITS  _NL_ITEM(LC_TIME,70)

#define INT_CURRENCY_SYMBOL     _NL_ITEM(LC_MONETARY,0)
#define CURRENCY_SYMBOL         _NL_ITEM(LC_MONETARY,1)
#define MON_DECIMAL_POINT       _NL_ITEM(LC_MONETARY,2)
#define MON_THOUSANDS_SEP       _NL_ITEM(LC_MONETARY,3)
#define MON_GROUPING            _NL_ITEM(LC_MONETARY,4)
#define POSITIVE_SIGN           _NL_ITEM(LC_MONETARY,5)
#define NEGATIVE_SIGN           _NL_ITEM(LC_MONETARY,6)
#define INT_FRAC_DIGITS         _NL_ITEM(LC_MONETARY,7)
#define FRAC_DIGITS             _NL_ITEM(LC_MONETARY,8)

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__LP64__)
char *nl_langinfo(nl_item);
char *nl_langinfo_l(nl_item, locale_t);
#endif // !__LP64__

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  /* NDK_ANDROID_SUPPORT_LANGINFO_H */

