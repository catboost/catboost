/* Determine a canonical name for the current locale's character encoding.

   Copyright (C) 2000-2006, 2008-2019 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>.  */

#include <config.h>

/* Specification.  */
#include "localcharset.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if defined __APPLE__ && defined __MACH__ && HAVE_LANGINFO_CODESET
# define DARWIN7 /* Darwin 7 or newer, i.e. Mac OS X 10.3 or newer */
#endif

#if defined _WIN32 && !defined __CYGWIN__
# define WINDOWS_NATIVE
# include <locale.h>
#endif

#if defined __EMX__
/* Assume EMX program runs on OS/2, even if compiled under DOS.  */
# ifndef OS2
#  define OS2
# endif
#endif

#if !defined WINDOWS_NATIVE
# if HAVE_LANGINFO_CODESET
#  include <langinfo.h>
# else
#  if 0 /* see comment regarding use of setlocale(), below */
#   include <locale.h>
#  endif
# endif
# ifdef __CYGWIN__
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
# endif
#elif defined WINDOWS_NATIVE
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif
#if defined OS2
# define INCL_DOS
# error #include <os2.h>
#endif

/* For MB_CUR_MAX_L */
#if defined DARWIN7
# include <xlocale.h>
#endif


#if HAVE_LANGINFO_CODESET || defined WINDOWS_NATIVE || defined OS2

/* On these platforms, we use a mapping from non-canonical encoding name
   to GNU canonical encoding name.  */

/* With glibc-2.1 or newer, we don't need any canonicalization,
   because glibc has iconv and both glibc and libiconv support all
   GNU canonical names directly.  */
# if !((defined __GNU_LIBRARY__ && __GLIBC__ >= 2) || defined __UCLIBC__)

struct table_entry
{
  const char alias[11+1];
  const char canonical[11+1];
};

/* Table of platform-dependent mappings, sorted in ascending order.  */
static const struct table_entry alias_table[] =
  {
#  if defined __FreeBSD__                                   /* FreeBSD */
  /*{ "ARMSCII-8",  "ARMSCII-8" },*/
    { "Big5",       "BIG5" },
    { "C",          "ASCII" },
  /*{ "CP1131",     "CP1131" },*/
  /*{ "CP1251",     "CP1251" },*/
  /*{ "CP866",      "CP866" },*/
  /*{ "GB18030",    "GB18030" },*/
  /*{ "GB2312",     "GB2312" },*/
  /*{ "GBK",        "GBK" },*/
  /*{ "ISCII-DEV",  "?" },*/
    { "ISO8859-1",  "ISO-8859-1" },
    { "ISO8859-13", "ISO-8859-13" },
    { "ISO8859-15", "ISO-8859-15" },
    { "ISO8859-2",  "ISO-8859-2" },
    { "ISO8859-5",  "ISO-8859-5" },
    { "ISO8859-7",  "ISO-8859-7" },
    { "ISO8859-9",  "ISO-8859-9" },
  /*{ "KOI8-R",     "KOI8-R" },*/
  /*{ "KOI8-U",     "KOI8-U" },*/
    { "SJIS",       "SHIFT_JIS" },
    { "US-ASCII",   "ASCII" },
    { "eucCN",      "GB2312" },
    { "eucJP",      "EUC-JP" },
    { "eucKR",      "EUC-KR" }
#   define alias_table_defined
#  endif
#  if defined __NetBSD__                                    /* NetBSD */
    { "646",        "ASCII" },
  /*{ "ARMSCII-8",  "ARMSCII-8" },*/
  /*{ "BIG5",       "BIG5" },*/
    { "Big5-HKSCS", "BIG5-HKSCS" },
  /*{ "CP1251",     "CP1251" },*/
  /*{ "CP866",      "CP866" },*/
  /*{ "GB18030",    "GB18030" },*/
  /*{ "GB2312",     "GB2312" },*/
    { "ISO8859-1",  "ISO-8859-1" },
    { "ISO8859-13", "ISO-8859-13" },
    { "ISO8859-15", "ISO-8859-15" },
    { "ISO8859-2",  "ISO-8859-2" },
    { "ISO8859-4",  "ISO-8859-4" },
    { "ISO8859-5",  "ISO-8859-5" },
    { "ISO8859-7",  "ISO-8859-7" },
  /*{ "KOI8-R",     "KOI8-R" },*/
  /*{ "KOI8-U",     "KOI8-U" },*/
  /*{ "PT154",      "PT154" },*/
    { "SJIS",       "SHIFT_JIS" },
    { "eucCN",      "GB2312" },
    { "eucJP",      "EUC-JP" },
    { "eucKR",      "EUC-KR" },
    { "eucTW",      "EUC-TW" }
#   define alias_table_defined
#  endif
#  if defined __OpenBSD__                                   /* OpenBSD */
    { "646",        "ASCII" },
    { "ISO8859-1",  "ISO-8859-1" },
    { "ISO8859-13", "ISO-8859-13" },
    { "ISO8859-15", "ISO-8859-15" },
    { "ISO8859-2",  "ISO-8859-2" },
    { "ISO8859-4",  "ISO-8859-4" },
    { "ISO8859-5",  "ISO-8859-5" },
    { "ISO8859-7",  "ISO-8859-7" }
#   define alias_table_defined
#  endif
#  if defined __APPLE__ && defined __MACH__                 /* Mac OS X */
    /* Darwin 7.5 has nl_langinfo(CODESET), but sometimes its value is
       useless:
       - It returns the empty string when LANG is set to a locale of the
         form ll_CC, although ll_CC/LC_CTYPE is a symlink to an UTF-8
         LC_CTYPE file.
       - The environment variables LANG, LC_CTYPE, LC_ALL are not set by
         the system; nl_langinfo(CODESET) returns "US-ASCII" in this case.
       - The documentation says:
           "... all code that calls BSD system routines should ensure
            that the const *char parameters of these routines are in UTF-8
            encoding. All BSD system functions expect their string
            parameters to be in UTF-8 encoding and nothing else."
         It also says
           "An additional caveat is that string parameters for files,
            paths, and other file-system entities must be in canonical
            UTF-8. In a canonical UTF-8 Unicode string, all decomposable
            characters are decomposed ..."
         but this is not true: You can pass non-decomposed UTF-8 strings
         to file system functions, and it is the OS which will convert
         them to decomposed UTF-8 before accessing the file system.
       - The Apple Terminal application displays UTF-8 by default.
       - However, other applications are free to use different encodings:
         - xterm uses ISO-8859-1 by default.
         - TextEdit uses MacRoman by default.
       We prefer UTF-8 over decomposed UTF-8-MAC because one should
       minimize the use of decomposed Unicode. Unfortunately, through the
       Darwin file system, decomposed UTF-8 strings are leaked into user
       space nevertheless.
       Then there are also the locales with encodings other than US-ASCII
       and UTF-8. These locales can be occasionally useful to users (e.g.
       when grepping through ISO-8859-1 encoded text files), when all their
       file names are in US-ASCII.
     */
    { "ARMSCII-8",  "ARMSCII-8" },
    { "Big5",       "BIG5" },
    { "Big5HKSCS",  "BIG5-HKSCS" },
    { "CP1131",     "CP1131" },
    { "CP1251",     "CP1251" },
    { "CP866",      "CP866" },
    { "CP949",      "CP949" },
    { "GB18030",    "GB18030" },
    { "GB2312",     "GB2312" },
    { "GBK",        "GBK" },
  /*{ "ISCII-DEV",  "?" },*/
    { "ISO8859-1",  "ISO-8859-1" },
    { "ISO8859-13", "ISO-8859-13" },
    { "ISO8859-15", "ISO-8859-15" },
    { "ISO8859-2",  "ISO-8859-2" },
    { "ISO8859-4",  "ISO-8859-4" },
    { "ISO8859-5",  "ISO-8859-5" },
    { "ISO8859-7",  "ISO-8859-7" },
    { "ISO8859-9",  "ISO-8859-9" },
    { "KOI8-R",     "KOI8-R" },
    { "KOI8-U",     "KOI8-U" },
    { "PT154",      "PT154" },
    { "SJIS",       "SHIFT_JIS" },
    { "eucCN",      "GB2312" },
    { "eucJP",      "EUC-JP" },
    { "eucKR",      "EUC-KR" }
#   define alias_table_defined
#  endif
#  if defined _AIX                                          /* AIX */
  /*{ "GBK",        "GBK" },*/
    { "IBM-1046",   "CP1046" },
    { "IBM-1124",   "CP1124" },
    { "IBM-1129",   "CP1129" },
    { "IBM-1252",   "CP1252" },
    { "IBM-850",    "CP850" },
    { "IBM-856",    "CP856" },
    { "IBM-921",    "ISO-8859-13" },
    { "IBM-922",    "CP922" },
    { "IBM-932",    "CP932" },
    { "IBM-943",    "CP943" },
    { "IBM-eucCN",  "GB2312" },
    { "IBM-eucJP",  "EUC-JP" },
    { "IBM-eucKR",  "EUC-KR" },
    { "IBM-eucTW",  "EUC-TW" },
    { "ISO8859-1",  "ISO-8859-1" },
    { "ISO8859-15", "ISO-8859-15" },
    { "ISO8859-2",  "ISO-8859-2" },
    { "ISO8859-5",  "ISO-8859-5" },
    { "ISO8859-6",  "ISO-8859-6" },
    { "ISO8859-7",  "ISO-8859-7" },
    { "ISO8859-8",  "ISO-8859-8" },
    { "ISO8859-9",  "ISO-8859-9" },
    { "TIS-620",    "TIS-620" },
  /*{ "UTF-8",      "UTF-8" },*/
    { "big5",       "BIG5" }
#   define alias_table_defined
#  endif
#  if defined __hpux                                        /* HP-UX */
    { "SJIS",      "SHIFT_JIS" },
    { "arabic8",   "HP-ARABIC8" },
    { "big5",      "BIG5" },
    { "cp1251",    "CP1251" },
    { "eucJP",     "EUC-JP" },
    { "eucKR",     "EUC-KR" },
    { "eucTW",     "EUC-TW" },
    { "gb18030",   "GB18030" },
    { "greek8",    "HP-GREEK8" },
    { "hebrew8",   "HP-HEBREW8" },
    { "hkbig5",    "BIG5-HKSCS" },
    { "hp15CN",    "GB2312" },
    { "iso88591",  "ISO-8859-1" },
    { "iso885913", "ISO-8859-13" },
    { "iso885915", "ISO-8859-15" },
    { "iso88592",  "ISO-8859-2" },
    { "iso88594",  "ISO-8859-4" },
    { "iso88595",  "ISO-8859-5" },
    { "iso88596",  "ISO-8859-6" },
    { "iso88597",  "ISO-8859-7" },
    { "iso88598",  "ISO-8859-8" },
    { "iso88599",  "ISO-8859-9" },
    { "kana8",     "HP-KANA8" },
    { "koi8r",     "KOI8-R" },
    { "roman8",    "HP-ROMAN8" },
    { "tis620",    "TIS-620" },
    { "turkish8",  "HP-TURKISH8" },
    { "utf8",      "UTF-8" }
#   define alias_table_defined
#  endif
#  if defined __sgi                                         /* IRIX */
    { "ISO8859-1",  "ISO-8859-1" },
    { "ISO8859-15", "ISO-8859-15" },
    { "ISO8859-2",  "ISO-8859-2" },
    { "ISO8859-5",  "ISO-8859-5" },
    { "ISO8859-7",  "ISO-8859-7" },
    { "ISO8859-9",  "ISO-8859-9" },
    { "eucCN",      "GB2312" },
    { "eucJP",      "EUC-JP" },
    { "eucKR",      "EUC-KR" },
    { "eucTW",      "EUC-TW" }
#   define alias_table_defined
#  endif
#  if defined __osf__                                       /* OSF/1 */
  /*{ "GBK",        "GBK" },*/
    { "ISO8859-1",  "ISO-8859-1" },
    { "ISO8859-15", "ISO-8859-15" },
    { "ISO8859-2",  "ISO-8859-2" },
    { "ISO8859-4",  "ISO-8859-4" },
    { "ISO8859-5",  "ISO-8859-5" },
    { "ISO8859-7",  "ISO-8859-7" },
    { "ISO8859-8",  "ISO-8859-8" },
    { "ISO8859-9",  "ISO-8859-9" },
    { "KSC5601",    "CP949" },
    { "SJIS",       "SHIFT_JIS" },
    { "TACTIS",     "TIS-620" },
  /*{ "UTF-8",      "UTF-8" },*/
    { "big5",       "BIG5" },
    { "cp850",      "CP850" },
    { "dechanyu",   "DEC-HANYU" },
    { "dechanzi",   "GB2312" },
    { "deckanji",   "DEC-KANJI" },
    { "deckorean",  "EUC-KR" },
    { "eucJP",      "EUC-JP" },
    { "eucKR",      "EUC-KR" },
    { "eucTW",      "EUC-TW" },
    { "sdeckanji",  "EUC-JP" }
#   define alias_table_defined
#  endif
#  if defined __sun                                         /* Solaris */
    { "5601",        "EUC-KR" },
    { "646",         "ASCII" },
  /*{ "BIG5",        "BIG5" },*/
    { "Big5-HKSCS",  "BIG5-HKSCS" },
    { "GB18030",     "GB18030" },
  /*{ "GBK",         "GBK" },*/
    { "ISO8859-1",   "ISO-8859-1" },
    { "ISO8859-11",  "TIS-620" },
    { "ISO8859-13",  "ISO-8859-13" },
    { "ISO8859-15",  "ISO-8859-15" },
    { "ISO8859-2",   "ISO-8859-2" },
    { "ISO8859-3",   "ISO-8859-3" },
    { "ISO8859-4",   "ISO-8859-4" },
    { "ISO8859-5",   "ISO-8859-5" },
    { "ISO8859-6",   "ISO-8859-6" },
    { "ISO8859-7",   "ISO-8859-7" },
    { "ISO8859-8",   "ISO-8859-8" },
    { "ISO8859-9",   "ISO-8859-9" },
    { "PCK",         "SHIFT_JIS" },
    { "TIS620.2533", "TIS-620" },
  /*{ "UTF-8",       "UTF-8" },*/
    { "ansi-1251",   "CP1251" },
    { "cns11643",    "EUC-TW" },
    { "eucJP",       "EUC-JP" },
    { "gb2312",      "GB2312" },
    { "koi8-r",      "KOI8-R" }
#   define alias_table_defined
#  endif
#  if defined __minix                                       /* Minix */
    { "646", "ASCII" }
#   define alias_table_defined
#  endif
#  if defined WINDOWS_NATIVE || defined __CYGWIN__          /* Windows */
    { "CP1361",  "JOHAB" },
    { "CP20127", "ASCII" },
    { "CP20866", "KOI8-R" },
    { "CP20936", "GB2312" },
    { "CP21866", "KOI8-RU" },
    { "CP28591", "ISO-8859-1" },
    { "CP28592", "ISO-8859-2" },
    { "CP28593", "ISO-8859-3" },
    { "CP28594", "ISO-8859-4" },
    { "CP28595", "ISO-8859-5" },
    { "CP28596", "ISO-8859-6" },
    { "CP28597", "ISO-8859-7" },
    { "CP28598", "ISO-8859-8" },
    { "CP28599", "ISO-8859-9" },
    { "CP28605", "ISO-8859-15" },
    { "CP38598", "ISO-8859-8" },
    { "CP51932", "EUC-JP" },
    { "CP51936", "GB2312" },
    { "CP51949", "EUC-KR" },
    { "CP51950", "EUC-TW" },
    { "CP54936", "GB18030" },
    { "CP65001", "UTF-8" },
    { "CP936",   "GBK" }
#   define alias_table_defined
#  endif
#  if defined OS2                                           /* OS/2 */
    /* The list of encodings is taken from "List of OS/2 Codepages"
       by Alex Taylor:
       <http://altsan.org/os2/toolkits/uls/index.html#codepages>.
       See also "IBM Globalization - Code page identifiers":
       <https://www-01.ibm.com/software/globalization/cp/cp_cpgid.html>.  */
    { "CP1089", "ISO-8859-6" },
    { "CP1208", "UTF-8" },
    { "CP1381", "GB2312" },
    { "CP1386", "GBK" },
    { "CP3372", "EUC-JP" },
    { "CP813",  "ISO-8859-7" },
    { "CP819",  "ISO-8859-1" },
    { "CP878",  "KOI8-R" },
    { "CP912",  "ISO-8859-2" },
    { "CP913",  "ISO-8859-3" },
    { "CP914",  "ISO-8859-4" },
    { "CP915",  "ISO-8859-5" },
    { "CP916",  "ISO-8859-8" },
    { "CP920",  "ISO-8859-9" },
    { "CP921",  "ISO-8859-13" },
    { "CP923",  "ISO-8859-15" },
    { "CP954",  "EUC-JP" },
    { "CP964",  "EUC-TW" },
    { "CP970",  "EUC-KR" }
#   define alias_table_defined
#  endif
#  if defined VMS                                           /* OpenVMS */
    /* The list of encodings is taken from the OpenVMS 7.3-1 documentation
       "Compaq C Run-Time Library Reference Manual for OpenVMS systems"
       section 10.7 "Handling Different Character Sets".  */
    { "DECHANYU",  "DEC-HANYU" },
    { "DECHANZI",  "GB2312" },
    { "DECKANJI",  "DEC-KANJI" },
    { "DECKOREAN", "EUC-KR" },
    { "ISO8859-1", "ISO-8859-1" },
    { "ISO8859-2", "ISO-8859-2" },
    { "ISO8859-5", "ISO-8859-5" },
    { "ISO8859-7", "ISO-8859-7" },
    { "ISO8859-8", "ISO-8859-8" },
    { "ISO8859-9", "ISO-8859-9" },
    { "SDECKANJI", "EUC-JP" },
    { "SJIS",      "SHIFT_JIS" },
    { "eucJP",     "EUC-JP" },
    { "eucTW",     "EUC-TW" }
#   define alias_table_defined
#  endif
#  ifndef alias_table_defined
    /* Just a dummy entry, to avoid a C syntax error.  */
    { "", "" }
#  endif
  };

# endif

#else

/* On these platforms, we use a mapping from locale name to GNU canonical
   encoding name.  */

struct table_entry
{
  const char locale[17+1];
  const char canonical[11+1];
};

/* Table of platform-dependent mappings, sorted in ascending order.  */
static const struct table_entry locale_table[] =
  {
# if defined __FreeBSD__                                    /* FreeBSD 4.2 */
    { "cs_CZ.ISO_8859-2",  "ISO-8859-2" },
    { "da_DK.DIS_8859-15", "ISO-8859-15" },
    { "da_DK.ISO_8859-1",  "ISO-8859-1" },
    { "de_AT.DIS_8859-15", "ISO-8859-15" },
    { "de_AT.ISO_8859-1",  "ISO-8859-1" },
    { "de_CH.DIS_8859-15", "ISO-8859-15" },
    { "de_CH.ISO_8859-1",  "ISO-8859-1" },
    { "de_DE.DIS_8859-15", "ISO-8859-15" },
    { "de_DE.ISO_8859-1",  "ISO-8859-1" },
    { "en_AU.DIS_8859-15", "ISO-8859-15" },
    { "en_AU.ISO_8859-1",  "ISO-8859-1" },
    { "en_CA.DIS_8859-15", "ISO-8859-15" },
    { "en_CA.ISO_8859-1",  "ISO-8859-1" },
    { "en_GB.DIS_8859-15", "ISO-8859-15" },
    { "en_GB.ISO_8859-1",  "ISO-8859-1" },
    { "en_US.DIS_8859-15", "ISO-8859-15" },
    { "en_US.ISO_8859-1",  "ISO-8859-1" },
    { "es_ES.DIS_8859-15", "ISO-8859-15" },
    { "es_ES.ISO_8859-1",  "ISO-8859-1" },
    { "fi_FI.DIS_8859-15", "ISO-8859-15" },
    { "fi_FI.ISO_8859-1",  "ISO-8859-1" },
    { "fr_BE.DIS_8859-15", "ISO-8859-15" },
    { "fr_BE.ISO_8859-1",  "ISO-8859-1" },
    { "fr_CA.DIS_8859-15", "ISO-8859-15" },
    { "fr_CA.ISO_8859-1",  "ISO-8859-1" },
    { "fr_CH.DIS_8859-15", "ISO-8859-15" },
    { "fr_CH.ISO_8859-1",  "ISO-8859-1" },
    { "fr_FR.DIS_8859-15", "ISO-8859-15" },
    { "fr_FR.ISO_8859-1",  "ISO-8859-1" },
    { "hr_HR.ISO_8859-2",  "ISO-8859-2" },
    { "hu_HU.ISO_8859-2",  "ISO-8859-2" },
    { "is_IS.DIS_8859-15", "ISO-8859-15" },
    { "is_IS.ISO_8859-1",  "ISO-8859-1" },
    { "it_CH.DIS_8859-15", "ISO-8859-15" },
    { "it_CH.ISO_8859-1",  "ISO-8859-1" },
    { "it_IT.DIS_8859-15", "ISO-8859-15" },
    { "it_IT.ISO_8859-1",  "ISO-8859-1" },
    { "ja_JP.EUC",         "EUC-JP" },
    { "ja_JP.SJIS",        "SHIFT_JIS" },
    { "ja_JP.Shift_JIS",   "SHIFT_JIS" },
    { "ko_KR.EUC",         "EUC-KR" },
    { "la_LN.ASCII",       "ASCII" },
    { "la_LN.DIS_8859-15", "ISO-8859-15" },
    { "la_LN.ISO_8859-1",  "ISO-8859-1" },
    { "la_LN.ISO_8859-2",  "ISO-8859-2" },
    { "la_LN.ISO_8859-4",  "ISO-8859-4" },
    { "lt_LN.ASCII",       "ASCII" },
    { "lt_LN.DIS_8859-15", "ISO-8859-15" },
    { "lt_LN.ISO_8859-1",  "ISO-8859-1" },
    { "lt_LN.ISO_8859-2",  "ISO-8859-2" },
    { "lt_LT.ISO_8859-4",  "ISO-8859-4" },
    { "nl_BE.DIS_8859-15", "ISO-8859-15" },
    { "nl_BE.ISO_8859-1",  "ISO-8859-1" },
    { "nl_NL.DIS_8859-15", "ISO-8859-15" },
    { "nl_NL.ISO_8859-1",  "ISO-8859-1" },
    { "no_NO.DIS_8859-15", "ISO-8859-15" },
    { "no_NO.ISO_8859-1",  "ISO-8859-1" },
    { "pl_PL.ISO_8859-2",  "ISO-8859-2" },
    { "pt_PT.DIS_8859-15", "ISO-8859-15" },
    { "pt_PT.ISO_8859-1",  "ISO-8859-1" },
    { "ru_RU.CP866",       "CP866" },
    { "ru_RU.ISO_8859-5",  "ISO-8859-5" },
    { "ru_RU.KOI8-R",      "KOI8-R" },
    { "ru_SU.CP866",       "CP866" },
    { "ru_SU.ISO_8859-5",  "ISO-8859-5" },
    { "ru_SU.KOI8-R",      "KOI8-R" },
    { "sl_SI.ISO_8859-2",  "ISO-8859-2" },
    { "sv_SE.DIS_8859-15", "ISO-8859-15" },
    { "sv_SE.ISO_8859-1",  "ISO-8859-1" },
    { "uk_UA.KOI8-U",      "KOI8-U" },
    { "zh_CN.EUC",         "GB2312" },
    { "zh_TW.BIG5",        "BIG5" },
    { "zh_TW.Big5",        "BIG5" }
#  define locale_table_defined
# endif
# if defined __DJGPP__                                      /* DOS / DJGPP 2.03 */
    /* The encodings given here may not all be correct.
       If you find that the encoding given for your language and
       country is not the one your DOS machine actually uses, just
       correct it in this file, and send a mail to
       Juan Manuel Guerrero <juan.guerrero@gmx.de>
       and <bug-gnulib@gnu.org>.  */
    { "C",     "ASCII" },
    { "ar",    "CP864" },
    { "ar_AE", "CP864" },
    { "ar_DZ", "CP864" },
    { "ar_EG", "CP864" },
    { "ar_IQ", "CP864" },
    { "ar_IR", "CP864" },
    { "ar_JO", "CP864" },
    { "ar_KW", "CP864" },
    { "ar_MA", "CP864" },
    { "ar_OM", "CP864" },
    { "ar_QA", "CP864" },
    { "ar_SA", "CP864" },
    { "ar_SY", "CP864" },
    { "be",    "CP866" },
    { "be_BE", "CP866" },
    { "bg",    "CP866" }, /* not CP855 ?? */
    { "bg_BG", "CP866" }, /* not CP855 ?? */
    { "ca",    "CP850" },
    { "ca_ES", "CP850" },
    { "cs",    "CP852" },
    { "cs_CZ", "CP852" },
    { "da",    "CP865" }, /* not CP850 ?? */
    { "da_DK", "CP865" }, /* not CP850 ?? */
    { "de",    "CP850" },
    { "de_AT", "CP850" },
    { "de_CH", "CP850" },
    { "de_DE", "CP850" },
    { "el",    "CP869" },
    { "el_GR", "CP869" },
    { "en",    "CP850" },
    { "en_AU", "CP850" }, /* not CP437 ?? */
    { "en_CA", "CP850" },
    { "en_GB", "CP850" },
    { "en_NZ", "CP437" },
    { "en_US", "CP437" },
    { "en_ZA", "CP850" }, /* not CP437 ?? */
    { "eo",    "CP850" },
    { "eo_EO", "CP850" },
    { "es",    "CP850" },
    { "es_AR", "CP850" },
    { "es_BO", "CP850" },
    { "es_CL", "CP850" },
    { "es_CO", "CP850" },
    { "es_CR", "CP850" },
    { "es_CU", "CP850" },
    { "es_DO", "CP850" },
    { "es_EC", "CP850" },
    { "es_ES", "CP850" },
    { "es_GT", "CP850" },
    { "es_HN", "CP850" },
    { "es_MX", "CP850" },
    { "es_NI", "CP850" },
    { "es_PA", "CP850" },
    { "es_PE", "CP850" },
    { "es_PY", "CP850" },
    { "es_SV", "CP850" },
    { "es_UY", "CP850" },
    { "es_VE", "CP850" },
    { "et",    "CP850" },
    { "et_EE", "CP850" },
    { "eu",    "CP850" },
    { "eu_ES", "CP850" },
    { "fi",    "CP850" },
    { "fi_FI", "CP850" },
    { "fr",    "CP850" },
    { "fr_BE", "CP850" },
    { "fr_CA", "CP850" },
    { "fr_CH", "CP850" },
    { "fr_FR", "CP850" },
    { "ga",    "CP850" },
    { "ga_IE", "CP850" },
    { "gd",    "CP850" },
    { "gd_GB", "CP850" },
    { "gl",    "CP850" },
    { "gl_ES", "CP850" },
    { "he",    "CP862" },
    { "he_IL", "CP862" },
    { "hr",    "CP852" },
    { "hr_HR", "CP852" },
    { "hu",    "CP852" },
    { "hu_HU", "CP852" },
    { "id",    "CP850" }, /* not CP437 ?? */
    { "id_ID", "CP850" }, /* not CP437 ?? */
    { "is",    "CP861" }, /* not CP850 ?? */
    { "is_IS", "CP861" }, /* not CP850 ?? */
    { "it",    "CP850" },
    { "it_CH", "CP850" },
    { "it_IT", "CP850" },
    { "ja",    "CP932" },
    { "ja_JP", "CP932" },
    { "kr",    "CP949" }, /* not CP934 ?? */
    { "kr_KR", "CP949" }, /* not CP934 ?? */
    { "lt",    "CP775" },
    { "lt_LT", "CP775" },
    { "lv",    "CP775" },
    { "lv_LV", "CP775" },
    { "mk",    "CP866" }, /* not CP855 ?? */
    { "mk_MK", "CP866" }, /* not CP855 ?? */
    { "mt",    "CP850" },
    { "mt_MT", "CP850" },
    { "nb",    "CP865" }, /* not CP850 ?? */
    { "nb_NO", "CP865" }, /* not CP850 ?? */
    { "nl",    "CP850" },
    { "nl_BE", "CP850" },
    { "nl_NL", "CP850" },
    { "nn",    "CP865" }, /* not CP850 ?? */
    { "nn_NO", "CP865" }, /* not CP850 ?? */
    { "no",    "CP865" }, /* not CP850 ?? */
    { "no_NO", "CP865" }, /* not CP850 ?? */
    { "pl",    "CP852" },
    { "pl_PL", "CP852" },
    { "pt",    "CP850" },
    { "pt_BR", "CP850" },
    { "pt_PT", "CP850" },
    { "ro",    "CP852" },
    { "ro_RO", "CP852" },
    { "ru",    "CP866" },
    { "ru_RU", "CP866" },
    { "sk",    "CP852" },
    { "sk_SK", "CP852" },
    { "sl",    "CP852" },
    { "sl_SI", "CP852" },
    { "sq",    "CP852" },
    { "sq_AL", "CP852" },
    { "sr",    "CP852" }, /* CP852 or CP866 or CP855 ?? */
    { "sr_CS", "CP852" }, /* CP852 or CP866 or CP855 ?? */
    { "sr_YU", "CP852" }, /* CP852 or CP866 or CP855 ?? */
    { "sv",    "CP850" },
    { "sv_SE", "CP850" },
    { "th",    "CP874" },
    { "th_TH", "CP874" },
    { "tr",    "CP857" },
    { "tr_TR", "CP857" },
    { "uk",    "CP1125" },
    { "uk_UA", "CP1125" },
    { "zh_CN", "GBK" },
    { "zh_TW", "CP950" } /* not CP938 ?? */
#  define locale_table_defined
# endif
# ifndef locale_table_defined
    /* Just a dummy entry, to avoid a C syntax error.  */
    { "", "" }
# endif
  };

#endif


/* Determine the current locale's character encoding, and canonicalize it
   into one of the canonical names listed in localcharset.h.
   The result must not be freed; it is statically allocated.
   If the canonical name cannot be determined, the result is a non-canonical
   name.  */

#ifdef STATIC
STATIC
#endif
const char *
locale_charset (void)
{
  const char *codeset;

#if HAVE_LANGINFO_CODESET || defined WINDOWS_NATIVE || defined OS2

# if HAVE_LANGINFO_CODESET

  /* Most systems support nl_langinfo (CODESET) nowadays.  */
  codeset = nl_langinfo (CODESET);

#  ifdef __CYGWIN__
  /* Cygwin < 1.7 does not have locales.  nl_langinfo (CODESET) always
     returns "US-ASCII".  Return the suffix of the locale name from the
     environment variables (if present) or the codepage as a number.  */
  if (codeset != NULL && strcmp (codeset, "US-ASCII") == 0)
    {
      const char *locale;
      static char buf[2 + 10 + 1];

      locale = getenv ("LC_ALL");
      if (locale == NULL || locale[0] == '\0')
        {
          locale = getenv ("LC_CTYPE");
          if (locale == NULL || locale[0] == '\0')
            locale = getenv ("LANG");
        }
      if (locale != NULL && locale[0] != '\0')
        {
          /* If the locale name contains an encoding after the dot, return
             it.  */
          const char *dot = strchr (locale, '.');

          if (dot != NULL)
            {
              const char *modifier;

              dot++;
              /* Look for the possible @... trailer and remove it, if any.  */
              modifier = strchr (dot, '@');
              if (modifier == NULL)
                return dot;
              if (modifier - dot < sizeof (buf))
                {
                  memcpy (buf, dot, modifier - dot);
                  buf [modifier - dot] = '\0';
                  return buf;
                }
            }
        }

      /* The Windows API has a function returning the locale's codepage as a
         number: GetACP().  This encoding is used by Cygwin, unless the user
         has set the environment variable CYGWIN=codepage:oem (which very few
         people do).
         Output directed to console windows needs to be converted (to
         GetOEMCP() if the console is using a raster font, or to
         GetConsoleOutputCP() if it is using a TrueType font).  Cygwin does
         this conversion transparently (see winsup/cygwin/fhandler_console.cc),
         converting to GetConsoleOutputCP().  This leads to correct results,
         except when SetConsoleOutputCP has been called and a raster font is
         in use.  */
      sprintf (buf, "CP%u", GetACP ());
      codeset = buf;
    }
#  endif

  if (codeset == NULL)
    /* The canonical name cannot be determined.  */
    codeset = "";

# elif defined WINDOWS_NATIVE

  static char buf[2 + 10 + 1];

  /* The Windows API has a function returning the locale's codepage as
     a number, but the value doesn't change according to what the
     'setlocale' call specified.  So we use it as a last resort, in
     case the string returned by 'setlocale' doesn't specify the
     codepage.  */
  char *current_locale = setlocale (LC_ALL, NULL);
  char *pdot;

  /* If they set different locales for different categories,
     'setlocale' will return a semi-colon separated list of locale
     values.  To make sure we use the correct one, we choose LC_CTYPE.  */
  if (strchr (current_locale, ';'))
    current_locale = setlocale (LC_CTYPE, NULL);

  pdot = strrchr (current_locale, '.');
  if (pdot && 2 + strlen (pdot + 1) + 1 <= sizeof (buf))
    sprintf (buf, "CP%s", pdot + 1);
  else
    {
      /* The Windows API has a function returning the locale's codepage as a
        number: GetACP().
        When the output goes to a console window, it needs to be provided in
        GetOEMCP() encoding if the console is using a raster font, or in
        GetConsoleOutputCP() encoding if it is using a TrueType font.
        But in GUI programs and for output sent to files and pipes, GetACP()
        encoding is the best bet.  */
      sprintf (buf, "CP%u", GetACP ());
    }
  codeset = buf;

# elif defined OS2

  const char *locale;
  static char buf[2 + 10 + 1];
  ULONG cp[3];
  ULONG cplen;

  codeset = NULL;

  /* Allow user to override the codeset, as set in the operating system,
     with standard language environment variables.  */
  locale = getenv ("LC_ALL");
  if (locale == NULL || locale[0] == '\0')
    {
      locale = getenv ("LC_CTYPE");
      if (locale == NULL || locale[0] == '\0')
        locale = getenv ("LANG");
    }
  if (locale != NULL && locale[0] != '\0')
    {
      /* If the locale name contains an encoding after the dot, return it.  */
      const char *dot = strchr (locale, '.');

      if (dot != NULL)
        {
          const char *modifier;

          dot++;
          /* Look for the possible @... trailer and remove it, if any.  */
          modifier = strchr (dot, '@');
          if (modifier == NULL)
            return dot;
          if (modifier - dot < sizeof (buf))
            {
              memcpy (buf, dot, modifier - dot);
              buf [modifier - dot] = '\0';
              return buf;
            }
        }

      /* For the POSIX locale, don't use the system's codepage.  */
      if (strcmp (locale, "C") == 0 || strcmp (locale, "POSIX") == 0)
        codeset = "";
    }

  if (codeset == NULL)
    {
      /* OS/2 has a function returning the locale's codepage as a number.  */
      if (DosQueryCp (sizeof (cp), cp, &cplen))
        codeset = "";
      else
        {
          sprintf (buf, "CP%u", cp[0]);
          codeset = buf;
        }
    }

# else

#  error "Add code for other platforms here."

# endif

  /* Resolve alias.  */
  {
# ifdef alias_table_defined
    /* On some platforms, UTF-8 locales are the most frequently used ones.
       Speed up the common case and slow down the less common cases by
       testing for this case first.  */
#  if defined __OpenBSD__ || (defined __APPLE__ && defined __MACH__) || defined __sun || defined __CYGWIN__
    if (strcmp (codeset, "UTF-8") == 0)
      goto done_table_lookup;
    else
#  endif
      {
        const struct table_entry * const table = alias_table;
        size_t const table_size =
          sizeof (alias_table) / sizeof (struct table_entry);
        /* The table is sorted.  Perform a binary search.  */
        size_t hi = table_size;
        size_t lo = 0;
        while (lo < hi)
          {
            /* Invariant:
               for i < lo, strcmp (table[i].alias, codeset) < 0,
               for i >= hi, strcmp (table[i].alias, codeset) > 0.  */
            size_t mid = (hi + lo) >> 1; /* >= lo, < hi */
            int cmp = strcmp (table[mid].alias, codeset);
            if (cmp < 0)
              lo = mid + 1;
            else if (cmp > 0)
              hi = mid;
            else
              {
                /* Found an i with
                     strcmp (table[i].alias, codeset) == 0.  */
                codeset = table[mid].canonical;
                goto done_table_lookup;
              }
          }
      }
    if (0)
      done_table_lookup: ;
    else
# endif
      {
        /* Did not find it in the table.  */
        /* On Mac OS X, all modern locales use the UTF-8 encoding.
           BeOS and Haiku have a single locale, and it has UTF-8 encoding.  */
# if (defined __APPLE__ && defined __MACH__) || defined __BEOS__ || defined __HAIKU__
        codeset = "UTF-8";
# else
        /* Don't return an empty string.  GNU libc and GNU libiconv interpret
           the empty string as denoting "the locale's character encoding",
           thus GNU libiconv would call this function a second time.  */
        if (codeset[0] == '\0')
          codeset = "ASCII";
# endif
      }
  }

#else

  /* On old systems which lack it, use setlocale or getenv.  */
  const char *locale = NULL;

  /* But most old systems don't have a complete set of locales.  Some
     (like DJGPP) have only the C locale.  Therefore we don't use setlocale
     here; it would return "C" when it doesn't support the locale name the
     user has set.  */
# if 0
  locale = setlocale (LC_CTYPE, NULL);
# endif
  if (locale == NULL || locale[0] == '\0')
    {
      locale = getenv ("LC_ALL");
      if (locale == NULL || locale[0] == '\0')
        {
          locale = getenv ("LC_CTYPE");
          if (locale == NULL || locale[0] == '\0')
            locale = getenv ("LANG");
            if (locale == NULL)
              locale = "";
        }
    }

  /* Map locale name to canonical encoding name.  */
  {
# ifdef locale_table_defined
    const struct table_entry * const table = locale_table;
    size_t const table_size =
      sizeof (locale_table) / sizeof (struct table_entry);
    /* The table is sorted.  Perform a binary search.  */
    size_t hi = table_size;
    size_t lo = 0;
    while (lo < hi)
      {
        /* Invariant:
           for i < lo, strcmp (table[i].locale, locale) < 0,
           for i >= hi, strcmp (table[i].locale, locale) > 0.  */
        size_t mid = (hi + lo) >> 1; /* >= lo, < hi */
        int cmp = strcmp (table[mid].locale, locale);
        if (cmp < 0)
          lo = mid + 1;
        else if (cmp > 0)
          hi = mid;
        else
          {
            /* Found an i with
                 strcmp (table[i].locale, locale) == 0.  */
            codeset = table[mid].canonical;
            goto done_table_lookup;
          }
      }
    if (0)
      done_table_lookup: ;
    else
# endif
      {
        /* Did not find it in the table.  */
        /* On Mac OS X, all modern locales use the UTF-8 encoding.
           BeOS and Haiku have a single locale, and it has UTF-8 encoding.  */
# if (defined __APPLE__ && defined __MACH__) || defined __BEOS__ || defined __HAIKU__
        codeset = "UTF-8";
# else
        /* The canonical name cannot be determined.  */
        /* Don't return an empty string.  GNU libc and GNU libiconv interpret
           the empty string as denoting "the locale's character encoding",
           thus GNU libiconv would call this function a second time.  */
        codeset = "ASCII";
# endif
      }
  }

#endif

#ifdef DARWIN7
  /* Mac OS X sets MB_CUR_MAX to 1 when LC_ALL=C, and "UTF-8"
     (the default codeset) does not work when MB_CUR_MAX is 1.  */
  if (strcmp (codeset, "UTF-8") == 0 && MB_CUR_MAX_L (uselocale (NULL)) <= 1)
    codeset = "ASCII";
#endif

  return codeset;
}
