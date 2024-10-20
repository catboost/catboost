/* nl_langinfo() replacement: query locale dependent information.

   Copyright (C) 2007-2016 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <config.h>

/* Specification.  */
#include <langinfo.h>

#include <locale.h>
#include <string.h>
#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
# define WIN32_LEAN_AND_MEAN  /* avoid including junk */
# include <windows.h>
# include <stdio.h>
#endif

/* Return the codeset of the current locale, if this is easily deducible.
   Otherwise, return "".  */
static char *
ctype_codeset (void)
{
  static char buf[2 + 10 + 1];
  char const *locale = setlocale (LC_CTYPE, NULL);
  char *codeset = buf;
  size_t codesetlen;
  codeset[0] = '\0';

  if (locale && locale[0])
    {
      /* If the locale name contains an encoding after the dot, return it.  */
      char *dot = strchr (locale, '.');

      if (dot)
        {
          /* Look for the possible @... trailer and remove it, if any.  */
          char *codeset_start = dot + 1;
          char const *modifier = strchr (codeset_start, '@');

          if (! modifier)
            codeset = codeset_start;
          else
            {
              codesetlen = modifier - codeset_start;
              if (codesetlen < sizeof buf)
                {
                  codeset = memcpy (buf, codeset_start, codesetlen);
                  codeset[codesetlen] = '\0';
                }
            }
        }
    }

#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
  /* If setlocale is successful, it returns the number of the
     codepage, as a string.  Otherwise, fall back on Windows API
     GetACP, which returns the locale's codepage as a number (although
     this doesn't change according to what the 'setlocale' call specified).
     Either way, prepend "CP" to make it a valid codeset name.  */
  codesetlen = strlen (codeset);
  if (0 < codesetlen && codesetlen < sizeof buf - 2)
    memmove (buf + 2, codeset, codesetlen + 1);
  else
    sprintf (buf + 2, "%u", GetACP ());
  codeset = memcpy (buf, "CP", 2);
#endif
  return codeset;
}


#if REPLACE_NL_LANGINFO

/* Override nl_langinfo with support for added nl_item values.  */

# undef nl_langinfo

char *
rpl_nl_langinfo (nl_item item)
{
  switch (item)
    {
# if GNULIB_defined_CODESET
    case CODESET:
      return ctype_codeset ();
# endif
# if GNULIB_defined_T_FMT_AMPM
    case T_FMT_AMPM:
      return (char *) "%I:%M:%S %p";
# endif
# if GNULIB_defined_ERA
    case ERA:
      /* The format is not standardized.  In glibc it is a sequence of strings
         of the form "direction:offset:start_date:end_date:era_name:era_format"
         with an empty string at the end.  */
      return (char *) "";
    case ERA_D_FMT:
      /* The %Ex conversion in strftime behaves like %x if the locale does not
         have an alternative time format.  */
      item = D_FMT;
      break;
    case ERA_D_T_FMT:
      /* The %Ec conversion in strftime behaves like %c if the locale does not
         have an alternative time format.  */
      item = D_T_FMT;
      break;
    case ERA_T_FMT:
      /* The %EX conversion in strftime behaves like %X if the locale does not
         have an alternative time format.  */
      item = T_FMT;
      break;
    case ALT_DIGITS:
      /* The format is not standardized.  In glibc it is a sequence of 10
         strings, appended in memory.  */
      return (char *) "\0\0\0\0\0\0\0\0\0\0";
# endif
# if GNULIB_defined_YESEXPR || !FUNC_NL_LANGINFO_YESEXPR_WORKS
    case YESEXPR:
      return (char *) "^[yY]";
    case NOEXPR:
      return (char *) "^[nN]";
# endif
    default:
      break;
    }
  return nl_langinfo (item);
}

#else

/* Provide nl_langinfo from scratch, either for native MS-Windows, or
   for old Unix platforms without locales, such as Linux libc5 or
   BeOS.  */

# include <time.h>

char *
nl_langinfo (nl_item item)
{
  static char nlbuf[100];
  struct tm tmm = { 0 };

  switch (item)
    {
    /* nl_langinfo items of the LC_CTYPE category */
    case CODESET:
      {
        char *codeset = ctype_codeset ();
        if (*codeset)
          return codeset;
      }
# ifdef __BEOS__
      return (char *) "UTF-8";
# else
      return (char *) "ISO-8859-1";
# endif
    /* nl_langinfo items of the LC_NUMERIC category */
    case RADIXCHAR:
      return localeconv () ->decimal_point;
    case THOUSEP:
      return localeconv () ->thousands_sep;
    /* nl_langinfo items of the LC_TIME category.
       TODO: Really use the locale.  */
    case D_T_FMT:
    case ERA_D_T_FMT:
      return (char *) "%a %b %e %H:%M:%S %Y";
    case D_FMT:
    case ERA_D_FMT:
      return (char *) "%m/%d/%y";
    case T_FMT:
    case ERA_T_FMT:
      return (char *) "%H:%M:%S";
    case T_FMT_AMPM:
      return (char *) "%I:%M:%S %p";
    case AM_STR:
      if (!strftime (nlbuf, sizeof nlbuf, "%p", &tmm))
        return (char *) "AM";
      return nlbuf;
    case PM_STR:
      tmm.tm_hour = 12;
      if (!strftime (nlbuf, sizeof nlbuf, "%p", &tmm))
        return (char *) "PM";
      return nlbuf;
    case DAY_1:
    case DAY_2:
    case DAY_3:
    case DAY_4:
    case DAY_5:
    case DAY_6:
    case DAY_7:
      {
        static char const days[][sizeof "Wednesday"] = {
          "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
          "Friday", "Saturday"
        };
        tmm.tm_wday = item - DAY_1;
        if (!strftime (nlbuf, sizeof nlbuf, "%A", &tmm))
          return (char *) days[item - DAY_1];
        return nlbuf;
      }
    case ABDAY_1:
    case ABDAY_2:
    case ABDAY_3:
    case ABDAY_4:
    case ABDAY_5:
    case ABDAY_6:
    case ABDAY_7:
      {
        static char const abdays[][sizeof "Sun"] = {
          "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"
        };
        tmm.tm_wday = item - ABDAY_1;
        if (!strftime (nlbuf, sizeof nlbuf, "%a", &tmm))
          return (char *) abdays[item - ABDAY_1];
        return nlbuf;
      }
    case MON_1:
    case MON_2:
    case MON_3:
    case MON_4:
    case MON_5:
    case MON_6:
    case MON_7:
    case MON_8:
    case MON_9:
    case MON_10:
    case MON_11:
    case MON_12:
      {
        static char const months[][sizeof "September"] = {
          "January", "February", "March", "April", "May", "June", "July",
          "September", "October", "November", "December"
        };
        tmm.tm_mon = item - MON_1;
        if (!strftime (nlbuf, sizeof nlbuf, "%B", &tmm))
          return (char *) months[item - MON_1];
        return nlbuf;
      }
    case ABMON_1:
    case ABMON_2:
    case ABMON_3:
    case ABMON_4:
    case ABMON_5:
    case ABMON_6:
    case ABMON_7:
    case ABMON_8:
    case ABMON_9:
    case ABMON_10:
    case ABMON_11:
    case ABMON_12:
      {
        static char const abmonths[][sizeof "Jan"] = {
          "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
          "Sep", "Oct", "Nov", "Dec"
        };
        tmm.tm_mon = item - ABMON_1;
        if (!strftime (nlbuf, sizeof nlbuf, "%b", &tmm))
          return (char *) abmonths[item - ABMON_1];
        return nlbuf;
      }
    case ERA:
      return (char *) "";
    case ALT_DIGITS:
      return (char *) "\0\0\0\0\0\0\0\0\0\0";
    /* nl_langinfo items of the LC_MONETARY category.  */
    case CRNCYSTR:
      return localeconv () ->currency_symbol;
#if 0
    case INT_CURR_SYMBOL:
      return localeconv () ->int_curr_symbol;
    case MON_DECIMAL_POINT:
      return localeconv () ->mon_decimal_point;
    case MON_THOUSANDS_SEP:
      return localeconv () ->mon_thousands_sep;
    case MON_GROUPING:
      return localeconv () ->mon_grouping;
    case POSITIVE_SIGN:
      return localeconv () ->positive_sign;
    case NEGATIVE_SIGN:
      return localeconv () ->negative_sign;
    case FRAC_DIGITS:
      return & localeconv () ->frac_digits;
    case INT_FRAC_DIGITS:
      return & localeconv () ->int_frac_digits;
    case P_CS_PRECEDES:
      return & localeconv () ->p_cs_precedes;
    case N_CS_PRECEDES:
      return & localeconv () ->n_cs_precedes;
    case P_SEP_BY_SPACE:
      return & localeconv () ->p_sep_by_space;
    case N_SEP_BY_SPACE:
      return & localeconv () ->n_sep_by_space;
    case P_SIGN_POSN:
      return & localeconv () ->p_sign_posn;
    case N_SIGN_POSN:
      return & localeconv () ->n_sign_posn;
#endif
    /* nl_langinfo items of the LC_MESSAGES category
       TODO: Really use the locale. */
    case YESEXPR:
      return (char *) "^[yY]";
    case NOEXPR:
      return (char *) "^[nN]";
    default:
      return (char *) "";
    }
}

#endif
