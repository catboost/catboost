// ATTN! this is port of FreeBSD LIBC code to Win32 - just for convenience.
// Locale is ignored!!!

/*
 * Powerdog Industries kindly requests feedback from anyone modifying
 * this function:
 *
 * Date: Thu, 05 Jun 1997 23:17:17 -0400
 * From: Kevin Ruddy <kevin.ruddy@powerdog.com>
 * To: James FitzGibbon <james@nexis.net>
 * Subject: Re: Use of your strptime(3) code (fwd)
 *
 * The reason for the "no mod" clause was so that modifications would
 * come back and we could integrate them and reissue so that a wider
 * audience could use it (thereby spreading the wealth).  This has
 * made it possible to get strptime to work on many operating systems.
 * I'm not sure why that's "plain unacceptable" to the FreeBSD team.
 *
 * Anyway, you can change it to "with or without modification" as
 * you see fit.  Enjoy.
 *
 * Kevin Ruddy
 * Powerdog Industries, Inc.
 */
/*
 * Copyright (c) 1994 Powerdog Industries.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer
 *    in the documentation and/or other materials provided with the
 *    distribution.
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgement:
 *      This product includes software developed by Powerdog Industries.
 * 4. The name of Powerdog Industries may not be used to endorse or
 *    promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY POWERDOG INDUSTRIES ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE POWERDOG INDUSTRIES BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <util/system/compat.h>
#include "systime.h"
#ifdef _win32_
    #ifndef lint
        #ifndef NOID
static char copyright[] =
    "@(#) Copyright (c) 1994 Powerdog Industries.  All rights reserved.";
static char sccsid[] = "@(#)strptime.c    0.1 (Powerdog) 94/03/27";
        #endif /* !defined NOID */
    #endif     /* not lint */
    //__FBSDID("$FreeBSD: src/lib/libc/stdtime/strptime.c,v 1.35 2003/11/17 04:19:15 nectar Exp $");

    //#include "namespace.h"
    #include <time.h>
    #include <ctype.h>
    #include <errno.h>
    #include <stdlib.h>
    #include <string.h>
//#include <pthread.h>
//#include "un-namespace.h"
//#include "libc_private.h"

// ******************* #include "timelocal.h" *********************
struct lc_time_T {
    const char* mon[12];
    const char* month[12];
    const char* wday[7];
    const char* weekday[7];
    const char* X_fmt;
    const char* x_fmt;
    const char* c_fmt;
    const char* am;
    const char* pm;
    const char* date_fmt;
    const char* alt_month[12];
    const char* md_order;
    const char* ampm_fmt;
};

// ******************* timelocal.c ******************
/*-
 * Copyright (c) 2001 Alexey Zelkin <phantom@FreeBSD.org>
 * Copyright (c) 1997 FreeBSD Inc.
 * All rights reserved.
*/
static const struct lc_time_T _C_time_locale = {
    {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"},
    {"January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"},
    {"Sun", "Mon", "Tue", "Wed",
     "Thu", "Fri", "Sat"},
    {"Sunday", "Monday", "Tuesday", "Wednesday",
     "Thursday", "Friday", "Saturday"},

    /* X_fmt */
    "%H:%M:%S",

    /*
         * x_fmt
         * Since the C language standard calls for
         * "date, using locale's date format," anything goes.
         * Using just numbers (as here) makes Quakers happier;
         * it's also compatible with SVR4.
         */
    "%m/%d/%y",

    /*
         * c_fmt
         */
    "%a %b %e %H:%M:%S %Y",

    /* am */
    "AM",

    /* pm */
    "PM",

    /* date_fmt */
    "%a %b %e %H:%M:%S %Z %Y",

    /* alt_month
         * Standalone months forms for %OB
         */
    {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"},

    /* md_order
         * Month / day order in dates
         */
    "md",

    /* ampm_fmt
         * To determine 12-hour clock format time (empty, if N/A)
         */
    "%I:%M:%S %p"};

struct lc_time_T*
__get_current_time_locale(void)
{
    return /*(_time_using_locale
                ? &_time_locale
                :*/
        (struct lc_time_T*)&_C_time_locale /*)*/;
}

// ******************* strptime.c *******************
static char* _strptime(const char*, const char*, struct tm*, int*);

    #define asizeof(a) (sizeof(a) / sizeof((a)[0]))

    #if defined(_MSC_VER) && (_MSC_VER >= 1900)
        #define tzname _tzname
    #endif

static char*
_strptime(const char* buf, const char* fmt, struct tm* tm, int* GMTp)
{
    char c;
    const char* ptr;
    int i;
    size_t len = 0;
    int Ealternative, Oalternative;
    struct lc_time_T* tptr = __get_current_time_locale();

    ptr = fmt;
    while (*ptr != 0) {
        if (*buf == 0)
            break;

        c = *ptr++;

        if (c != '%') {
            if (isspace((unsigned char)c))
                while (*buf != 0 && isspace((unsigned char)*buf))
                    ++buf;
            else if (c != *buf++)
                return 0;
            continue;
        }

        Ealternative = 0;
        Oalternative = 0;
    label:
        c = *ptr++;
        switch (c) {
            case 0:
            case '%':
                if (*buf++ != '%')
                    return 0;
                break;

            case '+':
                buf = _strptime(buf, tptr->date_fmt, tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'C':
                if (!isdigit((unsigned char)*buf))
                    return 0;

                /* XXX This will break for 3-digit centuries. */
                len = 2;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }
                if (i < 19)
                    return 0;

                tm->tm_year = i * 100 - 1900;
                break;

            case 'c':
                buf = _strptime(buf, tptr->c_fmt, tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'D':
                buf = _strptime(buf, "%m/%d/%y", tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'E':
                if (Ealternative || Oalternative)
                    break;
                ++Ealternative;
                goto label;

            case 'O':
                if (Ealternative || Oalternative)
                    break;
                ++Oalternative;
                goto label;

            case 'F':
                buf = _strptime(buf, "%Y-%m-%d", tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'R':
                buf = _strptime(buf, "%H:%M", tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'r':
                buf = _strptime(buf, tptr->ampm_fmt, tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'T':
                buf = _strptime(buf, "%H:%M:%S", tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'X':
                buf = _strptime(buf, tptr->X_fmt, tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'x':
                buf = _strptime(buf, tptr->x_fmt, tm, GMTp);
                if (buf == 0)
                    return 0;
                break;

            case 'j':
                if (!isdigit((unsigned char)*buf))
                    return 0;

                len = 3;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }
                if (i < 1 || i > 366)
                    return 0;

                tm->tm_yday = i - 1;
                break;

            case 'M':
            case 'S':
                if (*buf == 0 || isspace((unsigned char)*buf))
                    break;

                if (!isdigit((unsigned char)*buf))
                    return 0;

                len = 2;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }

                if (c == 'M') {
                    if (i > 59)
                        return 0;
                    tm->tm_min = i;
                } else {
                    if (i > 60)
                        return 0;
                    tm->tm_sec = i;
                }

                if (*buf != 0 && isspace((unsigned char)*buf))
                    while (*ptr != 0 && !isspace((unsigned char)*ptr))
                        ++ptr;
                break;

            case 'H':
            case 'I':
            case 'k':
            case 'l':
                /*
             * Of these, %l is the only specifier explicitly
             * documented as not being zero-padded.  However,
             * there is no harm in allowing zero-padding.
             *
             * XXX The %l specifier may gobble one too many
             * digits if used incorrectly.
             */
                if (!isdigit((unsigned char)*buf))
                    return 0;

                len = 2;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }
                if (c == 'H' || c == 'k') {
                    if (i > 23)
                        return 0;
                } else if (i > 12)
                    return 0;

                tm->tm_hour = i;

                if (*buf != 0 && isspace((unsigned char)*buf))
                    while (*ptr != 0 && !isspace((unsigned char)*ptr))
                        ++ptr;
                break;

            case 'p':
                /*
             * XXX This is bogus if parsed before hour-related
             * specifiers.
             */
                len = strlen(tptr->am);
                if (strnicmp(buf, tptr->am, len) == 0) {
                    if (tm->tm_hour > 12)
                        return 0;
                    if (tm->tm_hour == 12)
                        tm->tm_hour = 0;
                    buf += len;
                    break;
                }

                len = strlen(tptr->pm);
                if (strnicmp(buf, tptr->pm, len) == 0) {
                    if (tm->tm_hour > 12)
                        return 0;
                    if (tm->tm_hour != 12)
                        tm->tm_hour += 12;
                    buf += len;
                    break;
                }

                return 0;

            case 'A':
            case 'a':
                for (i = 0; i < asizeof(tptr->weekday); i++) {
                    len = strlen(tptr->weekday[i]);
                    if (strnicmp(buf, tptr->weekday[i],
                                 len) == 0)
                        break;
                    len = strlen(tptr->wday[i]);
                    if (strnicmp(buf, tptr->wday[i],
                                 len) == 0)
                        break;
                }
                if (i == asizeof(tptr->weekday))
                    return 0;

                tm->tm_wday = i;
                buf += len;
                break;

            case 'U':
            case 'W':
                /*
             * XXX This is bogus, as we can not assume any valid
             * information present in the tm structure at this
             * point to calculate a real value, so just check the
             * range for now.
             */
                if (!isdigit((unsigned char)*buf))
                    return 0;

                len = 2;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }
                if (i > 53)
                    return 0;

                if (*buf != 0 && isspace((unsigned char)*buf))
                    while (*ptr != 0 && !isspace((unsigned char)*ptr))
                        ++ptr;
                break;

            case 'w':
                if (!isdigit((unsigned char)*buf))
                    return 0;

                i = *buf - '0';
                if (i > 6)
                    return 0;

                tm->tm_wday = i;

                if (*buf != 0 && isspace((unsigned char)*buf))
                    while (*ptr != 0 && !isspace((unsigned char)*ptr))
                        ++ptr;
                break;

            case 'd':
            case 'e':
                /*
             * The %e specifier is explicitly documented as not
             * being zero-padded but there is no harm in allowing
             * such padding.
             *
             * XXX The %e specifier may gobble one too many
             * digits if used incorrectly.
             */
                if (!isdigit((unsigned char)*buf))
                    return 0;

                len = 2;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }
                if (i > 31)
                    return 0;

                tm->tm_mday = i;

                if (*buf != 0 && isspace((unsigned char)*buf))
                    while (*ptr != 0 && !isspace((unsigned char)*ptr))
                        ++ptr;
                break;

            case 'B':
            case 'b':
            case 'h':
                for (i = 0; i < asizeof(tptr->month); i++) {
                    if (Oalternative) {
                        if (c == 'B') {
                            len = strlen(tptr->alt_month[i]);
                            if (strnicmp(buf,
                                         tptr->alt_month[i],
                                         len) == 0)
                                break;
                        }
                    } else {
                        len = strlen(tptr->month[i]);
                        if (strnicmp(buf, tptr->month[i],
                                     len) == 0)
                            break;
                        len = strlen(tptr->mon[i]);
                        if (strnicmp(buf, tptr->mon[i],
                                     len) == 0)
                            break;
                    }
                }
                if (i == asizeof(tptr->month))
                    return 0;

                tm->tm_mon = i;
                buf += len;
                break;

            case 'm':
                if (!isdigit((unsigned char)*buf))
                    return 0;

                len = 2;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }
                if (i < 1 || i > 12)
                    return 0;

                tm->tm_mon = i - 1;

                if (*buf != 0 && isspace((unsigned char)*buf))
                    while (*ptr != 0 && !isspace((unsigned char)*ptr))
                        ++ptr;
                break;

            case 's': {
                char* cp;
                int sverrno;
                long n;
                time_t t;

                sverrno = errno;
                errno = 0;
                n = strtol(buf, &cp, 10);
                if (errno == ERANGE || (long)(t = n) != n) {
                    errno = sverrno;
                    return 0;
                }
                errno = sverrno;
                buf = cp;
                GmTimeR(&t, tm);
                *GMTp = 1;
            } break;

            case 'Y':
            case 'y':
                if (*buf == 0 || isspace((unsigned char)*buf))
                    break;

                if (!isdigit((unsigned char)*buf))
                    return 0;

                len = (c == 'Y') ? 4 : 2;
                for (i = 0; len && *buf != 0 && isdigit((unsigned char)*buf); buf++) {
                    i *= 10;
                    i += *buf - '0';
                    --len;
                }
                if (c == 'Y')
                    i -= 1900;
                if (c == 'y' && i < 69)
                    i += 100;
                if (i < 0)
                    return 0;

                tm->tm_year = i;

                if (*buf != 0 && isspace((unsigned char)*buf))
                    while (*ptr != 0 && !isspace((unsigned char)*ptr))
                        ++ptr;
                break;

            case 'Z': {
                const char* cp;
                char* zonestr;

                for (cp = buf; *cp && isupper((unsigned char)*cp); ++cp) { /*empty*/
                }
                if (cp - buf) {
                    zonestr = (char*)alloca(cp - buf + 1);
                    strncpy(zonestr, buf, cp - buf);
                    zonestr[cp - buf] = '\0';
                    tzset();
                    if (0 == strcmp(zonestr, "GMT")) {
                        *GMTp = 1;
                    } else if (0 == strcmp(zonestr, tzname[0])) {
                        tm->tm_isdst = 0;
                    } else if (0 == strcmp(zonestr, tzname[1])) {
                        tm->tm_isdst = 1;
                    } else {
                        return 0;
                    }
                    buf += cp - buf;
                }
            } break;
        }
    }
    return (char*)buf;
}

char* strptime(const char* buf, const char* fmt, struct tm* tm)
{
    char* ret;
    int gmt;

    gmt = 0;
    ret = _strptime(buf, fmt, tm, &gmt);
    if (ret && gmt) {
        time_t t = timegm(tm);
        localtime_r(&t, tm);
    }

    return (ret);
}
#endif //_win32_
