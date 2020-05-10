/*-
* Copyright 1997 Massachusetts Institute of Technology
*
* Permission to use, copy, modify, and distribute this software and
* its documentation for any purpose and without fee is hereby
* granted, provided that both the above copyright notice and this
* permission notice appear in all copies, that both the above
* copyright notice and this permission notice appear in all
* supporting documentation, and that the name of M.I.T. not be used
* in advertising or publicity pertaining to distribution of the
* software without specific, written prior permission.  M.I.T. makes
* no representations about the suitability of this software for any
* purpose.  It is provided "as is" without express or implied
* warranty.
*
* THIS SOFTWARE IS PROVIDED BY M.I.T. ``AS IS''.  M.I.T. DISCLAIMS
* ALL EXPRESS OR IMPLIED WARRANTIES WITH REGARD TO THIS SOFTWARE,
* INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT
* SHALL M.I.T. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
* LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
* USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGE.
*/
#include <util/system/defaults.h>

#include <sys/types.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <util/system/compat.h>   /* stricmp */
#include <util/system/yassert.h>
#include "httpdate.h"
#include <util/datetime/base.h>

static const char *wkdays[] = {
    "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"
};

static const char *months[] = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec"
};

int format_http_date(char buf[], size_t size, time_t when) {
    struct tm tms;
    GmTimeR(&when, &tms);

#ifndef HTTP_DATE_ISO_8601
    return snprintf(buf, size, "%s, %02d %s %04d %02d:%02d:%02d GMT",
        wkdays[tms.tm_wday], tms.tm_mday, months[tms.tm_mon],
        tms.tm_year + 1900, tms.tm_hour, tms.tm_min, tms.tm_sec);
#else /* ISO 8601 */
    return snprintf(buf, size, "%04d%02d%02dT%02d%02d%02d+0000",
        tms.tm_year + 1900, tms.tm_mon + 1, tms.tm_mday,
        tms.tm_hour, tms.tm_min, tms.tm_sec);
#endif
}

char* format_http_date(time_t when, char* buf, size_t buflen) {
    const int len = format_http_date(buf, buflen, when);

    if (len == 0) {
        return nullptr;
    }

    Y_ASSERT(len > 0 && size_t(len) < buflen);

    return buf;
}

TString FormatHttpDate(time_t when) {
    char str[64] = {0};
    format_http_date(str, Y_ARRAY_SIZE(str), when);
    return TString(str);
}
