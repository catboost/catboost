#include "shgetc.h"
#include "floatscan.h"
#include <wchar.h>
#include <wctype.h>

static long double wcstox(const wchar_t * restrict s,
                          wchar_t ** restrict p,
                          int prec)
{
    wchar_t *t = (wchar_t *)s;
    struct fake_file_t f;
    while (iswspace(*t)) t++;
    shinit_wcstring(&f, t);
    long double y = __floatscan(&f, prec, 1);
    if (p) {
        size_t cnt = shcnt(&f);
        *p = cnt ? t + cnt : (wchar_t *)s;
    }
    return y;
}

float wcstof(const wchar_t *restrict s, wchar_t **restrict p)
{
    return wcstox(s, p, 0);
}

double wcstod(const wchar_t *restrict s, wchar_t **restrict p)
{
    return wcstox(s, p, 1);
}

long double wcstold(const wchar_t *restrict s, wchar_t **restrict p)
{
    return wcstox(s, p, 2);
}
