#include "shgetc.h"
#include "intscan.h"
#include <inttypes.h>
#include <limits.h>
#include <wctype.h>
#include <wchar.h>

static unsigned long long wcstox(const wchar_t * restrict s,
                                 wchar_t ** restrict p,
                                 int base,
                                 unsigned long long lim)
{
    struct fake_file_t f;
    wchar_t *t = (wchar_t *)s;
    while (iswspace(*t)) t++;
    shinit_wcstring(&f, t);
    unsigned long long y = __intscan(&f, base, 1, lim);
    if (p) {
        size_t cnt = shcnt(&f);
        *p = cnt ? t + cnt : (wchar_t *)s;
    }
    return y;
}

unsigned long long wcstoull(const wchar_t *restrict s,
                            wchar_t **restrict p,
                            int base)
{
    return wcstox(s, p, base, ULLONG_MAX);
}

long long wcstoll(const wchar_t *restrict s, wchar_t **restrict p, int base)
{
    return wcstox(s, p, base, LLONG_MIN);
}

unsigned long wcstoul(const wchar_t *restrict s,
                      wchar_t **restrict p,
                      int base)
{
    return wcstox(s, p, base, ULONG_MAX);
}

long wcstol(const wchar_t *restrict s, wchar_t **restrict p, int base)
{
    return wcstox(s, p, base, 0UL+LONG_MIN);
}

intmax_t wcstoimax(const wchar_t *restrict s,
                   wchar_t **restrict p,
                   int base)
{
    return wcstoll(s, p, base);
}

uintmax_t wcstoumax(const wchar_t *restrict s,
                    wchar_t **restrict p,
                    int base)
{
    return wcstoull(s, p, base);
}
