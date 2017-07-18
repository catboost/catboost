#include <string.h>
#include <wchar.h>
#include <wctype.h>
#include <assert.h>

// Returns 1 if 'wc' is in the 'delim' string, 0 otherwise.
static int _wc_indelim(wchar_t wc, const wchar_t* delim) {
    while (*delim) {
        if (wc == *delim)
            return 1;
        delim++;
    }
    return 0;
}

wchar_t *wcpcpy(wchar_t *to, const wchar_t *from) {
    size_t n = 0;
    for (;;) {
        wchar_t wc = from[n];
        to[n] = wc;
        if (wc == L'\0')
            break;
        n++;
    }
    return to + n;
}

wchar_t *wcpncpy(wchar_t *dst, const wchar_t *src, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        wchar_t wc = src[i];
        dst[i] = wc;
        if (wc == L'\0')
            break;
    }
    while (i < n) {
        dst[i] = L'\0';
        ++i;
    }
    return &dst[n-1];
}

int wcscasecmp(const wchar_t *s1, const wchar_t *s2) {
    size_t n = 0;
    for (;;) {
        wchar_t wc1 = towlower(s1[n]);
        wchar_t wc2 = towlower(s2[n]);
        if (wc1 != wc2)
            return (wc1 > wc2) ? +1 : -1;
        if (wc1 == L'\0')
            return 0;
        n++;
    }
}

wchar_t *wcscat(wchar_t *s1, const wchar_t *s2) {
    size_t n = 0;
    while (s1[n] != L'\0')
        n++;

    size_t i = 0;
    for (;;) {
        wchar_t wc = s2[i];
        s1[n+i] = wc;
        if (wc == L'\0')
            break;
        i++;
    }
    return s1;
}

size_t wcslcat(wchar_t *dst, const wchar_t *src, size_t siz) {
    // Sanity check simplifies code below
    if (siz == 0)
        return 0;

    // Skip dst characters.
    size_t n = 0;
    while (n < siz && dst[n] != L'\0')
      n++;

    if (n == siz)
      return n + wcslen(src);

    // Copy as much source characters as they fit into siz-1 bytes.
    size_t i;
    for (i = 0; n+i+1 < siz && src[i] != L'\0'; ++i)
        dst[n+i] = src[i];

    // Always zero terminate destination
    dst[n+i] = L'\0';

    // Skip remaining source characters
    while (src[i] != L'\0')
        i++;

    return n+i;
}

size_t wcslcpy(wchar_t *dst, const wchar_t *src, size_t siz) {
    size_t i;

    // Copy all non-zero bytes that fit into siz-1 destination bytes
    for (i = 0; i + 1 < siz && src[i] != L'\0'; ++i)
        dst[i] = src[i];

    // Always zero-terminate destination buffer
    dst[i] = L'\0';

    // Skip other source characters.
    while (src[i] != L'\0')
        ++i;

    return i;
}

size_t wcslen(const wchar_t *s) {
    size_t n = 0;
    for (;;) {
        wchar_t wc = s[n];
        if (wc == L'\0')
            return n;
        n++;
    }
}

int wcsncasecmp(const wchar_t *s1, const wchar_t *s2, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        wchar_t wc1 = towlower(s1[i]);
        wchar_t wc2 = towlower(s2[i]);
        if (wc1 != wc2)
            return (wc1 > wc2) ? +1 : -1;
    }
    return 0;
}

wchar_t *wcsncat(wchar_t *s1, const wchar_t *s2, size_t n) {
    size_t start = 0;
    while (s1[start] != L'\0')
        start++;

    // Append s2.
    size_t i;
    for (i = 0; i < n; ++i) {
        wchar_t wc = s2[i];
        s1[start + i] = wc;
        if (wc == L'\0')
            break;
    }
    return (wchar_t*)s1;
}

int wcsncmp(const wchar_t *s1, const wchar_t *s2, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        wchar_t wc = s1[i];
        if (wc != s2[i])
            return (wc > s2[i]) ? +1 : -1;
        if (wc == L'\0')
            break;
    }
    return 0;
}

wchar_t *wcsncpy(wchar_t *dst, const wchar_t *src, size_t n) {
    // Copy initial characters.
    size_t i;
    for (i = 0; i < n; ++i) {
        wchar_t wc = src[i];
        if (wc == L'\0')
            break;
        dst[i] = wc;
    }
    // zero-pad the remainder.
    for ( ; i < n; ++i)
        dst[i] = L'\0';

    return dst;
}

size_t wcsnlen(const wchar_t *s, size_t maxlen) {
    size_t n;
    for (n = 0; n < maxlen; ++n) {
        if (s[n] == L'\0')
            break;
    }
    return n;
}

wchar_t *wcspbrk(const wchar_t *s, const wchar_t *set) {
    size_t n = 0;
    for (;;) {
        wchar_t wc = s[n];
        if (!wc)
            return NULL;
        if (_wc_indelim(wc, set))
            return (wchar_t*)&s[n];
        n++;
    }
}

wchar_t *wcschr(const wchar_t *s, wchar_t c) {
  size_t n = 0;
  for (;;) {
    wchar_t wc = s[n];
    if (wc == c)
      return (wchar_t*)s + n;
    if (wc == L'\0')
      return NULL;
    n++;
  }
}

wchar_t *wcsrchr(const wchar_t *s, wchar_t c) {
    size_t n = 0;
    wchar_t* last = NULL;
    for (;;) {
        wchar_t wc = s[n];
        if (wc == c)
            last = (wchar_t*)s + n;
        if (wc == L'\0')
            break;
        n++;
    }
    return last;
}

size_t wcsspn(const wchar_t *s, const wchar_t *set) {
    size_t n = 0;
    for (;;) {
        wchar_t wc = s[n];
        if (wc == L'\0')
            break;
        if (!_wc_indelim(wc, set))
            break;
        ++n;
    }
    return n;
}

wchar_t *wcsstr(const wchar_t *s, const wchar_t *find) {
    wchar_t find_c;

    // Always find the empty string
    find_c = *find++;
    if (!find_c)
        return (wchar_t*)s;

    size_t find_len = wcslen(find);

    for (;;) {
        wchar_t* p = wcschr(s, find_c);
        if (p == NULL)
            return NULL;

        if (!wmemcmp(p, find, find_len))
            return p;

        s = p + 1;
    }
    return NULL;
}

wchar_t *wcstok(wchar_t *s, const wchar_t *delim, wchar_t **last) {
    if (s == NULL) {
        s = *last;
        if (s == NULL)
            return NULL;
    }

    // Skip leading delimiters first.
    size_t i = 0;
    wchar_t wc;
    for (;;) {
        wc = s[i];
        if (wc && _wc_indelim(wc, delim)) {
            i++;
            continue;
        }
        break;
    }

    if (!wc) {
        // Nothing left.
        *last = NULL;
        return NULL;
    }

    size_t tok_start = i;

    // Skip non delimiters now.
    for (;;) {
        wc = s[i];
        if (wc && !_wc_indelim(wc, delim)) {
            i++;
            continue;
        }
        break;
    }

    if (!wc) {
        *last = NULL;
    } else {
        s[i] = L'\0';
        *last = &s[i+1];
    }
    return &s[tok_start];
}

wchar_t * wmemchr(const wchar_t *s, wchar_t c, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (s[i] == c)
            return (wchar_t*)&s[i];
    }
    return NULL;
}

int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (s1[i] == s2[i])
            continue;
        if (s1[i] > s2[i])
            return 1;
        else
            return -1;
    }
    return 0;
}

wchar_t * wmemcpy(wchar_t *d, const wchar_t *s, size_t n) {
    return (wchar_t *)memcpy((char*)d,
                             (const char*)s,
                             n * sizeof(wchar_t));
}

wchar_t* wmemmove(wchar_t* d, const wchar_t* s, size_t n) {
    return (wchar_t* )memmove((char*)d,
                              (const char*)s,
                              n * sizeof(wchar_t));
}

wchar_t* wmemset(wchar_t* s, wchar_t c, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i)
        s[i] = c;
    return s;
}
