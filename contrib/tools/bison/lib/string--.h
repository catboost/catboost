#pragma once

#include <string.h>

#if defined(_WIN32)
void *rawmemchr(const void *s, int c);
char *stpcpy(char *dest, const char *src);
#endif

int strverscmp(const char *s1, const char *s2);
