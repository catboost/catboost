#pragma once

#include <stddef.h>
#include <ctype.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__FreeBSD__) && !defined(__APPLE__)
size_t strlcpy(char* dst, const char* src, size_t len);
size_t strlcat(char* dst, const char* src, size_t len);
#endif

#if (!defined(__linux__) && !defined(__FreeBSD__) && !defined(__APPLE__)) || (defined(__ANDROID__) && __ANDROID_API__ < 21)
char* stpcpy(char* dst, const char* src);
#endif

#if !defined(_MSC_VER)

#define stricmp strcasecmp
#define strnicmp strncasecmp

char* strlwr(char*);
char* strupr(char*);

#else // _MSC_VER

#define strcasecmp stricmp
#define strncasecmp strnicmp

char* strcasestr(const char* s1, const char* s2);
char* strsep(char** stringp, const char* delim);

#endif // _MSC_VER

#if defined(_MSC_VER) || defined(__APPLE__)
void* memrchr(const void* s, int c, size_t n);
#endif

#ifdef __cplusplus
} //extern "C"
#endif
