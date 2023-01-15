#pragma once

#if !defined(__FreeBSD__)
size_t strlcpy(char* dst, const char* src, size_t len);

size_t strlcat(char* dst, const char* src, size_t len);
#endif
