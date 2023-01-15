#pragma once

#if !defined(__FreeBSD__)
size_t strlcpy(char* dst, const char* src, size_t len);

inline size_t strlcat(char* dst, const char* src, size_t len) {
    size_t dstlen = strlen(dst);
    size_t srclen = strlen(src);

    if (dstlen < len) {
        len -= dstlen;
        dst += dstlen;
        strlcpy(dst, src, len);
    }

    return dstlen + srclen;
}
#endif
