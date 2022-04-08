#include "output.h"
#include "printf.h"

#include <util/generic/scope.h>
#include <util/memory/tempbuf.h>

size_t Printf(IOutputStream& out, const char* fmt, ...) {
    va_list lst;
    va_start(lst, fmt);

    Y_DEFER {
        va_end(lst);
    };

    return Printf(out, fmt, lst);
}

static inline size_t TryPrintf(void* ptr, size_t len, IOutputStream& out, const char* fmt, va_list params) {
    va_list lst;
    va_copy(lst, params);
    const int ret = vsnprintf((char*)ptr, len, fmt, lst);
    va_end(lst);

    if (ret < 0) {
        return len;
    }

    if ((size_t)ret < len) {
        out.Write(ptr, (size_t)ret);
    }

    return (size_t)ret;
}

size_t Printf(IOutputStream& out, const char* fmt, va_list params) {
    size_t guess = 0;

    while (true) {
        TTempBuf tmp(guess);
        const size_t ret = TryPrintf(tmp.Data(), tmp.Size(), out, fmt, params);

        if (ret < tmp.Size()) {
            return ret;
        }

        guess = Max(tmp.Size() * 2, ret + 1);
    }

    return 0;
}
