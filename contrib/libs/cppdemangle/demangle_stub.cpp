#include "demangle.h"

#include <string.h>

size_t llvm_demangle_intbuf_len(const char*) {
    return 0;
}

char* llvm_demangle(const char* sym, char* buf, size_t* n, char* intbuf, size_t intbuflen, int* status) {
    (void)buf;
    (void)n;
    (void)intbuf;
    (void)intbuflen;

    if (status) {
        *status = 0;
    }

    return (char*)sym;
}

char* llvm_demangle_gnu3(const char* org) {
    return strdup(org);
}

char* llvm_demangle_4(const char* org) {
    return const_cast<char*>(org);
}
