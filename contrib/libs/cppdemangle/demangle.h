#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif
    size_t llvm_demangle_intbuf_len(const char* sym);

    //if !intbuf or intbuflen < llvm_demangle_intbuf_len(sym), intbuf will be allocated on heap
    //if !buf or *n < demangled length, function will return 0, and *n will contain proper length for buf
    //may return sym, if demangling not needed
    char* llvm_demangle(const char* sym, char* buf, size_t* n, char* intbuf, size_t intbuflen, int* status);

    char* llvm_demangle_gnu3(const char* org);

    //may return same pointer
    char* llvm_demangle_4(const char* org);
#if defined(__cplusplus)
}
#endif
