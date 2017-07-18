#include <stddef.h>

#define __cxxabiv1
#define __libcxxabi

namespace {
    namespace {
        class __node;
    }
}

void* operator new(size_t, __node* n) {
    return n;
}

#include "cxa_demangle.inc"
#include "demangle.h"

static char* llvm_demangle_impl(__demangle_tree dmg_tree, char* buf, size_t* n, int* status) {
    if (dmg_tree.__status() != success) {
        if (status)
            *status = dmg_tree.__status();
        return nullptr;
    }

    if (status) {
        *status = success;
    }

    const size_t bs = buf == NULL ? 0 : *n;
    ptrdiff_t sm = dmg_tree.__mangled_name_end_ - dmg_tree.__mangled_name_begin_;
    ptrdiff_t est = sm + 60 * (
                                (dmg_tree.__node_end_ - dmg_tree.__node_begin_) +
                                (dmg_tree.__sub_end_ - dmg_tree.__sub_begin_) +
                                (dmg_tree.__t_end_ - dmg_tree.__t_begin_));

    if ((size_t)est > bs) {
        est = dmg_tree.size() + 1;

        if ((size_t)est > bs) {
            *n = est;

            return nullptr;
        }
    }

    char* e = dmg_tree.__get_demangled_name(buf);

    *e++ = 0;
    *n = e - buf;

    return buf;
}

char* llvm_demangle(const char* sym, char* buf, size_t* n, char* intbuf, size_t intbuflen, int* status) {
    return llvm_demangle_impl(__demangle(sym, intbuf, intbuflen), buf, n, status);
}

size_t llvm_demangle_intbuf_len(const char* sym) {
    const size_t n = strlen(sym);

    return n + 2 * n * sizeof(__node) + 2 * n * sizeof(__node*);
}

char* llvm_demangle_gnu3(const char* org) {
    int status;

    return __cxa_demangleX(org, nullptr, nullptr, &status);
}

char* llvm_demangle_4(const char* org) {
    return llvm_demangle_gnu3(org);
}
