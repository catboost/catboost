#include <library/cpp/malloc/api/malloc.h>

using namespace NMalloc;

#if defined(_MSC_VER)
TMallocInfo NMalloc::MallocInfo() {
    TMallocInfo r;
    r.Name = "jemalloc";
    return r;
}
#else
#include <strings.h>
#include <stdlib.h>
#include <inttypes.h>

#include <contrib/libs/jemalloc/include/jemalloc/jemalloc.h>

namespace {
    static bool JESetParam(const char* param, const char*) {
        if (param) {
            if (strcmp(param, "j:reset_epoch") == 0) {
                uint64_t epoch = 1;
                size_t sz = sizeof(epoch);

                mallctl("epoch", &epoch, &sz, &epoch, sz);

                return true;
            }

            return false;
        }

        return false;
    }

    const char* JEGetParam(const char* param) {
        if (param) {
            if (strcmp(param, "allocated") == 0) {
                JESetParam("j:reset_epoch", nullptr);

                size_t allocated = 0;
                size_t sz = sizeof(allocated);

                mallctl("stats.allocated", &allocated, &sz, nullptr, 0);

                static_assert(sizeof(size_t) == sizeof(void*), "fix me");

                return (const char*)(void*)allocated;
            }

            return nullptr;
        }

        return nullptr;
    }
}

TMallocInfo NMalloc::MallocInfo() {
    TMallocInfo r;
    r.Name = "jemalloc";
    r.SetParam = JESetParam;
    r.GetParam = JEGetParam;
    return r;
}
#endif
