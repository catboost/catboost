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
    bool JESetParam(const char* param, const char* value) {
        if (param) {
            if (strcmp(param, "j:reset_epoch") == 0) {
                uint64_t epoch = 1;
                size_t sz = sizeof(epoch);

                mallctl("epoch", &epoch, &sz, &epoch, sz);

                return true;
            }

            if (strcmp(param, "j:prof") == 0) {
                if (strcmp(value, "start") == 0) {
                    bool is_active = true;
                    const int ret = mallctl("prof.active", nullptr, nullptr, &is_active, sizeof(is_active));
                    return ret == 0;
                }
                if (strcmp(value, "stop") == 0) {
                    bool is_active = false;
                    const int ret = mallctl("prof.active", nullptr, nullptr, &is_active, sizeof(is_active));
                    return ret == 0;
                }
                if (strcmp(value, "dump") == 0) {
                    const int ret = mallctl("prof.dump", nullptr, nullptr, nullptr, 0);
                    return ret == 0;
                }
            }
            if (strcmp(param, "j:bg_threads") == 0) {
                if (strcmp(value, "start") == 0) {
                    bool is_active = true;
                    const int ret = mallctl("background_thread", nullptr, nullptr, &is_active, sizeof(is_active));
                    return ret == 0;
                }
                if (strcmp(value, "stop") == 0) {
                    bool is_active = false;
                    // NOTE: joins bg thread
                    const int ret = mallctl("background_thread", nullptr, nullptr, &is_active, sizeof(is_active));
                    return ret == 0;
                }
                if (strncmp(value, "max=", 4) == 0) {
                    int num_value = atoi(value + 4);
                    if (num_value <= 0) {
                        return false;
                    }
                    size_t max_threads = num_value;
                    const int ret = mallctl("max_background_threads", nullptr, nullptr, &max_threads, sizeof(max_threads));
                    return ret == 0;
                }
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
            } else if (strcmp(param, "j:stats_print_func") == 0) {
                return (const char*)&malloc_stats_print;
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
