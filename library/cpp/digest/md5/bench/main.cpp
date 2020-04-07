#include <library/testing/benchmark/bench.h>
#include <library/cpp/digest/md5/md5.h>

#include <util/generic/xrange.h>

#define MD5_DEF(N)                                                  \
    Y_CPU_BENCHMARK(MD5_##N, iface) {                               \
        char buf[N];                                                \
        for (const auto i : xrange(iface.Iterations())) {           \
            Y_UNUSED(i);                                            \
            Y_DO_NOT_OPTIMIZE_AWAY(MD5().Update(buf, sizeof(buf))); \
        }                                                           \
    }

MD5_DEF(32)
MD5_DEF(64)
MD5_DEF(128)

MD5_DEF(1024)
MD5_DEF(2048)
