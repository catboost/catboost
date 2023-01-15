#include <library/testing/benchmark/bench.h>

#include <util/generic/ptr.h>
#include <util/generic/xrange.h>

struct X: public TAtomicRefCount<X> {
};

Y_CPU_BENCHMARK(SimplePtrConstruct, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(TSimpleIntrusivePtr<X>());
    }
}
