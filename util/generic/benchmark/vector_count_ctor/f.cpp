#include "f.h"

#include <library/testing/benchmark/bench.h>

#include <util/generic/vector.h>
#include <util/generic/ptr.h>

void CreateYvector(const size_t size, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        NBench::Clobber();
        TVector<ui8> v(size);
        NBench::Escape(v.data());
        NBench::Clobber();
    }
}

void CreateCarray(const size_t size, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        NBench::Clobber();
        TArrayHolder<ui8> v(new ui8[size]);
        memset(v.Get(), 0, size * sizeof(ui8));
        NBench::Escape(v.Get());
        NBench::Clobber();
    }
}
