#pragma once

#include "exception.h"
#include "restorable_rng.h"

#include <util/generic/hash_set.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <cmath>
#include <numeric>


namespace NCB {

    /*
     * Sample k element indices without repetition from {0,1,...,n-1}
     */
    template <class T>
    TVector<T> SampleIndices(size_t n, size_t k, TRestorableFastRng64* rand) {
        CB_ENSURE_INTERNAL(n >= k, "SampleIndices: k=" << k << " > n=" << n);

        TVector<T> result;

        if (n == k) {
            result.yresize(k);
            std::iota(result.begin(), result.end(), T(0));
        } else if (k > 1 && k > (n / std::log2(k))) {
            result.yresize(n);
            std::iota(result.begin(), result.end(), T(0));
            for (auto i : xrange(k)) {
                std::swap(result[i], result[rand->Uniform(i, n)]);
            }
            result.resize(k);
        } else {
            THashSet<T> sampleSet;
            while (sampleSet.size() < k) {
                sampleSet.insert(rand->Uniform((T)n));
            }
            result.assign(sampleSet.begin(), sampleSet.end());
        }

        return result;
    }

}
