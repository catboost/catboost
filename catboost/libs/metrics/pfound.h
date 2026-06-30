#pragma once

#include "metric_holder.h"
#include "doc_comparator.h"

#include <util/system/types.h>
#include <util/generic/utility.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/set.h>

class TPFoundCalcer {
public:

    explicit TPFoundCalcer(ui32 depth = -1, double decay = 0.85)
        : Depth(depth)
        , Decay(decay)
        , Statistic(2)
    {
    }

    template <bool isExpApprox, bool hasDelta, class TRelevsType, class TApproxType>
    void AddQuery(const TRelevsType* relevs, const TApproxType* approxes, const TApproxType* approxDelta, float queryWeight, const ui32* subgroupData, ui32 querySize) {
        TVector<int> qurls(querySize);
        std::iota(qurls.begin(), qurls.end(), 0);
        StableSort(qurls.begin(), qurls.end(), [&](int left, int right) -> bool {
            if (hasDelta) {
                if (isExpApprox) {
                    return CompareDocs(approxes[left] * approxDelta[left], relevs[left], approxes[right] * approxDelta[right], relevs[right]);
            } else {
                    return CompareDocs(approxes[left] + approxDelta[left], relevs[left], approxes[right] + approxDelta[right], relevs[right]);
                }
            } else {
                return CompareDocs(approxes[left], relevs[left], approxes[right], relevs[right]);
            }
        });

        double pLook = 1, pFound = 0;
        const ui32 depth = Min<ui32>(querySize, Depth);

        TSet<ui32> subgroupIds;
        for (ui32 position = 0; position < depth; position++) {
            const int docId = qurls[position];
            if (subgroupData != nullptr) {
                const ui32 subgroupId = subgroupData[docId];
                if (subgroupIds.contains(subgroupId)) {
                    continue;
                }
                subgroupIds.insert(subgroupId);
            }

            const double pRel =  relevs[docId];
            pFound += pRel * pLook;
            pLook *= (1 - pRel) * Decay;
        }

        Statistic.Stats[0] += queryWeight * pFound;
        Statistic.Stats[1] += queryWeight;
    }

    static double Score(const TMetricHolder& metric) {
        return metric.Stats[1] > 0 ? metric.Stats[0] / metric.Stats[1] : 0;
    }

    TMetricHolder GetMetric() const {
        return Statistic;
    }

private:
    const ui32 Depth = -1;
    const double Decay = 0.85f;
    TMetricHolder Statistic;
};
