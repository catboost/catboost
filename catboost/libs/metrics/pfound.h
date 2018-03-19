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
    {
    }

    template <class TRelevsType, class TApproxType>
    void AddQuery(const TRelevsType* relevs, const TApproxType* approxes, const ui32* subgroupData, ui32 querySize) {
        TVector<int> qurls(querySize);
        std::iota(qurls.begin(), qurls.end(), 0);
        Sort(qurls.begin(), qurls.end(), [&](int left, int right) -> bool {
            return CompareDocs(approxes[left], relevs[left], approxes[right], relevs[right]);
        });

        double pLook = 1, pFound = 0;
        const ui32 depth = Min<ui32>(querySize, Depth);

        TSet<ui32> subgroupIds;
        for (ui32 position = 0; position < depth; position++) {
            const int docId = qurls[position];
            if (subgroupData != nullptr) {
                const ui32 subgroupId = subgroupData[docId];
                if (subgroupIds.has(subgroupId)) {
                    continue;
                }
                subgroupIds.insert(subgroupId);
            }

            const double pRel =  relevs[docId];
            pFound += pRel * pLook;
            pLook *= (1 - pRel) * Decay;
        }

        Statistic.Error += pFound;
        Statistic.Weight++;
    }

    static double Score(TMetricHolder metric) {
        return metric.Weight > 0 ? metric.Error / metric.Weight : 0;
    }

    TMetricHolder GetMetric() const {
        return Statistic;
    }

private:
    ui32 Depth = -1;
    double Decay = 0.85f;
    TMetricHolder Statistic;
};
