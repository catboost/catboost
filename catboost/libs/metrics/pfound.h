#pragma once

#include "metric_holder.h"
#include <util/system/types.h>
#include <util/generic/utility.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/set.h>

class TPFoundCalcer {
public:

    explicit TPFoundCalcer(ui32 depth = -1,
                           double decay = 0.85)
    : Depth(depth)
    , Decay(decay) {
    }

    template <class TFloatType>
    void AddQuery(const TFloatType* relevs, const TFloatType* approxes, const ui32* groupData, ui32 querySize) {
        TVector<int> qurls(querySize);
        std::iota(qurls.begin(), qurls.end(), 0);
        Sort(qurls.begin(), qurls.end(), [&](int left, int right) -> bool {
            return approxes[left] != approxes[right] ? approxes[left] > approxes[right] : relevs[left] < relevs[right];
        });

        double pLook = 1, pFound = 0;
        const ui32 depth = Min<ui32>(querySize, Depth);

        TSet<ui32> groupIds;
        for (ui32 position = 0; position < depth; position++) {
            const int docId = qurls[position];
            const ui32 gid = groupData[docId];
            if (groupData && groupIds.has(gid)) {
                continue;
            }

            if (groupData) {
                groupIds.insert(gid);
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
