#pragma once

#include <library/cpp/containers/2d_array/2d_array.h>

#include <util/generic/is_in.h>
#include <util/generic/vector.h>

namespace NPar {
    struct TDistributionTreesData {
        TVector<ui16> Tree100K, Tree1M, Tree10M;
        TVector<ui16> TreeBranch6;
    };

    void BuildDistributionTree(TDistributionTreesData* res, const TArray2D<TVector<float>>& delayMatrixData);
    void GenerateSubtasks(const TVector<ui16>& src, TVector<TVector<ui16>>* subTasks);
    int SelectRandomHost(const TVector<ui16>& res);
    void ProjectExecPlan(TVector<ui16>* res, const TVector<bool>& selectedComps);

    inline void AddCompToSelectedList(TVector<bool>* res, int n) {
        if (n >= res->ysize())
            res->resize(n + 1, false);
        (*res)[n] = true;
    }
    void GetSelectedCompList(TVector<bool>* res, const TVector<ui16>& plan);

    // add as a leaf to the top node
    inline void AddCompToPlan(TVector<ui16>* plan, int compId) {
        if (!IsIn(*plan, compId))
            plan->push_back(compId);
    }
}
