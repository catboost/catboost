#include "permutation.h"

#include "data_utils.h"

#include <numeric>

void NCatboostCuda::TDataPermutation::FillOrder(TVector<ui32>& order) const {
    if (Index != IdentityPermutationId()) {
        if (DataProvider->MetaInfo.HasGroupId && !DataProvider->ObjectsGrouping->IsTrivial()) {
            GenerateQueryDocsOrder(GetSeed(), BlockSize, DataProvider->ObjectsGrouping->GetNonTrivialGroups(), &order);
        } else {
            Shuffle(GetSeed(), BlockSize, DataProvider->GetObjectCount(), &order);
        }
    } else {
        order.resize(DataProvider->GetObjectCount());
        std::iota(order.begin(), order.end(), 0);
    }
}

void NCatboostCuda::TDataPermutation::FillGroupOrder(TVector<ui32>& groupOrder) const {
    CB_ENSURE_INTERNAL(DataProvider->MetaInfo.HasGroupId, "FillGroupOrder supports only datasets with group ids");
    if (Index != IdentityPermutationId()) {
        Shuffle(GetSeed(), BlockSize, DataProvider->ObjectsGrouping->GetGroupCount(), &groupOrder);
    } else {
        groupOrder.yresize(DataProvider->ObjectsGrouping->GetGroupCount());
        std::iota(groupOrder.begin(), groupOrder.end(), 0);
    }
}

void NCatboostCuda::TDataPermutation::FillInversePermutation(TVector<ui32>& permutation) const {
    TVector<ui32> order;
    FillOrder(order);
    permutation.resize(order.size());
    for (ui32 i = 0; i < order.size(); ++i) {
        permutation[order[i]] = i;
    }
}
