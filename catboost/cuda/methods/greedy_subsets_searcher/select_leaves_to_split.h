#pragma once

namespace NCatboostCuda {
    void SelectLeavesToSplit(const NCattboconst TPointsSubsets& subsets,
                             TVector<ui32>* leavesToSkip,
                             TVector<ui32>* leavesToSplit);
}
