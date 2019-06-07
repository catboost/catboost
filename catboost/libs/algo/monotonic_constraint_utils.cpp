#include "monotonic_constraint_utils.h"

namespace {
    void BuildMonotonicLag(
        const TVector<int>& monotonicConstraint,
        TVector<ui32>& monotonicLag,
        TVector<ui32>& nonMonotonicLag
    ) {
        for (ui32 i = 0; i < monotonicConstraint.size(); ++i) {
            if (monotonicConstraint[i] != 0) {
                monotonicLag.push_back(i);
            } else {
                nonMonotonicLag.push_back(i);
            }
        }
    }

    ui32 BuildMask(const TVector<int>& monotonicConstraint) {
        ui32 mask = 0;
        for (ui32 i = 0; i < monotonicConstraint.size(); ++i) {
            if (monotonicConstraint[i] == -1) {
                mask += (1 << i);
            }
        }
        return mask;
    }

    ui32 MutateIntWLag(const TVector<ui32>& lag, const ui32 number) {
        ui32 result = 0;
        for (ui32 i = 0; i < lag.size(); ++i) {
            result += ((number >> i) & 1u) << lag[i];
        }
        return result;
    }
}

TVector<ui32> BuildLinearOrderOnLeafsOfMonotonicSubtree(
    const TVector<int>& treeMonotonicConstraints,
    const ui32 monotonicSubtreeIndex
) {
    TVector<ui32> monotonicLag;
    TVector<ui32> nonMonotonicLag;
    BuildMonotonicLag(treeMonotonicConstraints, monotonicLag, nonMonotonicLag);

    ui32 splitTreeSample = MutateIntWLag(nonMonotonicLag, monotonicSubtreeIndex);

    ui32 mask = BuildMask(treeMonotonicConstraints);

    TVector<ui32> result(1 << monotonicLag.size());
    for (ui32 i = 0; i < result.size(); ++i) {
        result[i] = MutateIntWLag(monotonicLag, i) ^ mask + splitTreeSample;
    }
    return result;
}

void CalcOneDimensionalIsotonicRegression(
    const TVector<double>& values,
    const TVector<double>& weight,
    const TVector<ui32>& indexOrder,
    TVector<double>* solution
) {
    const int size = indexOrder.size();

    TVector<double> activeValues(size, 0);
    TVector<double> activeWeight(size, 0);
    TVector<int> activeIndices(size + 1, 0);

    activeValues[0] = values[indexOrder[0]];
    activeWeight[0] = weight[indexOrder[0]];
    int current = 0;
    activeIndices[0] = -1;
    activeIndices[1] = 0;

    for (int i = 1; i < size; ++i) {
        current += 1;
        activeValues[current] = values[indexOrder[i]];
        activeWeight[current] = weight[indexOrder[i]];
        while (current > 0 && activeValues[current] < activeValues[current - 1]) {
            activeValues[current - 1] = (
                activeValues[current - 1] * activeWeight[current - 1] + activeValues[current] * activeWeight[current]
            ) / (activeWeight[current - 1] + activeWeight[current]);
            activeWeight[current - 1] = activeWeight[current - 1] + activeWeight[current];
            current -= 1;
        }
        activeIndices[current + 1] = i;
    }

    for (int i = 0; i <= current; ++i) {
        for (int l = activeIndices[i] + 1; l <= activeIndices[i + 1]; ++l) {
            (*solution)[indexOrder[l]] = activeValues[i];
        }
    }
}

bool CheckMonotonicity(const TVector<ui32>& indexOrder, const TVector<double>& values) {
    for (ui32 i = 0; i + 1 < indexOrder.size(); ++i) {
        if (values[indexOrder[i]] > values[indexOrder[i + 1]]) {
            return false;
        }
    }
    return true;
}
