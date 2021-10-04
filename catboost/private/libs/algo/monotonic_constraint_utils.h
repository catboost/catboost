#pragma once

#include "split.h"

#include <util/generic/map.h>
#include <util/generic/vector.h>

/* We can consider every oblivious tree with monotonic constraints as a tree on it's non-monotonic features
 * where at each leaf grows a fully monotonic subtree. For each monotonic subtree the monotonic constraints
 * (passed through treeMonotonicConstraints parameter) yield partial order on it's leafs. This function
 * returns indexes of leafs for monotonic subtree with num monotonicSubtreeIndex in some order consistent
 * with the partial order implied by the treeMonotonicConstraints.
 */
TVector<ui32> BuildLinearOrderOnLeafsOfMonotonicSubtree(
    const TVector<int>& treeMonotonicConstraints,
    const ui32 monotonicSubtreeIndex
);

/* For each fully monotonic subtree builds linear order on leafs consistent with treeMonotonicConstraints.
 */
TVector<TVector<ui32>> BuildMonotonicLinearOrdersOnLeafs(const TVector<int>& treeMonotonicConstraints);

/* This function solves one-dimensional isotonic regression problem.
 * See details here: https://en.wikipedia.org/wiki/Isotonic_regression#Simply_ordered_case
 * It implements algorithm known in literature as "pool adjacent violators algorithm".
 * It's time complexity is O(values.size()).
 * values and *solution may refer to the same vector.
 */
void CalcOneDimensionalIsotonicRegression(
    const TVector<double>& values,
    const TVector<double>& weights,
    const TVector<ui32>& indexOrder,
    TVector<double>* solution
);

TVector<int> GetTreeMonotoneConstraints(const TSplitTree& tree, const TMap<ui32, int>& monotoneConstraints);

inline TVector<int> GetTreeMonotoneConstraints(const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree, const TMap<ui32, int>& monotoneConstraints) {
    if (std::holds_alternative<TSplitTree>(tree)) {
        return GetTreeMonotoneConstraints(std::get<TSplitTree>(tree), monotoneConstraints);
    } else {
        CB_ENSURE_INTERNAL(monotoneConstraints.empty(), "Monotone constraints are unsupported for non-symmetric trees yet");
        return {};
    }
}

bool CheckMonotonicity(const TVector<ui32>& indexOrder, const TVector<double>& values);
