#pragma once

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

/* This function solves one-dimensional isotonic regression problem.
 * See details here: https://en.wikipedia.org/wiki/Isotonic_regression#Simply_ordered_case
 * It implements algorithm known in literature as "pool adjacent violators algorithm".
 * It's time complexity is O(values.size()).
 */
void CalcOneDimensionalIsotonicRegression(
    const TVector<double>& values,
    const TVector<double>& weight,
    const TVector<ui32>& indexOrder,
    TVector<double>* solution
);

bool CheckMonotonicity(const TVector<ui32>& indexOrder, const TVector<double>& values);
