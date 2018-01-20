#include "formula_evaluator.h"

void TFeatureCachedTreeEvaluator::Calc(size_t treeStart, size_t treeEnd, TArrayRef<double> results) const {
    CB_ENSURE(results.size() == DocCount * Model.ObliviousTrees.ApproxDimension);
    Fill(results.begin(), results.end(), 0.0);

    TVector<TCalcerIndexType> indexesVec(BlockSize);
    int id = 0;
    for (size_t blockStart = 0; blockStart < DocCount; blockStart += BlockSize) {
        const auto docCountInBlock = Min(BlockSize, DocCount - blockStart);
        if (Model.ObliviousTrees.ApproxDimension == 1) {
            CalcTrees<true, false>(
                    Model,
                    blockStart,
                    BinFeatures[id],
                    docCountInBlock,
                    indexesVec,
                    treeStart,
                    treeEnd,
                    results
            );
        } else {
            CalcTrees<false, false>(
                    Model,
                    blockStart,
                    BinFeatures[id],
                    docCountInBlock,
                    indexesVec,
                    treeStart,
                    treeEnd,
                    results
            );
        }
        ++id;
    }
}


