#pragma once

#include "catboost_api.h"
#include <vector>
#include <string>
#include <cstddef>

struct TSymmetricTree {
    std::vector<int> Features;
    std::vector<float> Conditions;
    std::vector<float> Leaves;
    std::vector<float> Weights;

    int OutputDim() const {
        return Weights.size() ? Leaves.size() / Weights.size() : 0;
    }
};

struct TEnsemble {
    std::vector<TSymmetricTree> Trees;
};

inline TEnsemble Train(const TDataSet& learn,
                       const TDataSet& test,
                       const std::string& paramsJson) {
    ResultHandle handle;
    TrainCatBoost(&learn,
                  &test,
                  paramsJson.data(),
                  &handle);

    if (!handle) {
        throw std::exception();
    }

    TEnsemble ensemble;
    ensemble.Trees.resize(TreesCount(handle));
    int outputDim = OutputDim(handle);
    int numTrees = ensemble.Trees.size();
    for (int tree = 0; tree < numTrees; ++tree) {
        auto depth = TreeDepth(handle, tree);

        auto& currentTree = ensemble.Trees[tree];
        currentTree.Features.resize(depth);
        currentTree.Conditions.resize(depth);
        currentTree.Leaves.resize(outputDim * (1 << depth));
        currentTree.Weights.resize((1 << depth));

        CopyTree(handle,
                 tree,
                 currentTree.Features.data(),
                 currentTree.Conditions.data(),
                 currentTree.Leaves.data(),
                 currentTree.Weights.data()
        );
    }


    FreeHandle(&handle);
    return ensemble;
}
