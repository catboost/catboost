#pragma once

#include "model.h"

namespace NCatBoost {
/**
 * Class for more optimal pointwise model apply:
 * Uses binarized feature values for fast tree index calculations
 * Binarized features in worst case consumes N_trees * 2 ^ max(TreeDepth) bytes
 * WARNING: Currently is will build optimized tree structures on construction or model load, that can
 */
class TFullModelCalcer {
public:
    TFullModelCalcer() = default;
    TFullModelCalcer(TFullModel&& model)
        : Model(std::move(model))
    {
        BuildBinTrees();
    }
    TFullModelCalcer(const TFullModel& model)
        : Model(model)
    {
        BuildBinTrees();
    }
    void Save(TOutputStream* out);
    void Load(TInputStream* in);

    template<typename TResult=double>
    TResult CalcOneResult(const float* features, int resultId) const {
        Y_ENSURE(resultId < Model.LeafValues[0].ysize());
        yvector<char> binFeatures = BinarizeFeatures(features);
        TResult result(0);
        for (size_t treeId = 0; treeId < BinaryTrees.size(); ++treeId) {
            const auto& tree = BinaryTrees[treeId];
            int index = 0;
            for (int depth = 0; depth < tree.ysize(); ++depth) {
                index |= binFeatures[tree[depth]] << depth;
            }
            result += Model.LeafValues[treeId][resultId][index];
        }
        return result;
    }

    template<typename TResult=double>
    void CalcMulti(const float* features, TResult* results, int resultsSize) const {
        Y_ENSURE(resultsSize == Model.LeafValues[0].ysize());
        yvector<char> binFeatures = BinarizeFeatures(features);
        std::fill(results, results + resultsSize, TResult(0));
        for (int treeId = 0; treeId < BinaryTrees.ysize(); ++treeId) {
            const auto& tree = BinaryTrees[treeId];
            int index = 0;
            for (int depth = 0; depth < tree.ysize(); ++depth) {
                index |= binFeatures[tree[depth]] << depth;
            }
            for (int resultId = 0; resultId < resultsSize; ++resultId) {
                results[resultId] += Model.LeafValues[treeId][resultId][index];
            }
        }
    }

    template<typename TResult=double>
    TResult Calc(const float* features) {
        return CalcOneResult(features, 0);
    }
private:
    void BuildBinTrees();

    // TODO(kirillovs): this function do features binarization, mn_sse is much more efficient in it
    // but it only supports tree depth <= 8, possibly we can use it for shallow trees computation
    template<typename T>
    yvector<char> BinarizeFeatures(const T* features) const {
        yvector<char> result(UsedBinaryFeaturesCount);
        size_t currendBinIndex = 0;
        for (const auto& floatFeature : UsedFloatFeatures) {
            const auto val = features[floatFeature.FeatureIndex];
            for (const auto& border : floatFeature.Borders) {
                result[currendBinIndex] = val > border;
                ++currendBinIndex;
            }
        }
        return result;
    };
protected:
    struct TFloatFeature {
        int FeatureIndex = -1;
        yvector<float> Borders;
    };
    struct TCtrFeature {
        TCtr Ctr;
        yvector<ui8> Borders;
    };
    TFullModel Model;
    // only bin feature trees supported for now, add corresponding UsedCatFeatures
    // and make proper feature indexes remapping if needed
    yvector<TFloatFeature> UsedFloatFeatures;
    yvector<TCtrFeature> UsedCtrFeatures;
    size_t UsedBinaryFeaturesCount = 0;
    // oblivious trees bin features trees - uses TSplit indexes from above array
    yvector<yvector<int>> BinaryTrees;
};
}
