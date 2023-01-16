#include "update_feature_weights.h"

#include <catboost/cuda/cuda_util/fill.h>

void UpdateFeatureWeightsForBestSplits(
    const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
    double modelSizeReg,
    TMirrorBuffer<float>& featureWeights,
    ui32 maxUniqueValues)
{

    ui32 featureCount = featuresManager.GetFeatureCount();

    if (featureWeights.GetMapping().GetObjectsSlice().IsEmpty()) {
        featureWeights.Reset(NCudaLib::TMirrorMapping(featureCount - featuresManager.GetTreeCtrCount()));

        FillBuffer(featureWeights, 1.0f);
    }

    if (featuresManager.GetCtrsCount() == 0) {
        return;
    }
    ui32 ctrBegin = featureCount - featuresManager.GetCtrsCount();

    TVector<float> weights;

    for (ui32 idx: xrange(ctrBegin, featureCount)) {
        if (featuresManager.IsCtr(idx) && !featuresManager.IsUsedCtr(idx)) {
            ui32 maxCtrUniqueValues = featuresManager.GetMaxCtrUniqueValues(idx);
            weights.push_back(maxCtrUniqueValues);
            maxUniqueValues = Max(maxUniqueValues, maxCtrUniqueValues);
        } else {
            weights.push_back(1);
        }
    }

    featureCount = featureWeights.GetMapping().GetObjectsSlice().Size();
    weights.resize(featureCount - ctrBegin);

    for (ui32 idx: xrange(weights.size())) {
        if (featuresManager.IsCtr(idx + ctrBegin) && !featuresManager.IsUsedCtr(idx + ctrBegin)) {
            weights[idx] = pow(1 + weights[idx] / maxUniqueValues, -modelSizeReg);
        }
    }

    auto sliceBuffer = featureWeights.SliceView(TSlice(ctrBegin, featureCount));
    sliceBuffer.Write(weights);
}
