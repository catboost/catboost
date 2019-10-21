#include "catboost_api.h"

#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <util/generic/singleton.h>
#include <util/stream/file.h>
#include <util/string/builder.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>

#define RESULT_PTR(x) ((TFullModel*)(x))

struct TErrorMessageHolder {
    TString Message;
};

using namespace NCB;

static TDataProviderPtr MakeDataProvider(
    bool isGpu,
    TConstArrayRef<float> features,
    ui32 fCount,
    TConstArrayRef<float> labels,
    TConstArrayRef<float> weights,
    TConstArrayRef<float> baseline
    ) {

    TDataProviderBuilderOptions builderOptions{!isGpu,
                                               isGpu,
                                               false};

    THolder<IDataProviderBuilder> dataProviderBuilder;
    IRawFeaturesOrderDataVisitor* builderVisitor;

    CreateDataProviderBuilderAndVisitor(builderOptions,
                                        &dataProviderBuilder,
                                        &builderVisitor);

    TDataMetaInfo metaInfo;
    metaInfo.HasTarget = true;
    metaInfo.HasWeights = weights.data() != nullptr;
    if (baseline.data() != nullptr) {
        metaInfo.BaselineCount = baseline.size() / labels.size();
    }
    metaInfo.FeaturesLayout = new TFeaturesLayout((ui32) fCount);

    const size_t samplesCount = features.size() / fCount;

    builderVisitor->Start(metaInfo,
                          samplesCount,
                          EObjectsOrder::Ordered, {});

    for (ui32 f = 0; f < fCount; ++f) {
        auto valHolder =
            MakeIntrusive<TTypeCastArrayHolder<float, float>>(
                TMaybeOwningConstArrayHolder<float>::CreateNonOwning(
                    features.Slice(f * samplesCount, samplesCount)));
        builderVisitor->AddFloatFeature(f, valHolder);
    }


    builderVisitor->AddTarget(labels);
    if (weights.data() != nullptr) {
        builderVisitor->AddWeights(weights);
    }

    if (baseline.data() != nullptr) {
        const ui32 baselineCount = baseline.size() / labels.size();
        for (ui32 baselineIdx = 0; baselineIdx < baselineCount; ++baselineIdx) {
            builderVisitor->AddBaseline(baselineIdx, baseline.Slice(baselineIdx * labels.size(), labels.size()));
        }
    }
    builderVisitor->Finish();

    return dataProviderBuilder->GetResult();
}

TDataProviders MakeDataProviders(TDataProviderPtr learn, TDataProviderPtr test) {
    TDataProviders providers;
    providers.Learn = learn;
    providers.Test = {test};
    return providers;
}

static inline TDataProviderPtr MakeProvider(bool gpu, const TDataSet& ds) {

    TConstArrayRef<float> features(ds.Features, ds.FeaturesCount * ds.SamplesCount);
    TConstArrayRef<float> labels(ds.Labels,  ds.SamplesCount);
    TConstArrayRef<float> weights(ds.Weights,  ds.SamplesCount);
    TConstArrayRef<float> baseline(ds.Baseline,  ds.SamplesCount * ds.BaselineDim);
    return MakeDataProvider(gpu, features, (ui32)ds.FeaturesCount, labels, weights, baseline);
}


extern "C" {

EXPORT const char* GetErrorString() {
    return Singleton<TErrorMessageHolder>()->Message.data();
}

EXPORT void FreeHandle(ResultHandle* modelHandle) {
    if (*modelHandle != nullptr) {
        delete RESULT_PTR(*modelHandle);
    }
    *modelHandle = nullptr;
}

EXPORT int TreesCount(ResultHandle handle) {
    if (handle != nullptr) {
        return (int) RESULT_PTR(handle)->ObliviousTrees->GetTreeCount();
    }
    return 0;
}

EXPORT int OutputDim(ResultHandle handle) {
    if (handle != nullptr) {
        return (int) RESULT_PTR(handle)->GetDimensionsCount();
    }
    return 0;
}

EXPORT int TreeDepth(ResultHandle handle, int treeIndex) {
    if (handle) {
        return (int) RESULT_PTR(handle)->ObliviousTrees->GetTreeSizes()[treeIndex];
    }
    return 0;
}

EXPORT bool CopyTree(
    ResultHandle handle,
    int treeIndex,
    int* features,
    float* conditions,
    float* leaves,
    float* weights) {
    if (handle) {
        try {
            const auto obliviousTrees = RESULT_PTR(handle)->ObliviousTrees.GetMutable();

            size_t treeLeafCount = (1uLL << obliviousTrees->GetTreeSizes()[treeIndex]) * obliviousTrees->GetDimensionsCount();
            auto srcLeafValues = obliviousTrees->GetFirstLeafPtrForTree(treeIndex);
            const auto& srcWeights = obliviousTrees->GetLeafWeights();

            for (size_t idx = 0; idx < treeLeafCount; ++idx) {
                leaves[idx] = (float) srcLeafValues[idx];
            }

            const size_t weightOffset = obliviousTrees->GetFirstLeafOffsets()[treeIndex] / obliviousTrees->GetDimensionsCount();
            for (size_t idx = 0; idx < (1uLL << obliviousTrees->GetTreeSizes()[treeIndex]); ++idx) {
                weights[idx] = (float) srcWeights[idx + weightOffset];
            }

            int treeSplitEnd;
            if (treeIndex + 1 < obliviousTrees->GetTreeStartOffsets().ysize()) {
                treeSplitEnd = obliviousTrees->GetTreeStartOffsets()[treeIndex + 1];
            } else {
                treeSplitEnd = obliviousTrees->GetTreeSplits().ysize();
            }
            const auto& binFeatures = obliviousTrees->GetBinFeatures();

            const auto offset = obliviousTrees->GetTreeStartOffsets()[treeIndex];
            for (int idx = offset; idx < treeSplitEnd; ++idx) {
                auto split = binFeatures[obliviousTrees->GetTreeSplits()[idx]];
                CB_ENSURE(split.Type == ESplitType::FloatFeature);
                features[idx - offset] = split.FloatFeature.FloatFeature;
                conditions[idx - offset] = split.FloatFeature.Split;
            }
        } catch (...) {
            Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
            return false;
        }
    }
    return true;
}

EXPORT bool TrainCatBoost(const TDataSet* trainPtr,
                          const TDataSet* testPtr,
                          const char* paramsJson,
                          ResultHandle* handlePtr) {
    const auto& train = *trainPtr;
    const auto& test = *testPtr;
    THolder<TFullModel> model = new TFullModel;

    try {

        NJson::TJsonValue plainJsonParams;
        NJson::ReadJsonTree(TString(paramsJson),
                            &plainJsonParams);

        NCatboostOptions::TOption<ETaskType> taskType("task_type", ETaskType::CPU);
        NCatboostOptions::TJsonFieldHelper<decltype(taskType)>::Read(plainJsonParams, &taskType);


        const bool useGpu = taskType == ETaskType::GPU;
        TDataProviders dataProviders = MakeDataProviders(
            MakeProvider(useGpu, train),
            MakeProvider(useGpu, test));
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;

        TMetricsAndTimeLeftHistory history;
        TEvalResult evalResult;
        TVector<TEvalResult*> evalResultsPtr = {&evalResult};

        TMaybe<TCustomObjectiveDescriptor> objectiveDescriptor;
        TMaybe<TCustomMetricDescriptor> evalMetricDescriptor;
        TString outModelPath = "";
        TrainModel(plainJsonParams,
                   quantizedFeaturesInfo,
                   objectiveDescriptor,
                   evalMetricDescriptor,
                   std::move(dataProviders),
                   /*initModel*/ Nothing(),
                   /*initLearnProgress*/ nullptr,
                   outModelPath,
                   model.Get(),
                   evalResultsPtr,
                   &history);
    } catch (...) {
        Cout << CurrentExceptionMessage() << Endl;
        Singleton<TErrorMessageHolder>()->Message = CurrentExceptionMessage();
        return false;
    }

    (*handlePtr) = model.Release();
    return true;
}

}
