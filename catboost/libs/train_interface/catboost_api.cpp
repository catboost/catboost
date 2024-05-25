#include "catboost_api.h"

#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/helpers/json_helpers.h>
#include <library/cpp/threading/local_executor/local_executor.h>
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
    TConstArrayRef<float> features,
    ui32 fCount,
    TConstArrayRef<float> labels,
    TConstArrayRef<float> weights,
    TConstArrayRef<float> baseline
    ) {

    TDataProviderBuilderOptions builderOptions;

    THolder<IDataProviderBuilder> dataProviderBuilder;
    IRawFeaturesOrderDataVisitor* builderVisitor;

    CreateDataProviderBuilderAndVisitor(builderOptions,
                                        &NPar::LocalExecutor(),
                                        &dataProviderBuilder,
                                        &builderVisitor);

    TDataMetaInfo metaInfo;
    metaInfo.TargetType = ERawTargetType::Float;
    metaInfo.TargetCount = 1;
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


    builderVisitor->AddTarget(
        MakeIntrusive<TTypeCastArrayHolder<float, float>>(
            TMaybeOwningConstArrayHolder<float>::CreateNonOwning(labels)));
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

static inline TDataProviderPtr MakeProvider(const TDataSet& ds) {

    TConstArrayRef<float> features(ds.Features, ds.FeaturesCount * ds.SamplesCount);
    TConstArrayRef<float> labels(ds.Labels,  ds.SamplesCount);
    TConstArrayRef<float> weights(ds.Weights,  ds.SamplesCount);
    TConstArrayRef<float> baseline(ds.Baseline,  ds.SamplesCount * ds.BaselineDim);
    return MakeDataProvider(features, (ui32)ds.FeaturesCount, labels, weights, baseline);
}


extern "C" {

CATBOOST_API const char* GetErrorString() {
    return Singleton<TErrorMessageHolder>()->Message.data();
}

CATBOOST_API void FreeHandle(ResultHandle* modelHandle) {
    if (*modelHandle != nullptr) {
        delete RESULT_PTR(*modelHandle);
    }
    *modelHandle = nullptr;
}

CATBOOST_API int TreesCount(ResultHandle handle) {
    if (handle != nullptr) {
        return (int) RESULT_PTR(handle)->ModelTrees->GetTreeCount();
    }
    return 0;
}

CATBOOST_API int OutputDim(ResultHandle handle) {
    if (handle != nullptr) {
        return (int) RESULT_PTR(handle)->GetDimensionsCount();
    }
    return 0;
}

CATBOOST_API int TreeDepth(ResultHandle handle, int treeIndex) {
    if (handle) {
        return (int) RESULT_PTR(handle)->ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIndex];
    }
    return 0;
}

CATBOOST_API bool CopyTree(
    ResultHandle handle,
    int treeIndex,
    int* features,
    float* conditions,
    float* leaves,
    float* weights) {
    if (handle) {
        try {
            const auto modelTrees = RESULT_PTR(handle)->ModelTrees.GetMutable();

            size_t treeLeafCount = (1uLL << modelTrees->GetModelTreeData()->GetTreeSizes()[treeIndex]) * modelTrees->GetDimensionsCount();
            auto srcLeafValues = modelTrees->GetFirstLeafPtrForTree(treeIndex);
            const auto& srcWeights = modelTrees->GetModelTreeData()->GetLeafWeights();

            for (size_t idx = 0; idx < treeLeafCount; ++idx) {
                leaves[idx] = (float) srcLeafValues[idx];
            }

            auto applyData = modelTrees->GetApplyData();
            const size_t weightOffset = applyData->TreeFirstLeafOffsets[treeIndex] / modelTrees->GetDimensionsCount();
            for (size_t idx = 0; idx < (1uLL << modelTrees->GetModelTreeData()->GetTreeSizes()[treeIndex]); ++idx) {
                weights[idx] = (float) srcWeights[idx + weightOffset];
            }

            int treeSplitEnd;
            if (treeIndex + 1 < modelTrees->GetModelTreeData()->GetTreeStartOffsets().ysize()) {
                treeSplitEnd = modelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIndex + 1];
            } else {
                treeSplitEnd = modelTrees->GetModelTreeData()->GetTreeSplits().ysize();
            }
            const auto& binFeatures = modelTrees->GetBinFeatures();

            const auto offset = modelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIndex];
            for (int idx = offset; idx < treeSplitEnd; ++idx) {
                auto split = binFeatures[modelTrees->GetModelTreeData()->GetTreeSplits()[idx]];
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

CATBOOST_API bool TrainCatBoost(const TDataSet* trainPtr,
                                const TDataSet* testPtr,
                                const char* paramsJson,
                                ResultHandle* handlePtr) {
    const auto& train = *trainPtr;
    const auto& test = *testPtr;
    THolder<TFullModel> model = MakeHolder<TFullModel>();

    try {

        NJson::TJsonValue plainJsonParams;
        NJson::ReadJsonTree(TString(paramsJson),
                            &plainJsonParams);

        NCatboostOptions::TOption<ETaskType> taskType("task_type", ETaskType::CPU);
        TJsonFieldHelper<decltype(taskType)>::Read(plainJsonParams, &taskType);


        TDataProviders dataProviders = MakeDataProviders(
            MakeProvider(train),
            MakeProvider(test));
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
                   Nothing(),
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
