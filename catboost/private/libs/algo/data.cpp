#include "data.h"

#include "approx_dimension.h"
#include "estimated_features.h"

#include <catboost/libs/data/borders_io.h>
#include <catboost/libs/data/quantization.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/load_options.h>
#include <catboost/private/libs/options/system_options.h>
#include <catboost/private/libs/target/data_providers.h>
#include <catboost/private/libs/feature_estimator/text_feature_estimators.h>
#include <catboost/private/libs/feature_estimator/embedding_feature_estimators.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/string/builder.h>


namespace NCB {

    static TVector<NCatboostOptions::TLossDescription> GetMetricDescriptions(
        const NCatboostOptions::TCatBoostOptions& params) {

        TVector<NCatboostOptions::TLossDescription> result;
        if (!IsUserDefined(params.LossFunctionDescription->GetLossFunction())) {
            result.emplace_back(params.LossFunctionDescription);
        }

        const auto& metricOptions = params.MetricOptions.Get();
        if (metricOptions.EvalMetric.IsSet()) {
            result.emplace_back(metricOptions.EvalMetric.Get());
        }
        if (metricOptions.CustomMetrics.IsSet()) {
            for (const auto& customMetric : metricOptions.CustomMetrics.Get()) {
                result.emplace_back(customMetric);
            }
        }
        return result;
    }

    using TInitialBorders = TMaybe<TVector<TConstArrayRef<float>>>;

    static TInitialBorders GetInitialBorders(TMaybe<TFullModel*> initModel) {
        if (!initModel) {
            return Nothing();
        }
        TVector<TConstArrayRef<float>> bordersInInitModel;
        bordersInInitModel.reserve((*initModel)->ModelTrees.GetMutable()->GetFloatFeatures().size());
        for (const auto& floatFeature : (*initModel)->ModelTrees.GetMutable()->GetFloatFeatures()) {
            bordersInInitModel.emplace_back(floatFeature.Borders.begin(), floatFeature.Borders.end());
        }
        return bordersInInitModel;
    }

    TTrainingDataProviderPtr GetTrainingData(
        TDataProviderPtr srcData,
        bool isLearnData,
        TStringBuf datasetName,
        const TMaybe<TString>& bordersFile,
        bool unloadCatFeaturePerfectHashFromRam,
        bool ensureConsecutiveIfDenseFeaturesDataForCpu,
        const TString& tmpDir,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        TMaybe<float>* targetBorder,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        TMaybe<TFullModel*> initModel) {

        const ui64 cpuRamLimit = ParseMemorySizeDescription(params->SystemOptions->CpuUsedRamLimit.Get());

        auto trainingData = MakeIntrusive<TTrainingDataProvider>();
        trainingData->OriginalFeaturesLayout = srcData->MetaInfo.FeaturesLayout;
        trainingData->MetaInfo = srcData->MetaInfo;
        trainingData->ObjectsGrouping = srcData->ObjectsGrouping;

        if (auto* quantizedObjectsDataProvider
                = dynamic_cast<TQuantizedObjectsDataProvider*>(srcData->ObjectsData.Get()))
        {
            if (params->GetTaskType() == ETaskType::CPU) {
                /*
                 * We need data to be consecutive for efficient blocked permutations
                 * but there're cases (e.g. CV with many folds) when limiting used CPU RAM is more important
                 */
                if (ensureConsecutiveIfDenseFeaturesDataForCpu) {
                    if (!quantizedObjectsDataProvider->GetFeaturesArraySubsetIndexing().IsConsecutive()) {
                        // TODO(akhropov): make it work in non-shared case
                        CB_ENSURE_INTERNAL(
                            (srcData->RefCount() <= 1) && (quantizedObjectsDataProvider->RefCount() <= 1),
                            "Cannot modify QuantizedForCPUObjectsDataProvider because it's shared"
                        );
                        quantizedObjectsDataProvider->EnsureConsecutiveIfDenseFeaturesData(localExecutor);
                    }
                }
            } else { // GPU
                /*
                 * if there're any cat features format should be CPU-compatible to enable final CTR
                 * calculations.
                 * TODO(akhropov): compatibility with final CTR calculation should not depend on this flag
                 */
                CB_ENSURE(
                    (srcData->MetaInfo.FeaturesLayout->GetCatFeatureCount() == 0) ||
                    quantizedObjectsDataProvider,
                    "Quantized objects data is not compatible with final CTR calculation"
                );
            }

            if (params->DataProcessingOptions.Get().IgnoredFeatures.IsSet()) {
                trainingData->ObjectsData = dynamic_cast<TQuantizedObjectsDataProvider*>(
                    quantizedObjectsDataProvider->GetFeaturesSubset(
                        params->DataProcessingOptions.Get().IgnoredFeatures,
                        localExecutor).Get()
                );
            } else {
                trainingData->ObjectsData = quantizedObjectsDataProvider;
            }
        } else {
            trainingData->ObjectsData = GetQuantizedObjectsData(
                *params,
                srcData,
                bordersFile,
                quantizedFeaturesInfo,
                localExecutor,
                rand,
                GetInitialBorders(initModel));
        }

        CB_ENSURE(
            trainingData->ObjectsData->GetFeaturesLayout()->HasAvailableAndNotIgnoredFeatures(),
            "All features are either constant or ignored.");

        //(TODO)
        // because some features can become unavailable/ignored due to quantization
        trainingData->MetaInfo.FeaturesLayout = trainingData->ObjectsData->GetFeaturesLayout();

        if (unloadCatFeaturePerfectHashFromRam) {
            trainingData->ObjectsData->GetQuantizedFeaturesInfo()->UnloadCatFeaturePerfectHashFromRam(tmpDir);
        }

        auto& dataProcessingOptions = params->DataProcessingOptions.Get();

        bool calcCtrs
            = (trainingData->ObjectsData->GetQuantizedFeaturesInfo()
                ->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn()
               > params->CatFeatureParams->OneHotMaxSize.Get());

        bool needTargetDataForCtrs = calcCtrs && CtrsNeedTargetData(params->CatFeatureParams) && isLearnData;

        TInputClassificationInfo inputClassificationInfo {
            dataProcessingOptions.ClassesCount.Get() ? TMaybe<ui32>(dataProcessingOptions.ClassesCount.Get()) : Nothing(),
            dataProcessingOptions.ClassWeights.Get(),
            dataProcessingOptions.AutoClassWeights.Get(),
            dataProcessingOptions.ClassLabels.Get(),
            *targetBorder
        };
        TOutputClassificationInfo outputClassificationInfo {
            dataProcessingOptions.ClassLabels.Get(),
            labelConverter,
            *targetBorder,
            dataProcessingOptions.ClassWeights.Get()
        };
        TOutputPairsInfo outputPairsInfo;

        const auto targetCreationOptions = MakeTargetCreationOptions(
            srcData->RawTargetData,
            GetMetricDescriptions(*params),
            /*knownModelApproxDimension*/ Nothing(),
            inputClassificationInfo
        );

        CB_ENSURE(!isLearnData || srcData->RawTargetData.GetObjectCount() > 0, "Train dataset is empty");

        trainingData->TargetData = CreateTargetDataProvider(
            srcData->RawTargetData,
            trainingData->ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ params->GetTaskType() == ETaskType::GPU,
            &params->LossFunctionDescription.Get(),
            /*metricsThatRequireTargetCanBeSkipped*/ !isLearnData,
            /*knownModelApproxDimension*/ Nothing(),
            targetCreationOptions,
            inputClassificationInfo,
            &outputClassificationInfo,
            rand,
            localExecutor,
            &outputPairsInfo
        );

        CheckTargetConsistency(
            trainingData->TargetData,
            GetMetricDescriptions(*params),
            &params->LossFunctionDescription.Get(),
            needTargetDataForCtrs,
            !isLearnData,
            datasetName,
            isLearnData,
            dataProcessingOptions.AllowConstLabel.Get()
        );

        trainingData->MetaInfo.HasPairs = outputPairsInfo.HasPairs;
        trainingData->MetaInfo.HasWeights |= !inputClassificationInfo.ClassWeights.empty();
        trainingData->MetaInfo.HasWeights |= inputClassificationInfo.AutoClassWeightsType != EAutoClassWeightsType::None;
        trainingData->MetaInfo.ClassLabels = outputClassificationInfo.ClassLabels;
        dataProcessingOptions.ClassLabels.Get() = outputClassificationInfo.ClassLabels;
        *targetBorder = outputClassificationInfo.TargetBorder;

        if (!outputClassificationInfo.ClassWeights.Empty()) {
            dataProcessingOptions.ClassWeights.Get() = *outputClassificationInfo.ClassWeights.Get();
        }

        trainingData->UpdateMetaInfo();

        if (outputPairsInfo.HasFakeGroupIds()) {
            trainingData = trainingData->GetSubset(
                TObjectsGroupingSubset(
                    trainingData->TargetData->GetObjectsGrouping(),
                    TArraySubsetIndexing<ui32>(TIndexedSubset<ui32>(outputPairsInfo.PermutationForGrouping)),
                    EObjectsOrder::Undefined
                ),
                cpuRamLimit,
                localExecutor
            );
            trainingData->TargetData->UpdateGroupInfos(
                MakeGroupInfos(
                    outputPairsInfo.FakeObjectsGrouping,
                    Nothing(),
                    TWeights(outputPairsInfo.PermutationForGrouping.size()),
                    TConstArrayRef<TPair>(outputPairsInfo.PairsInPermutedDataset)
                )
            );
        }

        return trainingData;
    }


    void CheckCompatibilityWithEvalMetric(
        const NCatboostOptions::TLossDescription& evalMetricDescription,
        const TTrainingDataProvider& trainingData,
        ui32 approxDimension) {

        if (trainingData.MetaInfo.TargetCount > 0) {
            return;
        }

        auto metrics = CreateMetricFromDescription(evalMetricDescription, (int)approxDimension);
        for (const auto& metric : metrics) {
            CB_ENSURE(
                !metric->NeedTarget(),
                "Eval metric " << metric->GetDescription() << " needs Target data for test dataset, but "
                "it is not available"
            );
        }
    }

    static TTextDataSetPtr CreateTextDataSet(
        const TQuantizedObjectsDataProvider& dataProvider,
        ui32 tokenizedTextFeatureIdx,
        NPar::ILocalExecutor* localExecutor
    ) {
        const TTextDigitizers& digitizers = dataProvider.GetQuantizedFeaturesInfo()->GetTextDigitizers();
        auto dictionary = digitizers.GetDigitizer(tokenizedTextFeatureIdx).Dictionary;
        const TTokenizedTextValuesHolder* textColumn = *dataProvider.GetTextFeature(tokenizedTextFeatureIdx);

        if (const auto* denseData = dynamic_cast<const TTokenizedTextArrayValuesHolder*>(textColumn)) {
            TMaybeOwningArrayHolder<TText> textData = denseData->ExtractValues(localExecutor);
            TMaybeOwningConstArrayHolder<TText> constTextData
                = TMaybeOwningConstArrayHolder<TText>::CreateOwning(*textData, textData.GetResourceHolder());
            return MakeIntrusive<TTextDataSet>(std::move(constTextData), dictionary);
        } else {
            CB_ENSURE_INTERNAL(false, "CreateTextDataSet: unsupported column type");
        }
        Y_UNREACHABLE();
    }

    static TEmbeddingDataSetPtr CreateEmbeddingDataSet(
        const TQuantizedObjectsDataProvider& dataProvider,
        ui32 embeddingFeatureIdx,
        NPar::ILocalExecutor* localExecutor
    ) {
        const TEmbeddingValuesHolder* embeddingColumn = *dataProvider.GetEmbeddingFeature(embeddingFeatureIdx);

        const auto* denseData = dynamic_cast<const TEmbeddingArrayValuesHolder*>(embeddingColumn);
        TMaybeOwningArrayHolder<TConstEmbedding> embeddingData = denseData->ExtractValues(localExecutor);
        TMaybeOwningConstArrayHolder<TConstEmbedding> constEmbeddingData
            = TMaybeOwningConstArrayHolder<TConstEmbedding>::CreateOwning(*embeddingData,
                                                                            embeddingData.GetResourceHolder());
        return MakeIntrusive<TEmbeddingDataSet>(std::move(constEmbeddingData));
    }

    static TTextClassificationTargetPtr CreateTextClassificationTarget(const TTargetDataProvider& targetDataProvider) {
        const ui32 numClasses = *targetDataProvider.GetTargetClassCount();
        TConstArrayRef<float> target = *targetDataProvider.GetOneDimensionalTarget();
        TVector<ui32> classes;
        classes.resize(target.size());

        for (ui32 i = 0; i < target.size(); i++) {
            classes[i] = static_cast<ui32>(target[i]);
        }
        return MakeIntrusive<TTextClassificationTarget>(std::move(classes), numClasses);
    }

    static TEmbeddingClassificationTargetPtr CreateEmbeddingClassificationTarget(const TTargetDataProvider& targetDataProvider) {
        const ui32 numClasses = *targetDataProvider.GetTargetClassCount();
        TConstArrayRef<float> target = *targetDataProvider.GetOneDimensionalTarget();
        TVector<ui32> classes(target.size());

        for (ui32 i = 0; i < target.size(); i++) {
            classes[i] = static_cast<ui32>(target[i]);
        }
        return MakeIntrusive<TEmbeddingClassificationTarget>(std::move(classes), numClasses);
    }

    static TFeatureEstimatorsPtr CreateEstimators(
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TTrainingDataProviders pools,
        NPar::ILocalExecutor* localExecutor
    ) {
        auto tokenizedFeaturesDescription = quantizedFeaturesInfo->GetTextProcessingOptions().GetTokenizedFeatureDescriptions();

        TFeatureEstimatorsBuilder estimatorsBuilder;

        const TQuantizedObjectsDataProvider& learnDataProvider = *pools.Learn->ObjectsData;
        auto learnTextTarget = CreateTextClassificationTarget(*pools.Learn->TargetData);

        pools.Learn->MetaInfo.FeaturesLayout->IterateOverAvailableFeatures<EFeatureType::Text>(
            [&](TTextFeatureIdx tokenizedTextFeatureIdx) {

                const ui32 tokenizedFeatureIdx = tokenizedTextFeatureIdx.Idx;
                auto learnTexts = CreateTextDataSet(learnDataProvider, tokenizedFeatureIdx, localExecutor);

                TVector<TTextDataSetPtr> testTexts;
                for (const auto& testDataProvider : pools.Test) {
                    testTexts.emplace_back(
                        CreateTextDataSet(
                            *testDataProvider->ObjectsData,
                            tokenizedFeatureIdx,
                            localExecutor
                        ));
                }

                const auto& featureDescription = tokenizedFeaturesDescription[tokenizedFeatureIdx];
                auto offlineEstimators = CreateTextEstimators(
                    featureDescription.FeatureEstimators.Get(),
                    learnTexts,
                    testTexts
                );

                const ui32 textFeatureId = tokenizedFeaturesDescription[tokenizedFeatureIdx].TextFeatureId;
                TEstimatorSourceId sourceFeatureIdx{textFeatureId, tokenizedFeatureIdx};
                for (auto&& estimator : offlineEstimators) {
                    estimatorsBuilder.AddFeatureEstimator(std::move(estimator), sourceFeatureIdx);
                }

                auto onlineEstimators = CreateTextEstimators(
                    featureDescription.FeatureEstimators.Get(),
                    learnTextTarget,
                    learnTexts,
                    testTexts
                );
                for (auto&& estimator : onlineEstimators) {
                    estimatorsBuilder.AddFeatureEstimator(std::move(estimator), sourceFeatureIdx);
                }
            }
        );

        auto embeddingFeaturesDescription = quantizedFeaturesInfo->GetEmbeddingProcessingOptions().GetFeatureDescriptions();
        auto learnEmbeddingTarget = CreateEmbeddingClassificationTarget(*pools.Learn->TargetData);

        pools.Learn->MetaInfo.FeaturesLayout->IterateOverAvailableFeatures<EFeatureType::Embedding>(
            [&](TEmbeddingFeatureIdx embeddingFeature) {

                const ui32 embeddingFeatureIdx = embeddingFeature.Idx;
                auto learnEmbeddings = CreateEmbeddingDataSet(learnDataProvider, embeddingFeatureIdx, localExecutor);

                TVector<TEmbeddingDataSetPtr> testEmbeddings;
                for (const auto& testDataProvider : pools.Test) {
                    testEmbeddings.emplace_back(
                        CreateEmbeddingDataSet(
                            *testDataProvider->ObjectsData,
                            embeddingFeatureIdx,
                            localExecutor
                        ));
                }

                const auto& featureDescription = embeddingFeaturesDescription[embeddingFeatureIdx];
                const ui32 featureId = embeddingFeaturesDescription[embeddingFeatureIdx].EmbeddingFeatureId;
                TEstimatorSourceId sourceFeatureIdx{featureId, embeddingFeatureIdx};

                auto onlineEstimators = CreateEmbeddingEstimators(
                    featureDescription.FeatureEstimators.Get(),
                    learnEmbeddingTarget,
                    learnEmbeddings,
                    testEmbeddings
                );
                for (auto&& estimator : onlineEstimators) {
                    estimatorsBuilder.AddFeatureEstimator(std::move(estimator), sourceFeatureIdx);
                }
            }
        );

        return estimatorsBuilder.Build();
    }


    TTrainingDataProviders GetTrainingData(
        TDataProviders srcData,
        const TMaybe<TString>& bordersFile, // load borders from it if specified
        bool ensureConsecutiveIfDenseLearnFeaturesDataForCpu,
        bool allowWriteFiles,
        const TString& tmpDir,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr, then create it
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        TMaybe<TFullModel*> initModel) {

        TTrainingDataProviders trainingData;

        TMaybe<float> targetBorder = params->DataProcessingOptions->TargetBorder;
        trainingData.Learn = GetTrainingData(
            std::move(srcData.Learn),
            /*isLearnData*/ !params->SystemOptions->IsWorker(),
            "learn",
            bordersFile,
            /*unloadCatFeaturePerfectHashFromRam*/ allowWriteFiles && srcData.Test.empty(),
            ensureConsecutiveIfDenseLearnFeaturesDataForCpu,
            tmpDir,
            quantizedFeaturesInfo,
            params,
            labelConverter,
            &targetBorder,
            localExecutor,
            rand,
            initModel
        );

        quantizedFeaturesInfo = trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo();

        for (auto testIdx : xrange(srcData.Test.size())) {
            trainingData.Test.push_back(
                GetTrainingData(
                    std::move(srcData.Test[testIdx]),
                    /*isLearnData*/ false,
                    TStringBuilder() << "test #" << testIdx,
                    Nothing(), // borders already loaded
                    /*unloadCatFeaturePerfectHashFromRam*/
                        allowWriteFiles && ((testIdx + 1) == srcData.Test.size()),
                    /*ensureConsecutiveIfDenseFeaturesDataForCpu*/ false, // not needed for test
                    tmpDir,
                    quantizedFeaturesInfo,
                    params,
                    labelConverter,
                    &targetBorder,
                    localExecutor,
                    rand
                )
            );
        }

        if (trainingData.Learn->MetaInfo.FeaturesLayout->GetTextFeatureCount() > 0
            || trainingData.Learn->MetaInfo.FeaturesLayout->GetEmbeddingFeatureCount() > 0) {
            CB_ENSURE(
                IsClassificationObjective(params->LossFunctionDescription->LossFunction),
                "Computation of online text features is supported only for classification task"
            );
            trainingData.FeatureEstimators = CreateEstimators(
                quantizedFeaturesInfo,
                trainingData,
                localExecutor);

            if (params->GetTaskType() == ETaskType::CPU) {
                trainingData.EstimatedObjectsData = CreateEstimatedFeaturesData(
                    params->DataProcessingOptions->FloatFeaturesBinarization.Get(),
                    /*maxSubsetSizeForBuildBordersAlgorithms*/ 100000,
                    /*quantizedFeaturesInfo*/ nullptr,
                    trainingData,
                    trainingData.FeatureEstimators,
                    /*learnPermutation*/ Nothing(), // offline features
                    localExecutor,
                    rand
                );
            }
        }

        if (params->MetricOptions->EvalMetric.IsSet() && (srcData.Test.size() > 0)) {
            CheckCompatibilityWithEvalMetric(
                params->MetricOptions->EvalMetric,
                *trainingData.Test.back(),
                GetApproxDimension(*params, *labelConverter, trainingData.Test.back()->TargetData->GetTargetDimension())
            );
        }


        return trainingData;
    }

    static TIntrusivePtr<TTrainingDataProvider> MakeFeatureSubsetDataProvider(
        const TVector<ui32>& ignoredFeatures,
        NCB::TTrainingDataProviderPtr trainingDataProvider
    ) {
        TQuantizedObjectsDataProviderPtr newObjects = dynamic_cast<TQuantizedObjectsDataProvider*>(
            trainingDataProvider->ObjectsData->GetFeaturesSubset(ignoredFeatures, &NPar::LocalExecutor()).Get());
        CB_ENSURE(
            newObjects,
            "Objects data provider must be TQuantizedObjectsDataProvider or TQuantizedObjectsDataProvider");
        TDataMetaInfo newMetaInfo = trainingDataProvider->MetaInfo;
        newMetaInfo.FeaturesLayout = newObjects->GetFeaturesLayout();
        return MakeIntrusive<TTrainingDataProvider>(
            trainingDataProvider->OriginalFeaturesLayout,
            TDataMetaInfo(newMetaInfo),
            trainingDataProvider->ObjectsGrouping,
            newObjects,
            trainingDataProvider->TargetData);
    }

    TTrainingDataProviders MakeFeatureSubsetTrainingData(
        const TVector<ui32>& ignoredFeatures,
        const NCB::TTrainingDataProviders& trainingData
    ) {
        TTrainingDataProviders newTrainingData;
        newTrainingData.Learn = MakeFeatureSubsetDataProvider(ignoredFeatures, trainingData.Learn);
        newTrainingData.Test.reserve(trainingData.Test.size());
        for (const auto& test : trainingData.Test) {
            newTrainingData.Test.push_back(MakeFeatureSubsetDataProvider(ignoredFeatures, test));
        }

        newTrainingData.FeatureEstimators = trainingData.FeatureEstimators;

        // TODO(akhropov): correctly support ignoring indices based on source data
        newTrainingData.EstimatedObjectsData = trainingData.EstimatedObjectsData;

        return newTrainingData;
    }

    bool HaveLearnFeaturesInMemory(
        const NCatboostOptions::TPoolLoadParams* loadOptions,
        const NCatboostOptions::TCatBoostOptions& catBoostOptions
    ) {
        #if defined(USE_MPI)
        const bool isGpuDistributed = catBoostOptions.GetTaskType() == ETaskType::GPU;
        #else
        const bool isGpuDistributed = false;
        #endif
        const bool isCpuDistributed = catBoostOptions.SystemOptions->IsMaster();
        if (!isCpuDistributed && !isGpuDistributed) {
            return true;
        }
        if (loadOptions == nullptr) {
            return true;
        }
        const auto& learnSetPath = loadOptions->LearnSetPath;
        const bool isQuantized = learnSetPath.Scheme.find("quantized") != std::string::npos;
        return !IsSharedFs(learnSetPath) || !isQuantized;
    }

}
