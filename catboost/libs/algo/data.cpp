#include "data.h"

#include "approx_dimension.h"


#include <catboost/libs/data_new/borders_io.h>
#include <catboost/libs/data_new/quantization.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/labels/label_converter.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/target/data_providers.h>
#include <catboost/libs/feature_estimator/text_feature_estimators.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/string/builder.h>


namespace NCB {

    static TVector<NCatboostOptions::TLossDescription> GetMetricDescriptions(
        const NCatboostOptions::TCatBoostOptions& params) {

        TVector<NCatboostOptions::TLossDescription> result;
        if (params.LossFunctionDescription->GetLossFunction() != ELossFunction::PythonUserDefinedPerObject) {
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

    TTrainingDataProviderPtr GetTrainingData(
        TDataProviderPtr srcData,
        bool isLearnData,
        TStringBuf datasetName,
        const TMaybe<TString>& bordersFile,
        bool unloadCatFeaturePerfectHashFromRamIfPossible,
        bool ensureConsecutiveFeaturesDataForCpu,
        bool allowWriteFiles,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        TMaybe<float>* targetBorder,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand) {

        auto trainingData = MakeIntrusive<TTrainingDataProvider>();
        trainingData->MetaInfo = srcData->MetaInfo;
        trainingData->ObjectsGrouping = srcData->ObjectsGrouping;

        if (auto* quantizedObjectsDataProviderPtr
                = dynamic_cast<TQuantizedObjectsDataProvider*>(srcData->ObjectsData.Get()))
        {
            if (params->GetTaskType() == ETaskType::CPU) {
                auto quantizedForCPUObjectsDataProvider
                    = dynamic_cast<TQuantizedForCPUObjectsDataProvider*>(quantizedObjectsDataProviderPtr);
                CB_ENSURE(
                    quantizedForCPUObjectsDataProvider,
                    "Quantized objects data is not compatible with CPU task type"
                );

                /*
                 * We need data to be consecutive for efficient blocked permutations
                 * but there're cases (e.g. CV with many folds) when limiting used CPU RAM is more important
                 */
                if (ensureConsecutiveFeaturesDataForCpu) {
                    if (!quantizedForCPUObjectsDataProvider->GetFeaturesArraySubsetIndexing().IsConsecutive()) {
                        // TODO(akhropov): make it work in non-shared case
                        CB_ENSURE_INTERNAL(
                            (srcData->RefCount() <= 1) && (quantizedForCPUObjectsDataProvider->RefCount() <= 1),
                            "Cannot modify QuantizedForCPUObjectsDataProvider because it's shared"
                        );
                        quantizedForCPUObjectsDataProvider->EnsureConsecutiveFeaturesData(localExecutor);
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
                    dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(quantizedObjectsDataProviderPtr),
                    "Quantized objects data is not compatible with final CTR calculation"
                );
            }

            trainingData->ObjectsData = quantizedObjectsDataProviderPtr;
            trainingData->ObjectsData->GetQuantizedFeaturesInfo()->SetAllowWriteFiles(allowWriteFiles);
        } else {
            TQuantizationOptions quantizationOptions;
            if (params->GetTaskType() == ETaskType::CPU) {
                quantizationOptions.GpuCompatibleFormat = false;

                quantizationOptions.ExclusiveFeaturesBundlingOptions.MaxBuckets
                    = params->ObliviousTreeOptions->DevExclusiveFeaturesBundleMaxBuckets.Get();
                quantizationOptions.ExclusiveFeaturesBundlingOptions.MaxConflictFraction
                    = params->ObliviousTreeOptions->SparseFeaturesConflictFraction.Get();
            } else {
                Y_ASSERT(params->GetTaskType() == ETaskType::GPU);

                /*
                 * if there're any cat features format should be CPU-compatible to enable final CTR
                 * calculations.
                 * TODO(akhropov): compatibility with final CTR calculation should not depend on this flag
                 */
                quantizationOptions.CpuCompatibleFormat
                    = srcData->MetaInfo.FeaturesLayout->GetCatFeatureCount() != 0;
                if (quantizationOptions.CpuCompatibleFormat) {
                    /* don't spend time on bundling preprocessing because it won't be used
                     *
                     * TODO(akhropov): maybe there are cases where CPU RAM usage reduction is more important
                     *    than calculation speed so it should be enabled
                     */
                    quantizationOptions.BundleExclusiveFeaturesForCpu = false;
                }
            }
            quantizationOptions.CpuRamLimit
                = ParseMemorySizeDescription(params->SystemOptions->CpuUsedRamLimit.Get());
            quantizationOptions.AllowWriteFiles = allowWriteFiles;

            if (!quantizedFeaturesInfo) {
                quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                    *srcData->MetaInfo.FeaturesLayout,
                    params->DataProcessingOptions->IgnoredFeatures.Get(),
                    params->DataProcessingOptions->FloatFeaturesBinarization.Get(),
                    params->DataProcessingOptions->PerFloatFeatureQuantization.Get(),
                    params->DataProcessingOptions->TextProcessing.Get(),
                    /*allowNansInTestOnly*/true
                );

                if (bordersFile) {
                    LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
                        *bordersFile,
                        quantizedFeaturesInfo.Get());
                }
            }

            TRawObjectsDataProviderPtr rawObjectsDataProvider(
                dynamic_cast<TRawObjectsDataProvider*>(srcData->ObjectsData.Get()));
            Y_VERIFY(rawObjectsDataProvider);

            if (srcData->RefCount() <= 1) {
                // can clean up
                auto dummy = srcData->ObjectsData.Release();
                Y_UNUSED(dummy);
            }

            trainingData->ObjectsData = Quantize(
                quantizationOptions,
                std::move(rawObjectsDataProvider),
                quantizedFeaturesInfo,
                rand,
                localExecutor
            );

            // because some features can become unavailable/ignored due to quantization
            trainingData->MetaInfo.FeaturesLayout = quantizedFeaturesInfo->GetFeaturesLayout();
        }

        if (unloadCatFeaturePerfectHashFromRamIfPossible) {
            trainingData->ObjectsData->GetQuantizedFeaturesInfo()
                ->UnloadCatFeaturePerfectHashFromRamIfPossible();
        }

        auto& dataProcessingOptions = params->DataProcessingOptions.Get();

        bool calcCtrs
            = (trainingData->ObjectsData->GetQuantizedFeaturesInfo()
                ->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn()
               > params->CatFeatureParams->OneHotMaxSize.Get());

        bool needTargetDataForCtrs = calcCtrs && CtrsNeedTargetData(params->CatFeatureParams) && isLearnData;

        trainingData->TargetData = CreateTargetDataProvider(
            srcData->RawTargetData,
            trainingData->ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ params->GetTaskType() == ETaskType::GPU,
            isLearnData,
            datasetName,
            GetMetricDescriptions(*params),
            &params->LossFunctionDescription.Get(),
            dataProcessingOptions.AllowConstLabel.Get(),
            /*metricsThatRequireTargetCanBeSkipped*/ !isLearnData,
            /*needTargetDataForCtrs*/ needTargetDataForCtrs,
            /*knownModelApproxDimension*/ Nothing(),
            dataProcessingOptions.ClassesCount.Get(),
            dataProcessingOptions.ClassWeights.Get(),
            &dataProcessingOptions.ClassNames.Get(),
            labelConverter,
            targetBorder,
            rand,
            localExecutor,
            &trainingData->MetaInfo.HasPairs
        );

        trainingData->UpdateMetaInfo();

        return trainingData;
    }


    void CheckCompatibilityWithEvalMetric(
        const NCatboostOptions::TLossDescription& evalMetricDescription,
        const TTrainingDataProvider& trainingData,
        ui32 approxDimension) {

        if (trainingData.MetaInfo.HasTarget) {
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

    static TTextDataSetPtr CreateTextDataSet(const TQuantizedObjectsDataProvider& dataProvider, TTextFeatureIdx textFeatureIdx) {
        auto dictionary = dataProvider.GetQuantizedFeaturesInfo()->GetDictionary(textFeatureIdx);

        const TTokenizedTextValuesHolder* textColumn = *dataProvider.GetTextFeature(textFeatureIdx.Idx);
        if (const auto* denseData = dynamic_cast<const TTokenizedTextArrayValuesHolder*>(textColumn)) {
            return MakeIntrusive<TTextDataSet>(*denseData->GetArrayData().GetSrc(), dictionary);
        } else {
            CB_ENSURE_INTERNAL(false, "CreateTextDataSet: unsupported column type");
        }
        Y_UNREACHABLE();
    }

    static TTextClassificationTargetPtr CreateTextClassificationTarget(const TTargetDataProvider& targetDataProvider) {

        const ui32 numClasses = *targetDataProvider.GetTargetClassCount();
        TConstArrayRef<float> target = *targetDataProvider.GetTarget();
        TVector<ui32> classes;
        classes.resize(target.size());

        for (ui32 i = 0; i < target.size(); i++) {
            classes[i] = static_cast<ui32>(target[i]);
        }
        return MakeIntrusive<TTextClassificationTarget>(std::move(classes), numClasses);
    }

    static TFeatureEstimators CreateEstimators(
        TConstArrayRef<EFeatureCalcerType> estimatorsTypes,
        TTrainingDataProviders pools) {

        TFeatureEstimators estimators;
        CB_ENSURE(
            !AnyOf(estimatorsTypes, IsEmbeddingFeatureEstimator),
            "Embedding features cannot be calculated yet"
        );

        auto learnTarget = CreateTextClassificationTarget(*pools.Learn->TargetData);
        pools.Learn->MetaInfo.FeaturesLayout->IterateOverAvailableFeatures<EFeatureType::Text>(
            [&estimators, &estimatorsTypes, &pools, &learnTarget](TTextFeatureIdx textFeatureIdx){
                auto learnTexts = CreateTextDataSet(*pools.Learn->ObjectsData, textFeatureIdx);

                TVector<TTextDataSetPtr> testTexts;
                for (const auto& testDataProvider : pools.Test) {
                    testTexts.emplace_back(CreateTextDataSet(*testDataProvider->ObjectsData, textFeatureIdx));
                }

                TEmbeddingPtr embedding;
                auto offlineEstimators = CreateEstimators(estimatorsTypes, embedding, learnTexts, testTexts);
                for (auto&& estimator : offlineEstimators) {
                    estimators.FeatureEstimators.emplace_back(std::move(estimator));
                }

                auto onlineEstimators = CreateEstimators(estimatorsTypes, embedding, learnTarget, learnTexts, testTexts);
                for (auto&& estimator : onlineEstimators) {
                    estimators.OnlineFeatureEstimators.emplace_back(std::move(estimator));
                }
            }
        );

        return estimators;
    }


    TTrainingDataProviders GetTrainingData(
        TDataProviders srcData,
        const TMaybe<TString>& bordersFile, // load borders from it if specified
        bool ensureConsecutiveLearnFeaturesDataForCpu,
        bool allowWriteFiles,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr, then create it
        NCatboostOptions::TCatBoostOptions* params,
        TLabelConverter* labelConverter,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand) {

        TTrainingDataProviders trainingData;

        TMaybe<float> targetBorder = params->DataProcessingOptions->TargetBorder;
        trainingData.Learn = GetTrainingData(
            std::move(srcData.Learn),
            /*isLearnData*/ true,
            "learn",
            bordersFile,
            /*unloadCatFeaturePerfectHashFromRamIfPossible*/ srcData.Test.empty(),
            ensureConsecutiveLearnFeaturesDataForCpu,
            allowWriteFiles,
            quantizedFeaturesInfo,
            params,
            labelConverter,
            &targetBorder,
            localExecutor,
            rand
        );

        quantizedFeaturesInfo = trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo();

        for (auto testIdx : xrange(srcData.Test.size())) {
            trainingData.Test.push_back(
                GetTrainingData(
                    std::move(srcData.Test[testIdx]),
                    /*isLearnData*/ false,
                    TStringBuilder() << "test #" << testIdx,
                    Nothing(), // borders already loaded
                    /*unloadCatFeaturePerfectHashFromRamIfPossible*/ (testIdx + 1) == srcData.Test.size(),
                    /*ensureConsecutiveFeaturesDataForCpu*/ false, // not needed for test
                    allowWriteFiles,
                    quantizedFeaturesInfo,
                    params,
                    labelConverter,
                    &targetBorder,
                    localExecutor,
                    rand
                )
            );
        }


        if (trainingData.Learn->MetaInfo.FeaturesLayout->GetTextFeatureCount() > 0 &&
            params->TextFeatureOptions->FeatureEstimators->size() > 0) {
            CB_ENSURE(
                IsClassificationObjective(params->LossFunctionDescription->LossFunction),
                "Computation of online text features is supported only for classification task"
            );
            trainingData.FeatureEstimators = CreateEstimators(params->TextFeatureOptions->FeatureEstimators.Get(), trainingData);
        }

        if (params->MetricOptions->EvalMetric.IsSet() && (srcData.Test.size() > 0)) {
            CheckCompatibilityWithEvalMetric(
                params->MetricOptions->EvalMetric,
                *trainingData.Test.back(),
                GetApproxDimension(*params, *labelConverter)
            );
        }


        return trainingData;
    }

    TConstArrayRef<TString> GetTargetForStratifiedSplit(const TDataProvider& dataProvider) {
        auto maybeTarget = dataProvider.RawTargetData.GetTarget();
        CB_ENSURE(maybeTarget, "Cannot do stratified split: Target data is unavailable");
        return *maybeTarget;
    }

    TConstArrayRef<float> GetTargetForStratifiedSplit(const TTrainingDataProvider& dataProvider) {
        return *dataProvider.TargetData->GetTarget();
    }
}
