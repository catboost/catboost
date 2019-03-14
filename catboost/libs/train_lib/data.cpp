#include "data.h"

#include "approx_dimension.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/data_new/borders_io.h>
#include <catboost/libs/data_new/quantization.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/target/data_providers.h>

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
            } else {
                Y_ASSERT(params->GetTaskType() == ETaskType::GPU);

                /*
                 * if there're any cat features format should be CPU-compatible to enable final CTR
                 * calculations.
                 * TODO(akhropov): compatibility with final CTR calculation should not depend on this flag
                 */
                quantizationOptions.CpuCompatibleFormat
                    = srcData->MetaInfo.FeaturesLayout->GetCatFeatureCount() != 0;
            }
            quantizationOptions.CpuRamLimit
                = ParseMemorySizeDescription(params->SystemOptions->CpuUsedRamLimit.Get());
            quantizationOptions.AllowWriteFiles = allowWriteFiles;

            if (!quantizedFeaturesInfo) {
                quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                    *srcData->MetaInfo.FeaturesLayout,
                    params->DataProcessingOptions->IgnoredFeatures.Get(),
                    params->DataProcessingOptions->FloatFeaturesBinarization.Get(),
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
                localExecutor);

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
            rand,
            localExecutor,
            &trainingData->MetaInfo.HasPairs);

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
            localExecutor,
            rand);

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
                    localExecutor,
                    rand));
        }



        if (params->MetricOptions->EvalMetric.IsSet() && (srcData.Test.size() > 0)) {
            CheckCompatibilityWithEvalMetric(
                params->MetricOptions->EvalMetric,
                *trainingData.Test.back(),
                GetApproxDimension(*params, *labelConverter));
        }


        return trainingData;
    }

}
