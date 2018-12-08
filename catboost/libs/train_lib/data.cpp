#include "data.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/data_new/borders_io.h>
#include <catboost/libs/data_new/quantization.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/target/data_providers.h>

#include <util/string/builder.h>


namespace NCB {

    static TVector<NCatboostOptions::TLossDescription> GetMetricDescriptions(
        const NCatboostOptions::TCatBoostOptions& params) {

        TVector<NCatboostOptions::TLossDescription> result;
        result.emplace_back(params.LossFunctionDescription);

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
        bool unloadCatFeaturePerfectHashFromRam,
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
            // We need data to be consecutive for efficient blocked permutations
            if (params->GetTaskType() == ETaskType::CPU) {
                auto quantizedForCPUObjectsDataProvider
                    = dynamic_cast<TQuantizedForCPUObjectsDataProvider*>(quantizedObjectsDataProviderPtr);
                CB_ENSURE(
                    quantizedForCPUObjectsDataProvider,
                    "Quantized objects data is not compatible with CPU task type"
                );
                if (!quantizedForCPUObjectsDataProvider->GetFeaturesArraySubsetIndexing().IsConsecutive()) {
                    // TODO(akhropov): make it work in non-shared case
                    CB_ENSURE_INTERNAL(
                        (srcData->RefCount() <= 1) && (quantizedForCPUObjectsDataProvider->RefCount() <= 1),
                        "Cannot modify QuantizedForCPUObjectsDataProvider because it's shared"
                    );
                    quantizedForCPUObjectsDataProvider->EnsureConsecutiveFeaturesData(localExecutor);
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

        if (unloadCatFeaturePerfectHashFromRam) {
            trainingData->ObjectsData->GetQuantizedFeaturesInfo()->UnloadCatFeaturePerfectHashFromRam();
        }

        auto& dataProcessingOptions = params->DataProcessingOptions.Get();

        trainingData->TargetData = CreateTargetDataProviders(
            srcData->RawTargetData,
            trainingData->ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ params->GetTaskType() == ETaskType::GPU,
            isLearnData,
            datasetName,
            GetMetricDescriptions(*params),
            &params->LossFunctionDescription.Get(),
            dataProcessingOptions.AllowConstLabel.Get(),
            dataProcessingOptions.ClassesCount.Get(),
            dataProcessingOptions.ClassWeights.Get(),
            &dataProcessingOptions.ClassNames.Get(),
            labelConverter,
            rand,
            localExecutor);

        // in case pairs were generated
        if (trainingData->TargetData.contains(TTargetDataSpecification(ETargetType::GroupPairwiseRanking))) {
            trainingData->MetaInfo.HasPairs = true;
        }

        return trainingData;
    }


    TTrainingDataProviders GetTrainingData(
        TDataProviders srcData,
        const TMaybe<TString>& bordersFile, // load borders from it if specified
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
            /*unloadCatFeaturePerfectHashFromRam*/ srcData.Test.empty(),
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
                    /*unloadCatFeaturePerfectHashFromRam*/ (testIdx + 1) == srcData.Test.size(),
                    quantizedFeaturesInfo,
                    params,
                    labelConverter,
                    localExecutor,
                    rand));
        }

        return trainingData;
    }

}
