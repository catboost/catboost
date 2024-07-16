#include "sage_values.h"

#include "loss_change_fstr.h"
#include "util.h"

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/objects_grouping.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/model/cpu/quantization.h>
#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/private/libs/algo/plot.h>
#include <catboost/private/libs/options/restrictions.h>
#include <catboost/private/libs/target/data_providers.h>

#include <library/cpp/accurate_accumulate/accurate_accumulate.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/function.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>
#include <util/random/shuffle.h>

using namespace NCB;

// class for TArraySubsetIndexing pointers long-living
class FullSubsetIndexingPtrWrapper {
public:
    FullSubsetIndexingPtrWrapper() = default;

    TAtomicSharedPtr<TArraySubsetIndexing<ui32>> Get(ui32 objectsCount) {
        CB_ENSURE_INTERNAL(objectsCount, "Size of TArraySubsetIndexing should be greater than 0");
        if (auto fullSubsetIndexingPtr = VSubsetIndexing[objectsCount]) {
            return fullSubsetIndexingPtr;
        }
        return VSubsetIndexing[objectsCount] =
               MakeAtomicShared<TArraySubsetIndexing<ui32>>(TFullSubset<ui32>{objectsCount});
    }

private:
    TMap<ui32, TAtomicSharedPtr<TArraySubsetIndexing<ui32>>> VSubsetIndexing;
};

class MarginalImputer {
public:
    MarginalImputer(
        const TDataProvider& dataset,
        NPar::ILocalExecutor* executor,
        TRestorableFastRng64* randPtr)
        : RandPtr(randPtr)
    {
        LocalExecutor = executor;
        ReferenceSamplesCount = dataset.GetObjectCount();

        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get());
        CB_ENSURE_INTERNAL(objectsPtr, "Zero pointer to raw objects");
        auto referenceDataProviderPtr = TRawObjectsDataProviderPtr(objectsPtr);

        // float
        ui32 floatFeaturesCount = dataset.MetaInfo.FeaturesLayout->GetFloatFeatureCount();
        for (ui32 featureIndex = 0; featureIndex < floatFeaturesCount; ++featureIndex) {
            auto featureValues = (*referenceDataProviderPtr->GetFloatFeature(featureIndex))->
                                 ExtractValues(LocalExecutor);
            ReferenceSamplesFloat.push_back(std::move(featureValues));
        }

        // categorical
        ui32 catFeaturesCount = dataset.MetaInfo.FeaturesLayout->GetCatFeatureCount();
        for (ui32 featureIndex = 0; featureIndex < catFeaturesCount; ++featureIndex) {
            auto featureValues = (*referenceDataProviderPtr->GetCatFeature(featureIndex))->
                                 ExtractValues(LocalExecutor);
            ReferenceSamplesCategorical.push_back(std::move(featureValues));
        }

        // text
        ui32 textFeaturesCount = dataset.MetaInfo.FeaturesLayout->GetTextFeatureCount();
        for (ui32 featureIndex = 0; featureIndex < textFeaturesCount; ++featureIndex) {
            auto featureValues = (*referenceDataProviderPtr->GetTextFeature(featureIndex))->
                                 ExtractValues(LocalExecutor);
            ReferenceSamplesText.push_back(std::move(featureValues));
        }

        // embedding
        ui32 embeddingFeaturesCount = dataset.MetaInfo.FeaturesLayout->GetEmbeddingFeatureCount();
        for (ui32 featureIndex = 0; featureIndex < embeddingFeaturesCount; ++featureIndex) {
            auto featureValues = (*referenceDataProviderPtr->GetEmbeddingFeature(featureIndex))->
                                 ExtractValues(LocalExecutor);
            ReferenceSamplesEmbedding.push_back(std::move(featureValues));
        }
    }

    void ImputeInplace(
        const TVector<std::pair<ui32, EFeatureType>>& features,
        FullSubsetIndexingPtrWrapper& fullSubsetIndexingPtrWrapper,
        TDataProvider* datasetPtr)
    {
        // sanity check
        CB_ENSURE_INTERNAL(ReferenceSamplesFloat.size() == datasetPtr->MetaInfo.FeaturesLayout->GetFloatFeatureCount(),
                           "Feature spaces of input and reference datasets must match, "
                           "but number of float features differ");
        CB_ENSURE_INTERNAL(ReferenceSamplesCategorical.size() == datasetPtr->MetaInfo.FeaturesLayout->GetCatFeatureCount(),
                           "Feature spaces of input and reference datasets must match, "
                           "but number of categorical features differ");
        CB_ENSURE_INTERNAL(ReferenceSamplesText.size() == datasetPtr->MetaInfo.FeaturesLayout->GetTextFeatureCount(),
                           "Feature spaces of input and reference datasets must match, "
                           "but number of text features differ");
        CB_ENSURE_INTERNAL(ReferenceSamplesEmbedding.size() == datasetPtr->MetaInfo.FeaturesLayout->GetEmbeddingFeatureCount(),
                           "Feature spaces of input and reference datasets must match, "
                           "but number of embedding features differ");

        // casting dataset to TRawObjectsDataProviderPtr
        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(datasetPtr->ObjectsData.Get());
        CB_ENSURE_INTERNAL(objectsPtr, "Zero pointer to raw objects");
        auto rawObjectsDataProviderPtr = TRawObjectsDataProviderPtr(objectsPtr);
        ui32 objectsCount = datasetPtr->GetObjectCount();

        // imputing values from their marginal distribution
        for (auto [featureIndex, featureType] : features) {
            switch (featureType) {
                case EFeatureType::Float: {
                    TVector<float> values(objectsCount);
                    for (auto& value: values) {
                        value = ReferenceSamplesFloat[featureIndex][RandPtr->Uniform(ReferenceSamplesCount)];
                    }
                    ui32 featureId = (*rawObjectsDataProviderPtr->GetFloatFeature(featureIndex))->GetId();
                    TFloatArrayValuesHolder floatValuesHolder(
                        featureId,
                        TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(values)),
                        fullSubsetIndexingPtrWrapper.Get(objectsCount).Get()
                    );
                    auto newFeatureValues = MakeHolder<TFloatArrayValuesHolder>(std::move(floatValuesHolder));
                    rawObjectsDataProviderPtr->SetFloatFeature(featureIndex, std::move(newFeatureValues));
                    break;
                }

                case EFeatureType::Categorical: {
                    TVector<ui32> values(objectsCount);
                    for (auto& value: values) {
                        value = ReferenceSamplesCategorical[featureIndex][RandPtr->Uniform(ReferenceSamplesCount)];
                    }
                    ui32 featureId = (*rawObjectsDataProviderPtr->GetCatFeature(featureIndex))->GetId();
                    THashedCatArrayValuesHolder catValuesHolder(
                        featureId,
                        TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(values)),
                        fullSubsetIndexingPtrWrapper.Get(objectsCount).Get()
                    );
                    auto newFeatureValues = MakeHolder<THashedCatArrayValuesHolder>(std::move(catValuesHolder));
                    rawObjectsDataProviderPtr->SetCatFeature(featureIndex, std::move(newFeatureValues));
                    break;
                }

                case EFeatureType::Text: {
                    TVector<TString> values(objectsCount);
                    for (auto& value: values) {
                        value = ReferenceSamplesText[featureIndex][RandPtr->Uniform(ReferenceSamplesCount)];
                    }
                    ui32 featureId = (*rawObjectsDataProviderPtr->GetTextFeature(featureIndex))->GetId();
                    TStringTextArrayValuesHolder textValuesHolder(
                        featureId,
                        TMaybeOwningConstArrayHolder<TString>::CreateOwning(std::move(values)),
                        fullSubsetIndexingPtrWrapper.Get(objectsCount).Get()
                    );
                    auto newFeatureValues = MakeHolder<TStringTextArrayValuesHolder>(std::move(textValuesHolder));
                    rawObjectsDataProviderPtr->SetTextFeature(featureIndex, std::move(newFeatureValues));
                    break;
                }

                case EFeatureType::Embedding: {
                    TVector<TConstEmbedding> values(objectsCount);
                    for (auto& value: values) {
                        value = ReferenceSamplesEmbedding[featureIndex][RandPtr->Uniform(ReferenceSamplesCount)];
                    }
                    ui32 featureId = (*rawObjectsDataProviderPtr->GetEmbeddingFeature(featureIndex))->GetId();
                    TEmbeddingArrayValuesHolder embeddingValuesHolder(
                        featureId,
                        TMaybeOwningConstArrayHolder<TConstEmbedding>::CreateOwning(std::move(values)),
                        fullSubsetIndexingPtrWrapper.Get(objectsCount).Get()
                    );
                    auto newFeatureValues = MakeHolder<TEmbeddingArrayValuesHolder>(std::move(embeddingValuesHolder));
                    rawObjectsDataProviderPtr->SetEmbeddingFeature(featureIndex, std::move(newFeatureValues));
                    break;
                }

                default:
                    CB_ENSURE_INTERNAL(false, "Unknown feature type");
            }
        }
    }

private:
    TVector<TMaybeOwningArrayHolder<float>> ReferenceSamplesFloat;
    TVector<TMaybeOwningArrayHolder<ui32>> ReferenceSamplesCategorical;
    TVector<TMaybeOwningArrayHolder<TString>> ReferenceSamplesText;
    TVector<TMaybeOwningArrayHolder<TConstEmbedding>> ReferenceSamplesEmbedding;
    TRestorableFastRng64* RandPtr;
    ui32 ReferenceSamplesCount;
    NPar::ILocalExecutor* LocalExecutor;
};

TVector<std::pair<ui32, EFeatureType>> GenerateFeaturesPermutation(TFeaturesLayoutPtr featuresLayout,
                                                                   TRestorableFastRng64* randPtr) {
    TVector<std::pair<ui32, EFeatureType>> featuresPermutation;

    for (ui32 i = 0; i < featuresLayout->GetFloatFeatureCount(); ++i) {
        featuresPermutation.emplace_back(i, EFeatureType::Float);
    }

    for (ui32 i = 0; i < featuresLayout->GetCatFeatureCount(); ++i) {
        featuresPermutation.emplace_back(i, EFeatureType::Categorical);
    }

    for (ui32 i = 0; i < featuresLayout->GetTextFeatureCount(); ++i) {
        featuresPermutation.emplace_back(i, EFeatureType::Text);
    }

    for (ui32 i = 0; i < featuresLayout->GetEmbeddingFeatureCount(); ++i) {
        featuresPermutation.emplace_back(i, EFeatureType::Embedding);
    }

    Shuffle(featuresPermutation.begin(), featuresPermutation.end(), *randPtr);

    return featuresPermutation;
}

TDataProvider GetRandomDatasetBatch(
    const TDataProvider& dataset,
    size_t batchSize,
    TRestorableFastRng64* randPtr,
    NPar::ILocalExecutor* localExecutor)
{
    if (dataset.ObjectsGrouping->IsTrivial()) {
        TVector<ui32> indices(dataset.GetObjectCount());
        Iota(indices.begin(), indices.end(), 0);
        PartialShuffle(indices.begin(), indices.end(), batchSize, *randPtr);

        auto subset = dataset.GetSubset(
            GetSubset(
                dataset.ObjectsGrouping.Get(),
                std::move(TArraySubsetIndexing<ui32>(TVector<ui32>(indices.begin(), indices.begin() + batchSize))),
                EObjectsOrder::Ordered
            ),
            GetMonopolisticFreeCpuRam(),
            localExecutor
        );

        return *subset;
    } else {
        TVector<ui32> groupIndices(dataset.ObjectsGrouping->GetGroupCount());
        Iota(groupIndices.begin(), groupIndices.end(), 0);
        Shuffle(groupIndices.begin(), groupIndices.end(), *randPtr);

        TVector<ui32> batchGroupIndices;
        ui32 realBatchSize = 0;
        ui32 groupShuffledIndex = 0;
        while (realBatchSize < batchSize) {
            realBatchSize += dataset.ObjectsGrouping->GetGroup(groupIndices[groupShuffledIndex]).GetSize();
            ++groupShuffledIndex;
        }

        auto subset = dataset.GetSubset(
            GetSubset(
                dataset.ObjectsGrouping.Get(),
                std::move(TArraySubsetIndexing<ui32>(TVector<ui32>(groupIndices.begin(), groupIndices.begin() + groupShuffledIndex))),
                EObjectsOrder::Ordered
            ),
            GetMonopolisticFreeCpuRam(),
            localExecutor
        );

        return *subset;
    }
}

double CalculateModelLoss(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TVector<THolder<IMetric>>& metrics,
    TRestorableFastRng64* randPtr,
    NPar::ILocalExecutor* localExecutor)
{
    TMetricsPlotCalcer estimator(
        model,
        metrics,
        /*tmpDir*/ "",  // fictive parameter
        model.GetTreeCount() - 1,
        model.GetTreeCount(),
        /*step*/ 1,
        /*processIterationStep*/ 1,
        localExecutor
    );

    auto datasetProvider = CreateModelCompatibleProcessedDataProvider(
        dataset,
        {},
        model,
        GetMonopolisticFreeCpuRam(),
        randPtr,
        localExecutor
    );
    estimator.ProceedDataSetForAdditiveMetrics(datasetProvider);

    return estimator.GetMetricsScore()[0][0];
}

namespace Statistics {

double Mean(const TVector<double>& values) {
    return FastAccumulate(values) / values.size();
}

double Std(const TVector<double>& values) {
    double mean = Mean(values);
    auto valuesDeviation = values;
    for (auto& value : valuesDeviation) {
        value = (value - mean) * (value - mean);
    }
    return sqrt(Mean(valuesDeviation));
}

} // namespace Statistics

bool CheckIfAllSageValuesConverged(const TVector<TVector<double>>& sageValues, double threshold) {
    double eps = 1e-12;
    double maxAbsSageValue = 0;
    double maxConfidenceIntervalHalfLength = 0;
    for (const auto& sageValue : sageValues) {
        double meanSageValue = Statistics::Mean(sageValue);
        maxAbsSageValue = Max(maxAbsSageValue, Abs(meanSageValue));
        double standardDeviation = Statistics::Std(sageValue);
        double currentConfidenceIntervalHalfLength = 1.96 * standardDeviation / sqrt(sageValue.size());
        maxConfidenceIntervalHalfLength = Max(maxConfidenceIntervalHalfLength,
                                              currentConfidenceIntervalHalfLength);
    }

    return maxConfidenceIntervalHalfLength / (maxAbsSageValue + eps) <= threshold;
}

TVector<TVector<double>> CalcSageValues(
    const TFullModel& model,
    const TDataProvider& dataset,
    int logPeriod,
    NPar::ILocalExecutor* localExecutor,
    size_t nSamples,
    size_t batchSize,
    bool detectConvergence)
{
    CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1, "Model must not be trained for multiclassification");

    // setting algorithm params
    TRestorableFastRng64 rand(228);
    size_t featuresCount = dataset.MetaInfo.GetFeatureCount();
    batchSize = Min(batchSize, size_t(dataset.GetObjectCount()));
    auto featuresLayout = dataset.MetaInfo.FeaturesLayout;
    const double convergenceThreshold = 0.1;

    FullSubsetIndexingPtrWrapper fullSubsetIndexingPtrWrapper;
    MarginalImputer imputer(dataset, localExecutor, &rand);

    // creating loss holder
    NCatboostOptions::TLossDescription metricDescription;
    NCatboostOptions::TLossDescription lossDescription;
    bool needYetiRankPairs = false;
    THolder<IMetric> metric;

    CreateMetricAndLossDescriptionForLossChange(
        model,
        &metricDescription,
        &lossDescription,
        &needYetiRankPairs,
        &metric
    );

    CB_ENSURE_INTERNAL(metric->IsAdditiveMetric(), "Loss function must be additive");

    TVector<THolder<IMetric>> metrics;
    metrics.push_back(std::move(metric));

    // main algorithm: calculating sage values
    TImportanceLogger samplingIterationsLogger(nSamples, "sampling iterations passed",
                                               "Calculating SAGE values...", logPeriod);
    TProfileInfo samplingItertionsProfile(nSamples);
    TVector<TVector<double>> sageValues(featuresCount, TVector<double>{0});
    for (size_t i = 0; i < nSamples; ++i) {
        samplingItertionsProfile.StartIterationBlock();

        // sampling batch of dataset elements
        auto datasetBatch = GetRandomDatasetBatch(dataset, batchSize, &rand, localExecutor);

        // generating features permutation
        auto featuresPermutation = GenerateFeaturesPermutation(featuresLayout, &rand);

        // running approximation algorithm
        double previousLoss = CalculateModelLoss(model, datasetBatch, metrics, &rand, localExecutor);
        for (size_t j = 0; j < featuresCount; ++j) {
            // preparing dataset, sampling disabled features
            imputer.ImputeInplace({featuresPermutation[j]}, fullSubsetIndexingPtrWrapper, &datasetBatch);

            // calculting loss and updating sage value
            double currentLoss = CalculateModelLoss(model, datasetBatch, metrics, &rand, localExecutor);
            ui32 externalFeatureIndex = featuresLayout->GetExternalFeatureIdx(featuresPermutation[j].first,
                                                                              featuresPermutation[j].second);
            sageValues[externalFeatureIndex].push_back(currentLoss - previousLoss);
            previousLoss = currentLoss;
        }

        // checking for convergence if needed
        if (detectConvergence && CheckIfAllSageValuesConverged(sageValues, convergenceThreshold)) {
            CATBOOST_INFO_LOG << "Sage Values Have Converged" << Endl;
            break;
        }

        samplingItertionsProfile.FinishIterationBlock(1);
        auto profileResults = samplingItertionsProfile.GetProfileResults();
        samplingIterationsLogger.Log(profileResults);
    }

    for (auto& sageValue : sageValues) {
        sageValue = {Statistics::Mean(sageValue)};
    }

    return sageValues;
}

void CalcAndOutputSageValues(
    const TFullModel& model,
    const TDataProvider& dataset,
    int logPeriod,
    const TString& outputPath,
    NPar::ILocalExecutor* localExecutor,
    size_t nSamples,
    size_t batchSize,
    bool detectConvergence)
{
    TFileOutput out(outputPath);

    TVector<TVector<double>> sageValues = CalcSageValues(
        model,
        dataset,
        logPeriod,
        localExecutor,
        nSamples,
        batchSize,
        detectConvergence
    );

    TVector<ui32> indices(sageValues.size());
    Iota(indices.begin(), indices.end(), 0);
    StableSortBy(indices.begin(), indices.end(), [sageValues](size_t index){ return -sageValues[index][0]; });

    for (size_t i = 0; i < sageValues.size(); ++i) {
        out << sageValues[indices[i]][0] << ' ' << indices[i] << '\n';
    }
}
