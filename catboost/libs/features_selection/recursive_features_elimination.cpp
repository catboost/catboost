#include "recursive_features_elimination.h"

#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/fstr/shap_values.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/full_model_saver.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/path_helpers.h>
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/hash_set.h>
#include <util/generic/typetraits.h>

#include <limits>
#include <utility>

using namespace NCatboostOptions;


namespace NCB {
    // At each step approximately the same percent of entities are eliminated
    static TVector<ui32> CalcNumEntitiesToEliminateBySteps(
        size_t numberOfEntitiesForSelect,
        int numberOfEntitiesToSelect,
        size_t steps,
        TStringBuf entitiesName
    ) {
        const double p = pow(
            (double)numberOfEntitiesToSelect / numberOfEntitiesForSelect,
            1.0 / steps
        );
        TVector<double> preciseValues(steps);
        for (auto step : xrange(steps)) {
            preciseValues[step] = numberOfEntitiesForSelect * pow(p, step) * (1 - p);
        }
        TVector<ui32> roundedValues(steps);
        double rem = 0.0;
        for (auto step : xrange(steps)) {
            const double preciseValue = preciseValues[step] + rem;
            roundedValues[step] = static_cast<ui32>(round(preciseValue));
            rem = preciseValue - roundedValues[step];
        }
        CATBOOST_DEBUG_LOG << entitiesName << " will be eliminated by:";
        for (auto value : roundedValues) {
            CATBOOST_DEBUG_LOG << " " << value;
        }
        CATBOOST_DEBUG_LOG << Endl;
        const auto nEntitiesToEliminate = numberOfEntitiesForSelect - numberOfEntitiesToSelect;
        Y_ASSERT(Accumulate(roundedValues, (ui32)0) == nEntitiesToEliminate);
        return roundedValues;
    }

    static TVector<ui32> CalcNumEntitiesToEliminateBySteps(
        const TFeaturesSelectOptions& featuresSelectOptions
    ) {
        if (featuresSelectOptions.Grouping == EFeaturesSelectionGrouping::Individual) {
            return CalcNumEntitiesToEliminateBySteps(
                featuresSelectOptions.FeaturesForSelect->size(),
                featuresSelectOptions.NumberOfFeaturesToSelect.Get(),
                featuresSelectOptions.Steps,
                "Features"
            );
        } else { // ByTags
            return CalcNumEntitiesToEliminateBySteps(
                featuresSelectOptions.FeaturesTagsForSelect->size(),
                featuresSelectOptions.NumberOfFeaturesTagsToSelect.Get(),
                featuresSelectOptions.Steps,
                "Features tags"
            );
        }
    }

    namespace {
        class TFeaturesSelectionCallbacks : public ITrainingCallbacks {
        public:
            TFeaturesSelectionCallbacks(
                const TFeaturesSelectOptions& featuresSelectOptions,
                TFeaturesSelectionSummary* summary
            )
                : FeaturesSelectOptions(featuresSelectOptions)
                , Summary(summary)
            {
            }

            bool IsContinueTraining(const TMetricsAndTimeLeftHistory& /*history*/) override {
                return true;
            }

            void OnSaveSnapshot(const NJson::TJsonValue& /*processors*/, IOutputStream* snapshot) override {
                Summary->Save(snapshot);
                NJson::TJsonValue options;
                FeaturesSelectOptions.Save(&options);
                ::SaveMany(snapshot, options);
            }

            bool OnLoadSnapshot(IInputStream* snapshot) override {
                if (!IsNextLoadValid) {
                    return false;
                }
                Summary->Load(snapshot);
                NJson::TJsonValue options;
                ::LoadMany(snapshot, options);
                TFeaturesSelectOptions featuresSelectOptions;
                featuresSelectOptions.Load(options);
                CB_ENSURE(
                    featuresSelectOptions == FeaturesSelectOptions,
                    "Current features selection options differ from options in snapshot"
                );
                FeaturesSelectOptions = featuresSelectOptions;
                IsNextLoadValid = false;
                return true;
            }

            void LoadSnapshot(ETaskType taskType, const TString& snapshotFile) {
                TProgressHelper progressHelper(ToString(taskType));
                IsNextLoadValid = true;
                progressHelper.CheckedLoad(
                    snapshotFile,
                    [&](TIFStream* input) {
                        OnLoadSnapshot(input);
                    });
                IsNextLoadValid = true;
            }

        private:
            TFeaturesSelectOptions FeaturesSelectOptions;
            TFeaturesSelectionSummary* const Summary;
            bool IsNextLoadValid = false;
        };

        class TFeaturesSelectionLossGraphBuilder {
        public:
            TFeaturesSelectionLossGraphBuilder(TFeaturesSelectionLossGraph* lossGraph)
                : LossGraph(lossGraph)
                {}

            void AddEstimatedPoint(ui32 removedEntitiesCount, double lossValue) {
                Y_ASSERT(
                    LossGraph->RemovedEntitiesCount.empty() ||
                    LossGraph->RemovedEntitiesCount.back() < removedEntitiesCount
                );
                LossGraph->RemovedEntitiesCount.push_back(removedEntitiesCount);
                LossGraph->LossValues.push_back(lossValue);
            }

            void AddPrecisePoint(ui32 removedEntitiesCount, double lossValue) {
                Y_ASSERT(
                    LossGraph->RemovedEntitiesCount.empty() ||
                    LossGraph->RemovedEntitiesCount.back() <= removedEntitiesCount
                );
                if (!LossGraph->RemovedEntitiesCount.empty() &&
                    LossGraph->RemovedEntitiesCount.back() == removedEntitiesCount)
                {
                    AdaptLossGraphValues(lossValue);
                } else {
                    LossGraph->RemovedEntitiesCount.push_back(removedEntitiesCount);
                    LossGraph->LossValues.push_back(lossValue);
                    LossGraph->MainIndices.push_back(LossGraph->LossValues.size() - 1);
                }
            }
        private:
            TFeaturesSelectionLossGraph* const LossGraph;

        private:
            void AdaptLossGraphValues(double lossValue) {
                Y_ASSERT(!LossGraph->LossValues.empty());

                const double expectedLossValue = LossGraph->LossValues.back();
                const double prevLossValue = LossGraph->LossValues[LossGraph->MainIndices.back()];
                const double expectedChange = expectedLossValue - prevLossValue;
                const double realChange = lossValue - prevLossValue;
                if (Abs(expectedChange) > 1e-9) {
                    const double coef = realChange / expectedChange;
                    CATBOOST_DEBUG_LOG << "Graph adaptation coef: " << coef << Endl;
                    size_t idx = LossGraph->LossValues.size() - 1;
                    for (; idx > LossGraph->MainIndices.back(); --idx) {
                        LossGraph->LossValues[idx]
                            = prevLossValue + (LossGraph->LossValues[idx] - prevLossValue) * coef;
                    }
                } else {
                    const double changePerFeature
                        = realChange / (LossGraph->LossValues.size() - LossGraph->MainIndices.back() - 1);
                    CATBOOST_DEBUG_LOG << "Expected change is 0, real change per feature "
                        << changePerFeature << Endl;
                    size_t idx = LossGraph->MainIndices.back() + 1;
                    for (; idx < LossGraph->LossValues.size(); ++idx) {
                        LossGraph->LossValues[idx] = LossGraph->LossValues[idx - 1] + changePerFeature;
                    }
                }
                LossGraph->MainIndices.push_back(LossGraph->LossValues.size() - 1);
            }
        };

        class TFeaturesSelectionLossGraphBuilders {
        public:
            explicit TFeaturesSelectionLossGraphBuilders(TFeaturesSelectionSummary* summary)
                : ForFeatures(&(summary->FeaturesLossGraph))
                , ForFeaturesTags(&(summary->FeaturesTagsLossGraph))
                , ForFeaturesTagsCost(&(summary->FeaturesTagsCostGraph))
            {}

        public:
            TFeaturesSelectionLossGraphBuilder ForFeatures;
            TFeaturesSelectionLossGraphBuilder ForFeaturesTags;
            TFeaturesSelectionLossGraphBuilder ForFeaturesTagsCost;
        };

        class TSelectSet {
        public:
            TSelectSet(
                const TFeaturesSelectOptions& featuresSelectOptions,
                const THashMap<TString, NCB::TTagDescription>& tagsMap,

                // may contain already eliminated features from snapshot
                const TFeaturesSelectionSummary& summary
            )
                : Grouping(featuresSelectOptions.Grouping)
            {
                if (Grouping == EFeaturesSelectionGrouping::Individual) {
                    auto eliminatedFeaturesSet = THashSet<ui32>(
                        summary.EliminatedFeatures.begin(),
                        summary.EliminatedFeatures.end()
                    );
                    for (auto featureIdx : featuresSelectOptions.FeaturesForSelect.Get()) {
                        if (!eliminatedFeaturesSet.contains(featureIdx)) {
                            Features.insert(featureIdx);
                        }
                    }
                } else { // ByTags
                    auto eliminatedFeaturesTagsSet = THashSet<TString>(
                        summary.EliminatedFeaturesTags.begin(),
                        summary.EliminatedFeaturesTags.end()
                    );
                    for (const auto& featuresTag : featuresSelectOptions.FeaturesTagsForSelect.Get()) {
                        if (!eliminatedFeaturesTagsSet.contains(featuresTag)) {
                            FeaturesTags.insert(featuresTag);
                            CB_ENSURE(
                                tagsMap.at(featuresTag).Cost != 0,
                                "Cost of features with tag '" << featuresTag << "' should be non-zero");
                            CurrentCostValue += tagsMap.at(featuresTag).Cost;
                        }
                    }
                }
            }

        public:
            EFeaturesSelectionGrouping Grouping;
            THashSet<ui32> Features;        // used if Grouping == Individual
            THashSet<TString> FeaturesTags; // used if Grouping == ByTags

            double CurrentLossValue = 0.0;
            double CurrentCostValue = 0.0; // used only if Grouping == ByTags
        };
    }

    template <class TEntity>
    static void SelectWorstEntity(
        const EMetricBestValue lossBestValueType,
        double currentLossValue,
        double lossBestValue,
        const THashSet<TEntity>& selectSet,
        TStringBuf entityName,
        std::function<double(typename TTypeTraits<TEntity>::TFuncParam)> getLossValueChange,
        std::function<double(typename TTypeTraits<TEntity>::TFuncParam)> getCost,
        TEntity* worstEntity,
        double* worstLossValueChange
    ) {
        THashMap<TEntity, double> entityToLossValueChange;
        for (const auto& entity : selectSet) {
            entityToLossValueChange[entity] = getLossValueChange(entity);
        }
        *worstEntity = entityToLossValueChange.begin()->first;
        double worstEntityScore = std::numeric_limits<double>::max();
        for (const auto& [entity, entityLossValueChange] : entityToLossValueChange) {
            double entityScore = entityLossValueChange;
            switch(lossBestValueType) {
                case EMetricBestValue::Min:
                    break;
                case EMetricBestValue::Max:
                    entityScore = -entityScore;
                    break;
                case EMetricBestValue::FixedValue:
                    entityScore = abs(currentLossValue + entityLossValueChange - lossBestValue)
                                 - abs(currentLossValue - lossBestValue);
                    break;
                default:
                    ythrow TCatBoostException() << "unsupported bestValue metric type";
            }
            entityScore /= getCost(entity);
            CATBOOST_DEBUG_LOG << entityName << ' ' << entity << " has score " << entityScore << Endl;

            if (entityScore < worstEntityScore) {
                *worstEntity = entity;
                worstEntityScore = entityScore;
            }
        }
        *worstLossValueChange = entityToLossValueChange.at(*worstEntity);
    }

    template <typename TCalcLoss>
    static void EliminateFeaturesBasedOnShapValues(
        const TFullModel& model,
        const TDataProviderPtr fstrPool,
        const ui32 numEntitiesToEliminateAtThisStep,
        const ECalcTypeShapValues shapCalcType,
        const TCalcLoss& calcLoss,
        const EMetricBestValue lossBestValueType,
        const float lossBestValue,
        const THashMap<TString, TTagDescription>& tagsDescription,
        TVector<TVector<double>> approx,
        TSelectSet* selectSet,
        TFeaturesSelectionSummary* summary,
        TFeaturesSelectionLossGraphBuilders* lossGraphBuilders,
        NPar::ILocalExecutor* executor
    ) {
        CATBOOST_DEBUG_LOG << "Calc Shap Values" << Endl;
        TVector<TVector<TVector<double>>> shapValues = CalcShapValuesMulti(
            model,
            *fstrPool.Get(),
            /*referenceDataset*/ nullptr,
            /*fixedFeatureParams*/ Nothing(),
            /*logPeriod*/1000000,
            EPreCalcShapValues::Auto,
            executor,
            shapCalcType
        );

        NPar::ILocalExecutor::TExecRangeParams blockParams(0, static_cast<int>(fstrPool->GetObjectCount()));
        blockParams.SetBlockCount(executor->GetThreadCount() + 1);

        CATBOOST_DEBUG_LOG << "Select features to eliminate" << Endl;
        const size_t approxDimension = approx.size();

        auto addShapValuesToApprox = [&] (ui32 featureIdx) {
            executor->ExecRange([&](ui32 docIdx) {
                for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                    approx[dimensionIdx][docIdx] += shapValues[docIdx][dimensionIdx][featureIdx];
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
        };
        auto substractShapValuesFromApprox = [&] (ui32 featureIdx) {
            executor->ExecRange([&](ui32 docIdx) {
                for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                    approx[dimensionIdx][docIdx] -= shapValues[docIdx][dimensionIdx][featureIdx];
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
        };

        for (ui32 i = 0; i < numEntitiesToEliminateAtThisStep; ++i) {
            if (selectSet->Grouping == EFeaturesSelectionGrouping::Individual) {
                ui32 worstFeatureIdx;
                double worstLossValueChange;
                SelectWorstEntity<ui32>(
                    lossBestValueType,
                    selectSet->CurrentLossValue,
                    lossBestValue,
                    selectSet->Features,
                    "Feature",
                    /*getLossValueChange*/ [&] (ui32 featureIdx) -> double {
                        substractShapValuesFromApprox(featureIdx);
                        double result = calcLoss(approx, model) - selectSet->CurrentLossValue;
                        addShapValuesToApprox(featureIdx);
                        return result;
                    },
                    /*getCost*/ [] (ui32 /*featureIdx*/) -> double { return 1.0; },
                    &worstFeatureIdx,
                    &worstLossValueChange
                );
                substractShapValuesFromApprox(worstFeatureIdx);

                CATBOOST_NOTICE_LOG << "Feature #" << worstFeatureIdx << " eliminated" << Endl;
                selectSet->Features.erase(worstFeatureIdx);
                summary->EliminatedFeatures.push_back(worstFeatureIdx);
                selectSet->CurrentLossValue += worstLossValueChange;
                lossGraphBuilders->ForFeatures.AddEstimatedPoint(
                    summary->EliminatedFeatures.size(),
                    selectSet->CurrentLossValue
                );
            } else { // ByTags
                TString worstFeaturesTag;
                double worstLossWithCostValueChange;
                SelectWorstEntity<TString>(
                    lossBestValueType,
                    selectSet->CurrentLossValue,
                    lossBestValue,
                    selectSet->FeaturesTags,
                    "Features tag",
                    /*getLossValueChange*/ [&] (const TString& featuresTag) -> double {
                        const auto& tagDescription = tagsDescription.at(featuresTag);
                        TConstArrayRef<ui32> featureIndices = tagDescription.Features;

                        executor->ExecRange(
                            [&](ui32 docIdx) {
                                for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                                    for (auto featureIdx : featureIndices) {
                                        approx[dimensionIdx][docIdx]
                                            -= shapValues[docIdx][dimensionIdx][featureIdx];
                                    }
                                }
                            },
                            blockParams,
                            NPar::TLocalExecutor::WAIT_COMPLETE
                        );
                        double result = calcLoss(approx, model) - selectSet->CurrentLossValue;
                        executor->ExecRange(
                            [&](ui32 docIdx) {
                                for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                                    for (auto featureIdx : featureIndices) {
                                        approx[dimensionIdx][docIdx]
                                            += shapValues[docIdx][dimensionIdx][featureIdx];
                                    }
                                }
                            },
                            blockParams,
                            NPar::TLocalExecutor::WAIT_COMPLETE
                        );
                        return result / tagDescription.Cost;
                    },
                    /*getCost*/ [&] (const TString& featuresTag) -> double {
                        return (double)tagsDescription.at(featuresTag).Cost;
                    },
                    &worstFeaturesTag,
                    &worstLossWithCostValueChange
                );

                const auto& worstTagDescription = tagsDescription.at(worstFeaturesTag);
                TConstArrayRef<ui32> featureIndices = worstTagDescription.Features;
                for (auto featureIdx : featureIndices) {
                    summary->EliminatedFeatures.push_back(featureIdx);
                    substractShapValuesFromApprox(featureIdx);
                    selectSet->CurrentLossValue = calcLoss(approx, model);
                    lossGraphBuilders->ForFeatures.AddEstimatedPoint(
                        summary->EliminatedFeatures.size(),
                        selectSet->CurrentLossValue
                    );
                }

                CATBOOST_NOTICE_LOG << "Features tag \"" << worstFeaturesTag << "\" eliminated" << Endl;
                selectSet->FeaturesTags.erase(worstFeaturesTag);
                summary->EliminatedFeaturesTags.push_back(worstFeaturesTag);
                lossGraphBuilders->ForFeaturesTags.AddEstimatedPoint(
                    summary->EliminatedFeaturesTags.size(),
                    selectSet->CurrentLossValue
                );
                selectSet->CurrentCostValue -= worstTagDescription.Cost;
                lossGraphBuilders->ForFeaturesTagsCost.AddEstimatedPoint(
                    summary->EliminatedFeaturesTags.size(),
                    selectSet->CurrentCostValue
                );
            }
        }
    }


    static void EliminateFeaturesBasedOnFeatureEffect(
        const TFullModel& model,
        const TDataProviderPtr fstrPool,
        const ui32 numEntitiesToEliminateAtThisStep,
        const EFstrType fstrType,
        const ECalcTypeShapValues shapCalcType,
        const EMetricBestValue lossBestValueType,
        const THashMap<TString, TTagDescription>& tagsDescription,
        TSelectSet* selectSet,
        TFeaturesSelectionSummary* summary,
        TFeaturesSelectionLossGraphBuilders* lossGraphBuilders,
        NPar::ILocalExecutor* executor
    ) {
        CATBOOST_DEBUG_LOG << "Calc Feature Effect" << Endl;
        TVector<double> featureEffect = CalcRegularFeatureEffect(
            model,
            fstrPool,
            fstrType,
            executor,
            shapCalcType
        );

        CATBOOST_DEBUG_LOG << "Select features to eliminate" << Endl;
        auto eliminateFeature = [&] (ui32 featureIdx) {
            CATBOOST_DEBUG_LOG << "Feature #" << featureIdx << " has effect " << featureEffect[featureIdx]
                << Endl;
            CATBOOST_NOTICE_LOG << "Feature #" << featureIdx << " eliminated" << Endl;
            summary->EliminatedFeatures.push_back(featureIdx);
            if (fstrType == EFstrType::LossFunctionChange &&
                lossBestValueType != EMetricBestValue::FixedValue)
            {
                if (lossBestValueType == EMetricBestValue::Min) {
                    selectSet->CurrentLossValue += featureEffect[featureIdx];
                } else {
                    Y_ASSERT(lossBestValueType == EMetricBestValue::Max);
                    selectSet->CurrentLossValue -= featureEffect[featureIdx];
                }
                lossGraphBuilders->ForFeatures.AddEstimatedPoint(
                    summary->EliminatedFeatures.size(),
                    selectSet->CurrentLossValue
                );
            }
        };

        if (selectSet->Grouping == EFeaturesSelectionGrouping::Individual) {
            TVector<ui32> featureIndices(featureEffect.size());
            Iota(featureIndices.begin(), featureIndices.end(), 0);
            StableSortBy(featureIndices, [&](ui32 featureIdx) {
                return featureEffect[featureIdx];
            });

            ui32 eliminatedEntitiesCount = 0;
            for (ui32 featureIdx : featureIndices) {
                if (selectSet->Features.contains(featureIdx)) {
                    eliminateFeature(featureIdx);
                    selectSet->Features.erase(featureIdx);
                    if (++eliminatedEntitiesCount == numEntitiesToEliminateAtThisStep) {
                        break;
                    }
                }
            }
        } else { // ByTags
            TVector<std::pair<TString, double>> tagsWithEffect;
            for (const auto& featuresTagName : selectSet->FeaturesTags) {
                const auto& featuresTagDescription = tagsDescription.at(featuresTagName);

                double effect = 0.0;
                for (auto featureIdx : featuresTagDescription.Features) {
                    effect += featureEffect[featureIdx];
                }
                // TSelectSet guarantees featuresTagDescription.Cost != 0
                tagsWithEffect.push_back(std::pair{featuresTagName, effect / featuresTagDescription.Cost});
            }

            StableSortBy(tagsWithEffect, [](const auto& pair) { return pair.second; } );

            ui32 eliminatedEntitiesCount = 0;
            for (const auto& [featuresTagName, effect] : tagsWithEffect) {
                const auto& tagDescription = tagsDescription.at(featuresTagName);

                for (auto featureIdx : tagDescription.Features) {
                    eliminateFeature(featureIdx);
                }
                CATBOOST_DEBUG_LOG << "Features tag \"" << featuresTagName << "\" has effect " << effect
                    << Endl;
                CATBOOST_NOTICE_LOG << "Features tag \"" << featuresTagName << "\" eliminated" << Endl;
                selectSet->FeaturesTags.erase(featuresTagName);
                summary->EliminatedFeaturesTags.push_back(featuresTagName);
                if (fstrType == EFstrType::LossFunctionChange &&
                    lossBestValueType != EMetricBestValue::FixedValue)
                {
                    lossGraphBuilders->ForFeaturesTags.AddEstimatedPoint(
                        summary->EliminatedFeaturesTags.size(),
                        selectSet->CurrentLossValue
                    );
                }
                selectSet->CurrentCostValue -= tagDescription.Cost;
                lossGraphBuilders->ForFeaturesTagsCost.AddEstimatedPoint(
                    summary->EliminatedFeaturesTags.size(),
                    selectSet->CurrentCostValue
                );

                if (++eliminatedEntitiesCount == numEntitiesToEliminateAtThisStep) {
                    break;
                }
            }
        }
    }


    static TFullModel TrainModel(
        const TCatBoostOptions& catBoostOptions,
        const TOutputFilesOptions& outputFileOptions,
        const TLabelConverter& labelConverter,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TTrainingDataProviders& trainingData,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        TFeaturesSelectionCallbacks* callbacks,
        NPar::ILocalExecutor* executor
    ) {
        TFullModel model;
        THolder<IModelTrainer> modelTrainerHolder(TTrainerFactory::Construct(catBoostOptions.GetTaskType()));
        TRestorableFastRng64 rnd(catBoostOptions.RandomSeed);
        const auto defaultCustomCallbacks = MakeHolder<TCustomCallbacks>(Nothing());
        modelTrainerHolder->TrainModel(
            TTrainModelInternalOptions(),
            catBoostOptions,
            outputFileOptions,
            /*objectiveDescriptor*/ Nothing(),
            evalMetricDescriptor,
            trainingData,
            /*precomputedSingleOnlineCtrDataForSingleFold*/ Nothing(),
            labelConverter,
            callbacks,
            defaultCustomCallbacks.Get(),
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            TDataProviders(),
            executor,
            &rnd,
            &model,
            evalResultPtrs,
            /*metricsAndTimeHistory*/ metricsAndTimeHistory,
            /*dstLearnProgress*/ nullptr
        );
        return model;
    }


    static THashMap<TString, NCB::TTagDescription> GetTagsMap(
        const TCatBoostOptions& catBoostOptions,
        const TFeaturesLayout& featuresLayout
    ) {
        if (catBoostOptions.PoolMetaInfoOptions->Tags.IsSet()) {
            return catBoostOptions.PoolMetaInfoOptions->Tags.Get();
        } else {
            THashMap<TString, NCB::TTagDescription> result;
            for (const auto& [tagName, featuresIndices] : featuresLayout.GetTagToExternalIndices()) {
                result.emplace(tagName, NCB::TTagDescription(featuresIndices));
            }
            return result;
        }
    }


    TFeaturesSelectionSummary DoRecursiveFeaturesElimination(
        const TCatBoostOptions& catBoostOptions,
        const TOutputFilesOptions& initialOutputFileOptions,
        const TFeaturesSelectOptions& featuresSelectOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TDataProviderPtr fstrPool,
        const TLabelConverter& labelConverter,
        TTrainingDataProviders trainingData,
        TFullModel* dstModel,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        NPar::ILocalExecutor* executor
    ) {
        auto outputFileOptions = initialOutputFileOptions;
        outputFileOptions.ResultModelPath.Reset();

        TVector<ui32> numEntitiesToEliminateBySteps = CalcNumEntitiesToEliminateBySteps(featuresSelectOptions);

        TFeaturesSelectionSummary summary;

        const auto callbacks = MakeHolder<TFeaturesSelectionCallbacks>(
            featuresSelectOptions,
            &summary);

        if (outputFileOptions.SaveSnapshot()) {
            const auto& absoluteSnapshotPath = MakeAbsolutePath(outputFileOptions.GetSnapshotFilename());
            outputFileOptions.SetSnapshotFilename(absoluteSnapshotPath);
            if (NFs::Exists(outputFileOptions.GetSnapshotFilename())) {
                callbacks->LoadSnapshot(
                    catBoostOptions.GetTaskType(),
                    outputFileOptions.GetSnapshotFilename()
                );
            }
        }
        TFeaturesSelectionLossGraphBuilders lossGraphBuilders(&summary);

        trainingData = MakeFeatureSubsetTrainingData(summary.EliminatedFeatures, trainingData);

        int alreadyPassedSteps = 0;
        {
            size_t eliminatedEntitiesCountInSnapshot
                = (featuresSelectOptions.Grouping == EFeaturesSelectionGrouping::Individual) ?
                    summary.EliminatedFeatures.size()
                    : summary.EliminatedFeaturesTags.size();

            size_t eliminatedEntitiesCount = 0;
            while (alreadyPassedSteps < featuresSelectOptions.Steps.Get() &&
                eliminatedEntitiesCount < eliminatedEntitiesCountInSnapshot
            ) {
                eliminatedEntitiesCount += numEntitiesToEliminateBySteps[alreadyPassedSteps++];
            }
            CB_ENSURE_INTERNAL(
                eliminatedEntitiesCount == eliminatedEntitiesCountInSnapshot,
                "Inconsistent snapshot loading for features selection."
            );
        }

        const TLossDescription lossDescription = catBoostOptions.MetricOptions->ObjectiveMetric.Get();
        const size_t approxDimension = GetApproxDimension(
            catBoostOptions,
            labelConverter,
            trainingData.Learn->TargetData->GetTargetDimension()
        );
        const THolder<IMetric> loss = std::move(
            CreateMetricFromDescription(lossDescription, approxDimension)[0]
        );

        const TTargetDataProviderPtr testTarget = trainingData.Test.empty()
            ? trainingData.Learn->TargetData
            : trainingData.Test[0]->TargetData;

        const auto trainModel = [&] (bool isFinal) {
            TVector<TEvalResult> tempEvalResults(trainingData.Test.size());
            return TrainModel(
                catBoostOptions,
                outputFileOptions,
                labelConverter,
                evalMetricDescriptor,
                trainingData,
                isFinal ? evalResultPtrs : GetMutablePointers(tempEvalResults),
                isFinal ? metricsAndTimeHistory : nullptr,
                callbacks.Get(),
                executor
            );
        };

        const auto calcLoss = [&] (const auto& approx, const TFullModel& model) {
            TRestorableFastRng64 rand(0);
            const auto fstrTarget = CreateModelCompatibleProcessedDataProvider(
                *fstrPool,
                { catBoostOptions.MetricOptions->ObjectiveMetric.Get() },
                model,
                GetMonopolisticFreeCpuRam(),
                &rand,
                executor
            ).TargetData;

            return CalcMetric(
                *loss.Get(),
                fstrTarget,
                approx,
                executor
            );
        };

        const auto applyModel = [&] (const TFullModel& model) {
            return ApplyModelMulti(
                model,
                *fstrPool->ObjectsData.Get(),
                EPredictionType::RawFormulaVal,
                0,
                0,
                executor,
                fstrPool->RawTargetData.GetBaseline()
            );
        };

        EMetricBestValue lossBestValueType;
        float lossBestValue;
        loss->GetBestValue(&lossBestValueType, &lossBestValue);

        THashMap<TString, NCB::TTagDescription> tagsMap;
        if (featuresSelectOptions.Grouping == EFeaturesSelectionGrouping::ByTags) {
            tagsMap = GetTagsMap(catBoostOptions, *trainingData.GetFeaturesLayout());
        }

        TSelectSet selectSet(featuresSelectOptions, tagsMap, summary);

        for (auto step : xrange(alreadyPassedSteps, featuresSelectOptions.Steps.Get())) {
            CATBOOST_NOTICE_LOG << "Step #" << step + 1 << " out of " << featuresSelectOptions.Steps.Get()
                << Endl;
            outputFileOptions.SetTrainDir(initialOutputFileOptions.GetTrainDir() + "/model-" + ToString(step));
            const TFullModel model = trainModel(/*isFinal*/ false);
            TVector<TVector<double>> approx = applyModel(model);
            selectSet.CurrentLossValue = calcLoss(approx, model);

            lossGraphBuilders.ForFeatures.AddPrecisePoint(
                summary.EliminatedFeatures.size(),
                selectSet.CurrentLossValue
            );

            if (featuresSelectOptions.Grouping == EFeaturesSelectionGrouping::ByTags) {
                lossGraphBuilders.ForFeaturesTags.AddPrecisePoint(
                    summary.EliminatedFeaturesTags.size(),
                    selectSet.CurrentLossValue
                );
            }

            switch (featuresSelectOptions.Algorithm) {
                case EFeaturesSelectionAlgorithm::RecursiveByShapValues: {
                    EliminateFeaturesBasedOnShapValues(
                        model,
                        fstrPool,
                        numEntitiesToEliminateBySteps[step],
                        featuresSelectOptions.ShapCalcType,
                        calcLoss,
                        lossBestValueType,
                        lossBestValue,
                        tagsMap,
                        std::move(approx),
                        &selectSet,
                        &summary,
                        &lossGraphBuilders,
                        executor
                    );
                    break;
                }
                case EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange:
                case EFeaturesSelectionAlgorithm::RecursiveByLossFunctionChange: {
                    const auto recursiveByPredictionValuesChange
                        = EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange;
                    EFstrType fstrType = featuresSelectOptions.Algorithm == recursiveByPredictionValuesChange
                        ? EFstrType::PredictionValuesChange
                        : EFstrType::LossFunctionChange;
                    EliminateFeaturesBasedOnFeatureEffect(
                        model,
                        fstrPool,
                        numEntitiesToEliminateBySteps[step],
                        fstrType,
                        featuresSelectOptions.ShapCalcType,
                        lossBestValueType,
                        tagsMap,
                        &selectSet,
                        &summary,
                        &lossGraphBuilders,
                        executor
                    );
                    break;
                }
                default:
                    CB_ENSURE_INTERNAL(false, "Unsupported algorithm: " << featuresSelectOptions.Algorithm);
            }

            trainingData = MakeFeatureSubsetTrainingData(summary.EliminatedFeatures, trainingData);
        }

        if (featuresSelectOptions.TrainFinalModel.Get() ||
            (featuresSelectOptions.Algorithm == EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange))
        {
            CATBOOST_NOTICE_LOG << "Train final model" << Endl;
            outputFileOptions.SetTrainDir(initialOutputFileOptions.GetTrainDir() + "/model-final");
            const TFullModel finalModel = trainModel(/*isFinal*/ featuresSelectOptions.TrainFinalModel.Get());
            const double lossValue = calcLoss(applyModel(finalModel), finalModel);

            lossGraphBuilders.ForFeatures.AddPrecisePoint(summary.EliminatedFeatures.size(), lossValue);
            if (featuresSelectOptions.Grouping == EFeaturesSelectionGrouping::ByTags) {
                lossGraphBuilders.ForFeaturesTags.AddPrecisePoint(
                    summary.EliminatedFeaturesTags.size(),
                    lossValue
                );
            }

            if (featuresSelectOptions.TrainFinalModel.Get()) {
                if (dstModel != nullptr) {
                    *dstModel = finalModel;
                } else {
                    CATBOOST_NOTICE_LOG << "Save final model" << Endl;
                    ExportFullModel(
                        finalModel,
                        initialOutputFileOptions.GetResultModelFilename(),
                        dynamic_cast<const TObjectsDataProvider*>(trainingData.Learn->ObjectsData.Get()),
                        outputFileOptions.GetModelFormats()
                    );
                }
            }
        }


        if (featuresSelectOptions.Grouping == EFeaturesSelectionGrouping::ByTags) {
            for (const auto& featureTag : selectSet.FeaturesTags) {
                summary.SelectedFeaturesTags.push_back(featureTag);
                for (auto featureIdx : tagsMap.at(featureTag).Features) {
                    summary.SelectedFeatures.push_back(featureIdx);
                }
            }
        } else {
            summary.SelectedFeatures = TVector<ui32>(selectSet.Features.begin(), selectSet.Features.end());
        }
        return summary;
    }
}
