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
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/hash_set.h>

#include <limits>

using namespace NCatboostOptions;


namespace NCB {
    // At each step approximately the same percent of features are eliminated
    static TVector<ui32> CalcNumFeaturesToEliminateBySteps(
        const TFeaturesSelectOptions& featuresSelectOptions
    ) {
        const double p = pow(
            (double)featuresSelectOptions.NumberOfFeaturesToSelect / featuresSelectOptions.FeaturesForSelect->size(),
            1.0 / featuresSelectOptions.Steps
        );
        TVector<double> preciseValues(featuresSelectOptions.Steps);
        for (auto step : xrange(featuresSelectOptions.Steps.Get())) {
            preciseValues[step] = featuresSelectOptions.FeaturesForSelect->size() * pow(p, step) * (1 - p);
        }
        TVector<ui32> roundedValues(featuresSelectOptions.Steps);
        double rem = 0.0;
        for (auto step : xrange(featuresSelectOptions.Steps.Get())) {
            const double preciseValue = preciseValues[step] + rem;
            roundedValues[step] = static_cast<ui32>(round(preciseValue));
            rem = preciseValue - roundedValues[step];
        }
        CATBOOST_DEBUG_LOG << "Features will be eliminated by:";
        for (auto value : roundedValues) {
            CATBOOST_DEBUG_LOG << " " << value;
        }
        CATBOOST_DEBUG_LOG << Endl;
        const auto nFeaturesToEliminate = featuresSelectOptions.FeaturesForSelect->size() - featuresSelectOptions.NumberOfFeaturesToSelect.Get();
        Y_ASSERT(Accumulate(roundedValues, (ui32)0) == nFeaturesToEliminate);
        return roundedValues;
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
                CB_ENSURE(featuresSelectOptions == FeaturesSelectOptions, "Current features selection options differ from options in snapshot");
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

            void AddEstimatedPoint(ui32 removedFeaturesCount, double lossValue) {
                Y_ASSERT(LossGraph->RemovedFeaturesCount.empty() || LossGraph->RemovedFeaturesCount.back() < removedFeaturesCount);
                LossGraph->RemovedFeaturesCount.push_back(removedFeaturesCount);
                LossGraph->LossValues.push_back(lossValue);
            }

            void AddPrecisePoint(ui32 removedFeaturesCount, double lossValue) {
                Y_ASSERT(LossGraph->RemovedFeaturesCount.empty() || LossGraph->RemovedFeaturesCount.back() <= removedFeaturesCount);
                if (!LossGraph->RemovedFeaturesCount.empty() && LossGraph->RemovedFeaturesCount.back() == removedFeaturesCount) {
                    AdaptLossGraphValues(lossValue);
                } else {
                    LossGraph->RemovedFeaturesCount.push_back(removedFeaturesCount);
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
                    for (size_t idx = LossGraph->LossValues.size() - 1; idx > LossGraph->MainIndices.back(); --idx) {
                        LossGraph->LossValues[idx] = prevLossValue + (LossGraph->LossValues[idx] - prevLossValue) * coef;
                    }
                } else {
                    const double changePerFeature = realChange / (LossGraph->LossValues.size() - LossGraph->MainIndices.back() - 1);
                    CATBOOST_DEBUG_LOG << "Expected change is 0, real change per feature " << changePerFeature << Endl;
                    for (size_t idx = LossGraph->MainIndices.back() + 1; idx < LossGraph->LossValues.size(); ++idx) {
                        LossGraph->LossValues[idx] = LossGraph->LossValues[idx - 1] + changePerFeature;
                    }
                }
                LossGraph->MainIndices.push_back(LossGraph->LossValues.size() - 1);
            }
        };
    }


    template <typename TCalcLoss>
    static void EliminateFeaturesBasedOnShapValues(
        const TFullModel& model,
        const TDataProviderPtr fstrPool,
        const ui32 numFeaturesToEliminateAtThisStep,
        const ECalcTypeShapValues shapCalcType,
        double currentLossValue,
        const TCalcLoss& calcLoss,
        const EMetricBestValue lossBestValueType,
        const float lossBestValue,
        TVector<TVector<double>> approx,
        THashSet<ui32>* featuresForSelectSet,
        TFeaturesSelectionSummary* summary,
        TFeaturesSelectionLossGraphBuilder* lossGraphBuilder,
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
        for (ui32 i = 0; i < numFeaturesToEliminateAtThisStep; ++i) {
            THashMap<ui32, double> featureToLossValueChange;
            for (auto featureIdx : *featuresForSelectSet) {
                executor->ExecRange([&](ui32 docIdx) {
                    for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                        approx[dimensionIdx][docIdx] -= shapValues[docIdx][dimensionIdx][featureIdx];
                    }
                }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
                featureToLossValueChange[featureIdx] = calcLoss(approx) - currentLossValue;
                executor->ExecRange([&](ui32 docIdx) {
                    for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                        approx[dimensionIdx][docIdx] += shapValues[docIdx][dimensionIdx][featureIdx];
                    }
                }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
            }
            ui32 worstFeatureIdx = featureToLossValueChange.begin()->first;
            double worstFeatureScore = std::numeric_limits<double>::max();
            for (auto [featureIdx, featureLossValueChange] : featureToLossValueChange) {
                double featureScore = featureLossValueChange;
                switch(lossBestValueType) {
                    case EMetricBestValue::Min:
                        break;
                    case EMetricBestValue::Max:
                        featureScore = -featureScore;
                        break;
                    case EMetricBestValue::FixedValue:
                        featureScore = abs(currentLossValue + featureLossValueChange - lossBestValue)
                                     - abs(currentLossValue - lossBestValue);
                        break;
                    default:
                        ythrow TCatBoostException() << "unsupported bestValue metric type";
                }
                CATBOOST_DEBUG_LOG << "Feature #" << featureIdx << " has score " << featureScore << Endl;

                if (featureScore < worstFeatureScore) {
                    worstFeatureIdx = featureIdx;
                    worstFeatureScore = featureScore;
                }
            }
            executor->ExecRange([&](ui32 docIdx) {
                for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                    approx[dimensionIdx][docIdx] -= shapValues[docIdx][dimensionIdx][worstFeatureIdx];
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);

            CATBOOST_NOTICE_LOG << "Feature #" << worstFeatureIdx << " eliminated" << Endl;
            featuresForSelectSet->erase(worstFeatureIdx);
            summary->EliminatedFeatures.push_back(worstFeatureIdx);
            currentLossValue += featureToLossValueChange[worstFeatureIdx];
            lossGraphBuilder->AddEstimatedPoint(summary->EliminatedFeatures.size(), currentLossValue);
        }
    }


    static void EliminateFeaturesBasedOnFeatureEffect(
        const TFullModel& model,
        const TDataProviderPtr fstrPool,
        const ui32 numFeaturesToEliminateAtThisStep,
        const EFstrType fstrType,
        const ECalcTypeShapValues shapCalcType,
        double currentLossValue,
        const EMetricBestValue lossBestValueType,
        THashSet<ui32>* featuresForSelectSet,
        TFeaturesSelectionSummary* summary,
        TFeaturesSelectionLossGraphBuilder* lossGraphBuilder,
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

        TVector<ui32> featureIndices(featureEffect.size());
        Iota(featureIndices.begin(), featureIndices.end(), 0);
        SortBy(featureIndices, [&](ui32 featureIdx) {
            return featureEffect[featureIdx];
        });

        CATBOOST_DEBUG_LOG << "Select features to eliminate" << Endl;
        ui32 eliminatedFeaturesCount = 0;
        for (ui32 featureIdx : featureIndices) {
            if (featuresForSelectSet->contains(featureIdx)) {
                CATBOOST_DEBUG_LOG << "Feature #" << featureIdx << " has effect " << featureEffect[featureIdx] << Endl;
                CATBOOST_NOTICE_LOG << "Feature #" << featureIdx << " eliminated" << Endl;
                featuresForSelectSet->erase(featureIdx);
                summary->EliminatedFeatures.push_back(featureIdx);
                if (fstrType == EFstrType::LossFunctionChange && lossBestValueType != EMetricBestValue::FixedValue) {
                    if (lossBestValueType == EMetricBestValue::Min) {
                        currentLossValue += featureEffect[featureIdx];
                    } else {
                        Y_ASSERT(lossBestValueType == EMetricBestValue::Max);
                        currentLossValue -= featureEffect[featureIdx];
                    }
                    lossGraphBuilder->AddEstimatedPoint(summary->EliminatedFeatures.size(), currentLossValue);
                }
                if (++eliminatedFeaturesCount == numFeaturesToEliminateAtThisStep) {
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


    TFeaturesSelectionSummary DoRecursiveFeaturesElimination(
        const TCatBoostOptions& catBoostOptions,
        const TOutputFilesOptions& initialOutputFileOptions,
        const TFeaturesSelectOptions& featuresSelectOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const NCB::TDataProviders& pools,
        const TLabelConverter& labelConverter,
        TTrainingDataProviders trainingData,
        TFullModel* dstModel,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        NPar::ILocalExecutor* executor
    ) {
        auto outputFileOptions = initialOutputFileOptions;
        outputFileOptions.ResultModelPath.Reset();

        TVector<ui32> numFeaturesToEliminateBySteps = CalcNumFeaturesToEliminateBySteps(featuresSelectOptions);

        THashSet<ui32> featuresForSelectSet(
            featuresSelectOptions.FeaturesForSelect->begin(),
            featuresSelectOptions.FeaturesForSelect->end()
        );

        TFeaturesSelectionSummary summary;

        const auto callbacks = MakeHolder<TFeaturesSelectionCallbacks>(
            featuresSelectOptions,
            &summary);

        if (outputFileOptions.SaveSnapshot() && NFs::Exists(outputFileOptions.GetSnapshotFilename())) {
            callbacks->LoadSnapshot(catBoostOptions.GetTaskType(), outputFileOptions.GetSnapshotFilename());
        }
        TFeaturesSelectionLossGraphBuilder lossGraphBuilder(&summary.LossGraph);

        trainingData = MakeFeatureSubsetTrainingData(summary.EliminatedFeatures, trainingData);
        for (auto feature : summary.EliminatedFeatures) {
            featuresForSelectSet.erase(feature);
        }

        int alreadyPassedSteps = 0;
        {
            int eliminatedFeaturesCount = 0;
            while (alreadyPassedSteps < featuresSelectOptions.Steps.Get() &&
                eliminatedFeaturesCount < static_cast<int>(summary.EliminatedFeatures.size())
            ) {
                eliminatedFeaturesCount += numFeaturesToEliminateBySteps[alreadyPassedSteps++];
            }
            CB_ENSURE_INTERNAL(
                eliminatedFeaturesCount == static_cast<int>(summary.EliminatedFeatures.size()),
                "Inconsistent snapshot loading for features selection."
            );
        }

        const TLossDescription lossDescription = catBoostOptions.MetricOptions->ObjectiveMetric.Get();
        const size_t approxDimension = GetApproxDimension(
            catBoostOptions,
            labelConverter,
            trainingData.Learn->TargetData->GetTargetDimension()
        );
        const THolder<IMetric> loss = std::move(CreateMetricFromDescription(lossDescription, approxDimension)[0]);

        const auto& testPool = pools.Test.empty() ? pools.Learn : pools.Test[0];
        const auto fstrPool = GetSubsetForFstrCalc(testPool, executor);
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

        const auto calcLoss = [&] (const auto& approx) {
            return CalcMetric(
                *loss.Get(),
                testTarget,
                approx,
                executor
            );
        };

        const auto applyModel = [&] (const TFullModel& model) {
            return ApplyModelMulti(
                model,
                *testPool->ObjectsData.Get(),
                EPredictionType::RawFormulaVal,
                0,
                0,
                executor,
                testPool->RawTargetData.GetBaseline()
            );
        };

        EMetricBestValue lossBestValueType;
        float lossBestValue;
        loss->GetBestValue(&lossBestValueType, &lossBestValue);

        for (auto step : xrange(alreadyPassedSteps, featuresSelectOptions.Steps.Get())) {
            CATBOOST_NOTICE_LOG << "Step #" << step + 1 << " out of " << featuresSelectOptions.Steps.Get() << Endl;
            outputFileOptions.SetTrainDir(initialOutputFileOptions.GetTrainDir() + "/model-" + ToString(step));
            const TFullModel model = trainModel(/*isFinal*/ false);
            TVector<TVector<double>> approx = applyModel(model);
            double currentLossValue = calcLoss(approx);

            lossGraphBuilder.AddPrecisePoint(summary.EliminatedFeatures.size(), currentLossValue);

            switch (featuresSelectOptions.Algorithm) {
                case EFeaturesSelectionAlgorithm::RecursiveByShapValues: {
                    EliminateFeaturesBasedOnShapValues(
                        model,
                        fstrPool,
                        numFeaturesToEliminateBySteps[step],
                        featuresSelectOptions.ShapCalcType,
                        currentLossValue,
                        calcLoss,
                        lossBestValueType,
                        lossBestValue,
                        std::move(approx),
                        &featuresForSelectSet,
                        &summary,
                        &lossGraphBuilder,
                        executor
                    );
                    break;
                }
                case EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange:
                case EFeaturesSelectionAlgorithm::RecursiveByLossFunctionChange: {
                    EFstrType fstrType = featuresSelectOptions.Algorithm == EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange
                        ? EFstrType::PredictionValuesChange
                        : EFstrType::LossFunctionChange;
                    EliminateFeaturesBasedOnFeatureEffect(
                        model,
                        fstrPool,
                        numFeaturesToEliminateBySteps[step],
                        fstrType,
                        featuresSelectOptions.ShapCalcType,
                        currentLossValue,
                        lossBestValueType,
                        &featuresForSelectSet,
                        &summary,
                        &lossGraphBuilder,
                        executor
                    );
                    break;
                }
                default:
                    CB_ENSURE_INTERNAL(false, "Unsupported algorithm: " << featuresSelectOptions.Algorithm);
            }

            trainingData = MakeFeatureSubsetTrainingData(summary.EliminatedFeatures, trainingData);
        }

        if (featuresSelectOptions.TrainFinalModel.Get()) {
            CATBOOST_NOTICE_LOG << "Train final model" << Endl;
            outputFileOptions.SetTrainDir(initialOutputFileOptions.GetTrainDir() + "/model-final");
            const TFullModel finalModel = trainModel(/*isFinal*/ true);
            const double lossValue = calcLoss(applyModel(finalModel));

            lossGraphBuilder.AddPrecisePoint(summary.EliminatedFeatures.size(), lossValue);

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

        summary.SelectedFeatures = TVector<ui32>(featuresForSelectSet.begin(), featuresForSelectSet.end());
        return summary;
    }
}
