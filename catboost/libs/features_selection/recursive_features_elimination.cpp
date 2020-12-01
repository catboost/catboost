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
    }


    static TFullModel TrainModel(
        const TCatBoostOptions& catBoostOptions,
        const TOutputFilesOptions& outputFileOptions,
        const TLabelConverter& labelConverter,
        const TTrainingDataProviders& trainingData,
        TFeaturesSelectionCallbacks* callbacks,
        NPar::ILocalExecutor* executor
    ) {
        TFullModel model;
        TVector<TEvalResult> evalResults(trainingData.Test.ysize());
        THolder<IModelTrainer> modelTrainerHolder(TTrainerFactory::Construct(catBoostOptions.GetTaskType()));
        TRestorableFastRng64 rnd(catBoostOptions.RandomSeed);
        modelTrainerHolder->TrainModel(
            TTrainModelInternalOptions(),
            catBoostOptions,
            outputFileOptions,
            /*objectiveDescriptor*/ Nothing(),
            /*evalMetricDescriptor*/ Nothing(),
            trainingData,
            /*precomputedSingleOnlineCtrDataForSingleFold*/ Nothing(),
            labelConverter,
            callbacks,
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            TDataProviders(),
            executor,
            &rnd,
            &model,
            GetMutablePointers(evalResults),
            /*metricsAndTimeHistory*/ nullptr,
            /*dstLearnProgress*/ nullptr
        );
        return model;
    }


    static void AdaptLossGraphValues(
        double lossValue,
        TFeaturesSelectionSummary* summary
    ) {
        Y_ASSERT(!summary->LossGraph.LossValues.empty());

        const double expectedLossValue = summary->LossGraph.LossValues.back();
        const double prevLossValue = summary->LossGraph.LossValues[summary->LossGraph.MainIndices.back()];
        const double expectedChange = expectedLossValue - prevLossValue;
        const double realChange = lossValue - prevLossValue;
        const double coef = Abs(expectedChange) < 1e-9 ? 0.0 : realChange / expectedChange;
        CATBOOST_DEBUG_LOG << "Graph adaptation coef: " << coef << Endl;
        for (size_t idx = summary->LossGraph.LossValues.size() - 1; idx > summary->LossGraph.MainIndices.back(); --idx) {
            summary->LossGraph.LossValues[idx] = prevLossValue + (summary->LossGraph.LossValues[idx] - prevLossValue) * coef;
        }
        summary->LossGraph.MainIndices.push_back(summary->LossGraph.LossValues.size() - 1);
    }

    TFeaturesSelectionSummary DoRecursiveFeaturesElimination(
        const TCatBoostOptions& catBoostOptions,
        const TOutputFilesOptions& outputFileOptions,
        const TFeaturesSelectOptions& featuresSelectOptions,
        const NCB::TDataProviders& pools,
        const TLabelConverter& labelConverter,
        TTrainingDataProviders trainingData,
        NPar::ILocalExecutor* executor
    ) {
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

        const TLossDescription lossDescription = catBoostOptions.LossFunctionDescription.Get();
        const size_t approxDimension = GetApproxDimension(
            catBoostOptions,
            labelConverter,
            trainingData.Learn->TargetData->GetTargetDimension()
        );
        const THolder<IMetric> loss = std::move(CreateMetricFromDescription(lossDescription, approxDimension)[0]);

        const auto& testPool = pools.Test.empty() ? pools.Learn : pools.Test[0];
        const TTargetDataProviderPtr testTarget = trainingData.Test.empty()
            ? trainingData.Learn->TargetData
            : trainingData.Test[0]->TargetData;

        const auto trainModel = [&] () {
            return TrainModel(
                catBoostOptions,
                outputFileOptions,
                labelConverter,
                trainingData,
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
                executor
            );
        };

        for (auto step : xrange(alreadyPassedSteps, featuresSelectOptions.Steps.Get())) {
            CATBOOST_INFO_LOG << "Step #" << step + 1 << " out of " << featuresSelectOptions.Steps.Get() << Endl;
            const TFullModel model = trainModel();
            TVector<TVector<double>> approx = applyModel(model);
            double currentLossValue = calcLoss(approx);

            if (summary.LossGraph.LossValues.size() > 0) {
                AdaptLossGraphValues(
                    currentLossValue,
                    &summary
                );
            } else {
                summary.LossGraph.LossValues.push_back(currentLossValue);
                summary.LossGraph.MainIndices.push_back(0);
            }

            CATBOOST_DEBUG_LOG << "Calc Shap Values" << Endl;
            TVector<TVector<TVector<double>>> shapValues = CalcShapValuesMulti(
                model,
                *testPool.Get(),
                /*referenceDataset*/ nullptr,
                /*fixedFeatureParams*/ Nothing(),
                /*logPeriod*/1000000,
                EPreCalcShapValues::Auto,
                executor,
                featuresSelectOptions.ShapCalcType
            );

            NPar::ILocalExecutor::TExecRangeParams blockParams(0, static_cast<int>(testPool->GetObjectCount()));
            blockParams.SetBlockCount(executor->GetThreadCount() + 1);

            CATBOOST_DEBUG_LOG << "Select features to eliminate" << Endl;
            TVector<ui32> eliminatedFeatures;
            for (ui32 i = 0; i < numFeaturesToEliminateBySteps[step]; ++i) {
                THashMap<ui32, double> featureToLossValueChange;
                for (auto featureIdx : featuresForSelectSet) {
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
                for (auto [featureIdx, featureLossValueChange] : featureToLossValueChange) {
                    CATBOOST_DEBUG_LOG << "Feature #" << featureIdx << " has loss function change " << featureLossValueChange << Endl;
                    if (featureLossValueChange < featureToLossValueChange[worstFeatureIdx]) {
                        worstFeatureIdx = featureIdx;
                    }
                }
                executor->ExecRange([&](ui32 docIdx) {
                    for (size_t dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                        approx[dimensionIdx][docIdx] -= shapValues[docIdx][dimensionIdx][worstFeatureIdx];
                    }
                }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);

                CATBOOST_INFO_LOG << "Feature #" << worstFeatureIdx << " eliminated" << Endl;
                featuresForSelectSet.erase(worstFeatureIdx);
                eliminatedFeatures.push_back(worstFeatureIdx);
                summary.EliminatedFeatures.push_back(worstFeatureIdx);
                currentLossValue += featureToLossValueChange[worstFeatureIdx];
                summary.LossGraph.LossValues.push_back(currentLossValue);
            }

            trainingData = MakeFeatureSubsetTrainingData(summary.EliminatedFeatures, trainingData);
        }

        if (featuresSelectOptions.TrainFinalModel.Get()) {
            CATBOOST_INFO_LOG << "Train final model" << Endl;
            const TFullModel finalModel = trainModel();
            const double lossValue = calcLoss(applyModel(finalModel));

            AdaptLossGraphValues(
                lossValue,
                &summary
            );

            CATBOOST_INFO_LOG << "Save final model" << Endl;
            ExportFullModel(
                finalModel,
                "model.bin",
                dynamic_cast<const TObjectsDataProvider*>(trainingData.Learn->ObjectsData.Get()),
                outputFileOptions.GetModelFormats()
            );
        }

        summary.SelectedFeatures = TVector<ui32>(featuresForSelectSet.begin(), featuresForSelectSet.end());
        return summary;
    }
}
