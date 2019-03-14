#include "full_model_saver.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/static_ctr_provider.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/system_options.h>

#include <library/svnversion/svnversion.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/datetime/base.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>


namespace NCB {

    static void CreateTargetClasses(
        NPar::TLocalExecutor& localExecutor,
        TConstArrayRef<float> targets,
        const TVector<TTargetClassifier>& targetClassifiers,
        TVector<TVector<int>>* learnTargetClasses,
        TVector<int>* targetClassesCount
    ) {
        ui64 ctrCount = targetClassifiers.size();
        const int sampleCount = static_cast<const int>(targets.size());

        learnTargetClasses->assign(ctrCount, TVector<int>(sampleCount));
        targetClassesCount->resize(ctrCount);

        for (ui32 ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
            NPar::ParallelFor(localExecutor, 0, (ui32)sampleCount, [&](int sample) {
                (*learnTargetClasses)[ctrIdx][sample] = targetClassifiers[ctrIdx].GetTargetClass(targets[sample]);
            });

            (*targetClassesCount)[ctrIdx] = targetClassifiers[ctrIdx].GetClassesCount();
        }
    }

    static bool NeedTargetClasses(const TFullModel& coreModel) {
        return AnyOf(coreModel.ObliviousTrees.GetUsedModelCtrs(), [](const TModelCtr& modelCtr) {
            return NeedTargetClassifier(modelCtr.Base.CtrType);
        });
    }


    namespace {
        class TIncompleteData {
        public:
            TIncompleteData(
                TTrainingForCPUDataProviders&& trainingData,
                THashMap<TFeatureCombination, TProjection>&& featureCombinationToProjection,
                const TVector<TTargetClassifier>& targetClassifiers,
                ECounterCalc counterCalcMethod,
                ui32 numThreads
            )
                : TrainingData(std::move(trainingData))
                , TargetClassifiers(targetClassifiers)
                , NumThreads(numThreads)
                , FeatureCombinationToProjection(std::move(featureCombinationToProjection))
            {
                if (counterCalcMethod == ECounterCalc::SkipTest) {
                    TrainingData.Test.clear();
                }
            }

            void operator()(
                const TFullModel& coreModel,
                TDatasetDataForFinalCtrs* outDatasetDataForFinalCtrs,
                const THashMap<TFeatureCombination, TProjection>** outFeatureCombinationToProjection
            ) {
                outDatasetDataForFinalCtrs->Data = std::move(TrainingData);
                outDatasetDataForFinalCtrs->LearnPermutation = Nothing();
                outDatasetDataForFinalCtrs->Targets =
                    *outDatasetDataForFinalCtrs->Data.Learn->TargetData->GetTarget();

                *outFeatureCombinationToProjection = &FeatureCombinationToProjection;

                if (NeedTargetClasses(coreModel)) {
                    NPar::TLocalExecutor localExecutor;
                    localExecutor.RunAdditionalThreads(NumThreads - 1);

                    CreateTargetClasses(
                        localExecutor,
                        *outDatasetDataForFinalCtrs->Targets,
                        TargetClassifiers,
                        &LearnTargetClasses,
                        &TargetClassesCount
                    );
                    outDatasetDataForFinalCtrs->LearnTargetClass = &LearnTargetClasses;
                    outDatasetDataForFinalCtrs->TargetClassesCount = &TargetClassesCount;
                } else {
                    outDatasetDataForFinalCtrs->LearnTargetClass = Nothing();
                    outDatasetDataForFinalCtrs->TargetClassesCount = Nothing();
                }
            }

        private:
            TTrainingForCPUDataProviders TrainingData;

            const TVector<TTargetClassifier>& TargetClassifiers;
            ui32 NumThreads;

            TVector<TVector<int>> LearnTargetClasses;
            TVector<int> TargetClassesCount;

            THashMap<TFeatureCombination, TProjection> FeatureCombinationToProjection;
        };
    }

    TCoreModelToFullModelConverter::TCoreModelToFullModelConverter(
        const NCatboostOptions::TCatBoostOptions& options,
        const TClassificationTargetHelper& classificationTargetHelper,
        ui64 ctrLeafCountLimit,
        bool storeAllSimpleCtrs,
        EFinalCtrComputationMode finalCtrComputationMode
    )
        : NumThreads(options.SystemOptions->NumThreads)
        , FinalCtrComputationMode(finalCtrComputationMode)
        , CpuRamLimit(ParseMemorySizeDescription(options.SystemOptions->CpuUsedRamLimit.Get()))
        , CtrLeafCountLimit(ctrLeafCountLimit)
        , StoreAllSimpleCtrs(storeAllSimpleCtrs)
        , Options(options)
        , ClassificationTargetHelper(classificationTargetHelper)
    {}

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithCoreModelFrom(
         TFullModel* coreModel
    ) {
        CoreModel = coreModel;
        return *this;
    }

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithObjectsDataFrom(
        TObjectsDataProviderPtr learnObjectsData
    ) {
        LearnObjectsData = learnObjectsData;
        return *this;
    }

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithBinarizedDataComputedFrom(
         TDatasetDataForFinalCtrs&& datasetDataForFinalCtrs,
         THashMap<TFeatureCombination, TProjection>&& featureCombinationToProjection
    ) {
        if (FinalCtrComputationMode != EFinalCtrComputationMode::Skip) {
            GetBinarizedDataFunc = [
                datasetDataForFinalCtrs = std::move(datasetDataForFinalCtrs),
                featureCombinationToProjection = std::move(featureCombinationToProjection)
            ] (
                const TFullModel& /*coreModel*/,
                TDatasetDataForFinalCtrs* outDatasetDataForFinalCtrs,
                const THashMap<TFeatureCombination, TProjection>** outFeatureCombinationToProjection
            ) {
                *outDatasetDataForFinalCtrs = datasetDataForFinalCtrs;
                *outFeatureCombinationToProjection = &featureCombinationToProjection;
            };
        }
        return *this;
    }

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithBinarizedDataComputedFrom(
        TTrainingForCPUDataProviders&& trainingData,
        THashMap<TFeatureCombination, TProjection>&& featureCombinationToProjection,
        const TVector<TTargetClassifier>& targetClassifiers
    ) {
        if (FinalCtrComputationMode != EFinalCtrComputationMode::Skip) {
            GetBinarizedDataFunc = TIncompleteData(
                std::move(trainingData),
                std::move(featureCombinationToProjection),
                targetClassifiers,
                Options.CatFeatureParams.Get().CounterCalcMethod,
                NumThreads
            );
        }
        return *this;
    }

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithPerfectHashedToHashedCatValuesMap(
        const TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap
    ) {
        if (FinalCtrComputationMode != EFinalCtrComputationMode::Skip) {
            PerfectHashedToHashedCatValuesMap = perfectHashedToHashedCatValuesMap;
        }
        return *this;
    }

    void TCoreModelToFullModelConverter::Do(bool requiresStaticCtrProvider, TFullModel* dstModel) {
        DoImpl(requiresStaticCtrProvider, dstModel);
    }

    void TCoreModelToFullModelConverter::Do(
        const TString& fullModelPath,
        const TVector<EModelType>& formats,
        bool addFileFormatExtension
    ) {
        TFullModel fullModel;

        DoImpl(
            AnyOf(
                formats,
                [](EModelType format) {
                    return format == EModelType::Python || format == EModelType::Cpp || format == EModelType::Json;
                }
            ),
            &fullModel
        );

        const auto& featuresLayout = *LearnObjectsData->GetFeaturesLayout();
        TVector<TString> featureIds = featuresLayout.GetExternalFeatureIds();

        THashMap<ui32, TString> catFeaturesHashToString;
        bool haveAvailableCatFeatures = false;
        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&] (TCatFeatureIdx catFeatureIdx) {
                Y_UNUSED(catFeatureIdx);
                haveAvailableCatFeatures = true;
            }
        );
        if (haveAvailableCatFeatures) {
            catFeaturesHashToString = MergeCatFeaturesHashToString(*LearnObjectsData);
        }

        for (const auto& format: formats) {
            ExportModel(fullModel, fullModelPath, format, "", addFileFormatExtension, &featureIds, &catFeaturesHashToString);
        }
    }

    void TCoreModelToFullModelConverter::DoImpl(bool requiresStaticCtrProvider, TFullModel* dstModel) {
        CB_ENSURE_INTERNAL(CoreModel, "CoreModel has not been specified");

        if (CoreModel != dstModel) {
            *dstModel = std::move(*CoreModel);
        }
        dstModel->ModelInfo["train_finish_time"] = TInstant::Now().ToStringUpToSeconds();
        dstModel->ModelInfo["catboost_version_info"] = GetProgramSvnVersion();

        {
            NJson::TJsonValue jsonOptions(NJson::EJsonValueType::JSON_MAP);
            Options.Save(&jsonOptions);
            dstModel->ModelInfo["params"] = ToString(jsonOptions);
            for (const auto& keyValue : Options.Metadata.Get().GetMap()) {
                dstModel->ModelInfo[keyValue.first] = keyValue.second.GetString();
            }
        }

        ELossFunction lossFunction = Options.LossFunctionDescription.Get().GetLossFunction();
        if (IsMultiClassMetric(lossFunction)) {
            dstModel->ModelInfo["multiclass_params"] = ClassificationTargetHelper.Serialize();
        }

        if (FinalCtrComputationMode == EFinalCtrComputationMode::Skip) {
            return;
        }
        if (dstModel->HasValidCtrProvider()) {
            // ModelBase apparently has valid ctrs table
            // TODO(kirillovs): add here smart check for ctrprovider serialization ability
            // after implementing non-storing ctr providers
            return;
        }

        CB_ENSURE_INTERNAL(GetBinarizedDataFunc, "Need BinarizedDataFunc data specified");

        TDatasetDataForFinalCtrs datasetDataForFinalCtrs;
        const THashMap<TFeatureCombination, TProjection>* featureCombinationToProjectionMap;

        GetBinarizedDataFunc(*dstModel, &datasetDataForFinalCtrs, &featureCombinationToProjectionMap);


        CB_ENSURE_INTERNAL(
            PerfectHashedToHashedCatValuesMap,
            "PerfectHashedToHashedCatValuesMap has not been specified"
        );

        if (requiresStaticCtrProvider) {
            dstModel->CtrProvider = new TStaticCtrProvider;

            TMutex lock;

            CalcFinalCtrs(
                datasetDataForFinalCtrs,
                *featureCombinationToProjectionMap,
                dstModel->ObliviousTrees.GetUsedModelCtrBases(),
                [&dstModel, &lock](TCtrValueTable&& table) {
                    with_lock(lock) {
                        dstModel->CtrProvider->AddCtrCalcerData(std::move(table));
                    }
                }
            );

            dstModel->UpdateDynamicData();
        } else {
            dstModel->CtrProvider = new TStaticCtrOnFlightSerializationProvider(
                dstModel->ObliviousTrees.GetUsedModelCtrBases(),
                [this,
                 datasetDataForFinalCtrs = std::move(datasetDataForFinalCtrs),
                 featureCombinationToProjectionMap] (
                    const TVector<TModelCtrBase>& ctrBases,
                    TCtrDataStreamWriter* streamWriter
                ) {
                    CalcFinalCtrs(
                        datasetDataForFinalCtrs,
                        *featureCombinationToProjectionMap,
                        ctrBases,
                        [&streamWriter](TCtrValueTable&& table) {
                            // there's lock inside, so it is thread-safe
                            streamWriter->SaveOneCtr(table);
                        }
                    );
                }
            );
        }
    }

    void TCoreModelToFullModelConverter::CalcFinalCtrs(
        const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
        const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjectionMap,
        const TVector<TModelCtrBase>& ctrBases,
        std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(NumThreads - 1);

        CalcFinalCtrsAndSaveToModel(
            CpuRamLimit,
            localExecutor,
            featureCombinationToProjectionMap,
            datasetDataForFinalCtrs,
            *PerfectHashedToHashedCatValuesMap,
            CtrLeafCountLimit,
            StoreAllSimpleCtrs,
            Options.CatFeatureParams.Get().CounterCalcMethod,
            ctrBases,
            std::move(asyncCtrValueTableCallback)
        );
    };
}
