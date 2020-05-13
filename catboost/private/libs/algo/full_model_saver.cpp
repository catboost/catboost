#include "full_model_saver.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/ctr_value_table.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/model/static_ctr_provider.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/system_options.h>
#include <catboost/private/libs/target/classification_target_helper.h>

#include <library/svnversion/svnversion.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/datetime/base.h>
#include <util/generic/algorithm.h>
#include <util/generic/guid.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>


namespace {
    using namespace NCB;

    class TTextCollectionBuilder {
    public:
        TTextCollectionBuilder(
            const TFeatureEstimators& estimators,
            const TTextDigitizers& textDigitizers,
            NPar::TLocalExecutor* localExecutor
        )
            : WasBuilt(false)
            , FeatureEstimators(estimators)
            , TextDigitizers(textDigitizers)
            , PerFeatureDigitizers()
            , PerTokenizedFeatureCalcers()
            , LocalExecutor(localExecutor)
        {
            const ui32 textFeatureCount = textDigitizers.GetSourceTextsCount();
            const ui32 tokenizedFeatureCount = textDigitizers.GetDigitizedTextsCount();

            PerFeatureDigitizers.resize(textFeatureCount);
            PerTokenizedFeatureCalcers.resize(tokenizedFeatureCount);
        }

        TTextCollectionBuilder& AddFeatureEstimator(
            const TGuid& calcerId,
            TConstArrayRef<TEstimatedFeature> estimatedFeatures
        ) {
            TEstimatorSourceId estimatorSourceId = FeatureEstimators.GetEstimatorSourceFeatureIdx(calcerId);
            const ui32 textFeatureIdx = estimatorSourceId.TextFeatureId;
            const ui32 tokenizedFeatureIdx = estimatorSourceId.TokenizedFeatureId;

            TDigitizer digitizer = TextDigitizers.GetDigitizer(tokenizedFeatureIdx);
            const ui32 digitizerFlatIdx = AddDigitizer(digitizer);
            CalcerToDigitizer[calcerId] = digitizer.Id();

            TVector<ui32> localIds;
            for (const auto& estimatedFeature : estimatedFeatures) {
                localIds.push_back(estimatedFeature.LocalIndex);
            }

            const ui32 calcerFlatIdx = AddCalcer(MakeFinalFeatureCalcer(calcerId, MakeConstArrayRef(localIds)));

            AddEstimatedFeature(textFeatureIdx, calcerFlatIdx, estimatedFeatures);
            RegisterIndices(
                textFeatureIdx,
                tokenizedFeatureIdx,
                digitizerFlatIdx,
                calcerFlatIdx
            );

            return *this;
        }

        void InitializeDigitizer(
            TVector<TDigitizer>* digitizers,
            TVector<TVector<ui32>>* perFeatureDigitizers
        ) {
            digitizers->resize(DigitizerToId.size());
            for (const auto& [digitizer, id]: DigitizerToId) {
                (*digitizers)[id] = digitizer;
            }

            perFeatureDigitizers->resize(PerFeatureDigitizers.size());

            for (ui32 textFeature: xrange(PerFeatureDigitizers.size())) {
                auto& digitizersIds = (*perFeatureDigitizers)[textFeature];
                for (const auto& [tokenizedFeature, digitizerId]: PerFeatureDigitizers[textFeature]) {
                    Y_UNUSED(tokenizedFeature); // we need just sort by tokenized feature index
                    digitizersIds.push_back(digitizerId);
                }
                textFeature++;
            }
        }

        void Build(
            TTextProcessingCollection* textProcessingCollection,
            TVector<TEstimatedFeature>* estimatedFeatures
        ) {
            CB_ENSURE_INTERNAL(!WasBuilt, "TTextCollectionBuilder: Build can be done only once");
            WasBuilt = true;

            TVector<TDigitizer> digitizers;
            TVector<TVector<ui32>> perFeatureDigitizers;
            InitializeDigitizer(&digitizers, &perFeatureDigitizers);

            EraseIf(
                PerTokenizedFeatureCalcers,
                [&] (TVector<ui32>& calcers) {
                    return calcers.empty();
                }
            );

            CheckFeatureIndexes(digitizers, Calcers, perFeatureDigitizers, PerTokenizedFeatureCalcers);

            *textProcessingCollection = TTextProcessingCollection(
                digitizers, Calcers, perFeatureDigitizers, PerTokenizedFeatureCalcers
            );

            *estimatedFeatures = EstimatedFeatures;
        }

    private:
        ui32 AddDigitizer(TDigitizer digitizer) {
            if (DigitizerToId.contains(digitizer)) {
                return DigitizerToId[digitizer];
            }

            const ui32 digitizerIdx = DigitizerToId.size();
            DigitizerToId[digitizer] = digitizerIdx;
            return digitizerIdx;
        }

        ui32 AddCalcer(TTextFeatureCalcerPtr&& calcer) {
            const ui32 calcerId = Calcers.size();
            Calcers.push_back(calcer);
            return calcerId;
        }

        void AddEstimatedFeature(
            ui32 textFeatureId,
            ui32 calcerFlatIdx,
            TConstArrayRef<TEstimatedFeature> estimatedFeatures
        ) {
            const auto& calcer = Calcers[calcerFlatIdx];
            for (ui32 localIndex: xrange(calcer->FeatureCount())) {
                TEstimatedFeature estimatedFeature = TEstimatedFeature{
                    SafeIntegerCast<int>(textFeatureId),
                    calcer->Id(),
                    SafeIntegerCast<int>(localIndex)
                };
                estimatedFeature.Borders = estimatedFeatures[localIndex].Borders;
                EstimatedFeatures.push_back(estimatedFeature);
            }
        }

        void RegisterIndices(ui32 textFeatureId, ui32 tokenizedFeatureId, ui32 digitizerId, ui32 calcerId) {
            PerFeatureDigitizers[textFeatureId][tokenizedFeatureId] = digitizerId;
            PerTokenizedFeatureCalcers[tokenizedFeatureId].push_back(calcerId);
        }

        void CheckFeatureIndexes(
            const TVector<TDigitizer>& digitizers,
            const TVector<TTextFeatureCalcerPtr>& calcers,
            const TVector<TVector<ui32>>& perFeatureDigitizers,
            const TVector<TVector<ui32>>& perTokenizedFeatureCalcers
        ) {
            ui32 tokenizedFeatureId = 0;
            for (const auto& featureDigitizers: perFeatureDigitizers) {
                for (ui32 digitizerId: featureDigitizers) {
                    CB_ENSURE(
                        tokenizedFeatureId < perTokenizedFeatureCalcers.size(),
                        "Tokenized feature id=" << tokenizedFeatureId
                            << " should be less than " << LabeledOutput(perTokenizedFeatureCalcers.size())
                    );
                    const TVector<ui32>& calcersIds = perTokenizedFeatureCalcers[tokenizedFeatureId];
                    CB_ENSURE(
                        IsSorted(calcersIds.begin(), calcersIds.end()),
                        "Feature calcers must be sorted"
                    );
                    for (ui32 calcerId: calcersIds) {
                        const TGuid& calcerGuid = calcers[calcerId]->Id();
                        const auto& digitizerGuid = digitizers[digitizerId].Id();
                        CB_ENSURE(
                            CalcerToDigitizer[calcerGuid] == digitizerGuid,
                            "FeatureCalcer id=" << calcerGuid << " should be computed for "
                                << "tokenized feature id=" << tokenizedFeatureId
                                << " with tokenizer id=" << digitizerGuid.TokenizerId << " and "
                                << " with dictionary id=" << digitizerGuid.DictionaryId << Endl
                        );
                    }
                    tokenizedFeatureId++;
                }
            }
        }

        TTextFeatureCalcerPtr MakeFinalFeatureCalcer(const TGuid& calcerId, TConstArrayRef<ui32> featureIds) {
            TFeatureEstimatorPtr estimator = FeatureEstimators.GetEstimatorByGuid(calcerId);

            auto calcerHolder = estimator->MakeFinalFeatureCalcer(featureIds, LocalExecutor);
            TTextFeatureCalcerPtr calcerPtr = dynamic_cast<TTextFeatureCalcer*>(calcerHolder.Release());
            CB_ENSURE(calcerPtr, "CalcerPtr == null after MakeFinalFeatureCalcer");
            CB_ENSURE(estimator->Id() == calcerPtr->Id(), "Estimator.Id() != Calcer.Id()");
            return calcerPtr;
        }

        bool WasBuilt = false;

        const TFeatureEstimators& FeatureEstimators;
        const TTextDigitizers& TextDigitizers;

        TVector<TMap<ui32, ui32>> PerFeatureDigitizers;
        TVector<TVector<ui32>> PerTokenizedFeatureCalcers;

        THashMap<TDigitizer, ui32> DigitizerToId;
        TVector<TTextFeatureCalcerPtr> Calcers;

        THashMap<TGuid, TDigitizerId> CalcerToDigitizer;
        TVector<TEstimatedFeature> EstimatedFeatures;

        NPar::TLocalExecutor* LocalExecutor;
    };
}


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
            NPar::ParallelFor(
                localExecutor,
                0,
                (ui32)sampleCount,
                [&](int sample) {
                    (*learnTargetClasses)[ctrIdx][sample]
                        = targetClassifiers[ctrIdx].GetTargetClass(targets[sample]);
                }
            );

            (*targetClassesCount)[ctrIdx] = targetClassifiers[ctrIdx].GetClassesCount();
        }
    }

    static bool NeedTargetClasses(const TFullModel& coreModel) {
        return AnyOf(
            coreModel.ModelTrees->GetUsedModelCtrs(),
            [](const TModelCtr& modelCtr) {
                return NeedTargetClassifier(modelCtr.Base.CtrType);
            }
        );
    }


    namespace {
        class TIncompleteData {
        public:
            TIncompleteData(
                TTrainingDataProviders&& trainingData,
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

                // since counters are not implemented for mult-dimensional target
                if (outDatasetDataForFinalCtrs->Data.Learn->TargetData->GetTargetDimension() == 1) {
                    outDatasetDataForFinalCtrs->Targets = *outDatasetDataForFinalCtrs->Data.Learn->TargetData->GetOneDimensionalTarget();
                }

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
            TTrainingDataProviders TrainingData;

            const TVector<TTargetClassifier>& TargetClassifiers;
            ui32 NumThreads;

            TVector<TVector<int>> LearnTargetClasses;
            TVector<int> TargetClassesCount;

            THashMap<TFeatureCombination, TProjection> FeatureCombinationToProjection;
        };
    }

    TCoreModelToFullModelConverter::TCoreModelToFullModelConverter(
        const NCatboostOptions::TCatBoostOptions& options,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TClassificationTargetHelper& classificationTargetHelper,
        ui64 ctrLeafCountLimit,
        bool storeAllSimpleCtrs,
        EFinalCtrComputationMode finalCtrComputationMode,
        EFinalFeatureCalcersComputationMode finalFeatureCalcerComputationMode
    )
        : NumThreads(options.SystemOptions->NumThreads)
        , FinalCtrComputationMode(finalCtrComputationMode)
        , FinalFeatureCalcerComputationMode(finalFeatureCalcerComputationMode)
        , CpuRamLimit(ParseMemorySizeDescription(options.SystemOptions->CpuUsedRamLimit.Get()))
        , CtrLeafCountLimit(ctrLeafCountLimit)
        , StoreAllSimpleCtrs(storeAllSimpleCtrs)
        , Options(options)
        , outputOptions(outputOptions)
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
        TTrainingDataProviders&& trainingData,
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

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithFeatureEstimators(
        TFeatureEstimatorsPtr featureEstimators
    ) {
        FeatureEstimators = featureEstimators;
        return *this;
    }

    void TCoreModelToFullModelConverter::Do(
        bool requiresStaticCtrProvider,
        TFullModel* dstModel,
        NPar::TLocalExecutor* localExecutor) {

        DoImpl(requiresStaticCtrProvider, dstModel, localExecutor);
    }

    void TCoreModelToFullModelConverter::Do(
        const TString& fullModelPath,
        const TVector<EModelType>& formats,
        bool addFileFormatExtension,
        NPar::TLocalExecutor* localExecutor
    ) {
        TFullModel fullModel;

        DoImpl(
            AnyOf(
                formats,
                [](EModelType format) {
                    return format == EModelType::Python ||
                        format == EModelType::Cpp ||
                        format == EModelType::Json;
                }
            ),
            &fullModel,
            localExecutor
        );

        ExportFullModel(fullModel, fullModelPath, LearnObjectsData.Get(), formats, addFileFormatExtension);
    }

    void TCoreModelToFullModelConverter::DoImpl(
        bool requiresStaticCtrProvider,
        TFullModel* dstModel,
        NPar::TLocalExecutor* localExecutor
    ) {
        CB_ENSURE_INTERNAL(CoreModel, "CoreModel has not been specified");

        if (CoreModel != dstModel) {
            *dstModel = std::move(*CoreModel);
        }
        dstModel->ModelInfo["model_guid"] = CreateGuidAsString();
        dstModel->ModelInfo["train_finish_time"] = TInstant::Now().ToStringUpToSeconds();
        dstModel->ModelInfo["catboost_version_info"] = GetProgramSvnVersion();

        {
            NJson::TJsonValue jsonOptions(NJson::EJsonValueType::JSON_MAP);
            Options.Save(&jsonOptions);
            dstModel->ModelInfo["params"] = WriteTJsonValue(jsonOptions);
            NJson::TJsonValue jsonOutputOptions(NJson::EJsonValueType::JSON_MAP);
            outputOptions.Save(&jsonOutputOptions);
            dstModel->ModelInfo["output_options"] = WriteTJsonValue(jsonOutputOptions);
            for (const auto& keyValue : Options.Metadata.Get().GetMap()) {
                dstModel->ModelInfo[keyValue.first] = keyValue.second.GetString();
            }
        }

        ELossFunction lossFunction = Options.LossFunctionDescription.Get().GetLossFunction();
        if (IsClassificationObjective(lossFunction)) {
            dstModel->ModelInfo["class_params"] = ClassificationTargetHelper.Serialize();
        }

        if (
            FinalFeatureCalcerComputationMode == EFinalFeatureCalcersComputationMode::Default &&
            !dstModel->ModelTrees->GetEstimatedFeatures().empty()
        ) {
            CB_ENSURE_INTERNAL(
                FeatureEstimators,
                "FinalFeatureCalcerComputation: FeatureEstimators shouldn't be empty"
            );

            const TTextDigitizers& textDigitizers =
                LearnObjectsData->GetQuantizedFeaturesInfo()->GetTextDigitizers();

            TTextProcessingCollection textProcessingCollection;
            TVector<TEstimatedFeature> remappedEstimatedFeatures;
            CreateTextProcessingCollection(
                *FeatureEstimators,
                textDigitizers,
                TVector<TEstimatedFeature>(
                    dstModel->ModelTrees->GetEstimatedFeatures().begin(),
                    dstModel->ModelTrees->GetEstimatedFeatures().end()
                ),
                &textProcessingCollection,
                &remappedEstimatedFeatures,
                localExecutor
            );

            dstModel->TextProcessingCollection = MakeIntrusive<TTextProcessingCollection>(textProcessingCollection);
            dstModel->UpdateEstimatedFeaturesIndices(std::move(remappedEstimatedFeatures));
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
                dstModel->ModelTrees->GetUsedModelCtrBases(),
                [&dstModel, &lock](TCtrValueTable&& table) {
                    with_lock(lock) {
                        dstModel->CtrProvider->AddCtrCalcerData(std::move(table));
                    }
                }
            );

            dstModel->UpdateDynamicData();
        } else {
            dstModel->CtrProvider = new TStaticCtrOnFlightSerializationProvider(
                dstModel->ModelTrees->GetUsedModelCtrBases(),
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
            featureCombinationToProjectionMap,
            datasetDataForFinalCtrs,
            *PerfectHashedToHashedCatValuesMap,
            CtrLeafCountLimit,
            StoreAllSimpleCtrs,
            Options.CatFeatureParams.Get().CounterCalcMethod,
            ctrBases,
            std::move(asyncCtrValueTableCallback),
            &localExecutor
        );
    }

    void CreateTextProcessingCollection(
        const TFeatureEstimators& featureEstimators,
        const TTextDigitizers& textDigitizers,
        const TVector<TEstimatedFeature>& estimatedFeatures,
        TTextProcessingCollection* textProcessingCollection,
        TVector<TEstimatedFeature>* reorderedEstimatedFeatures,
        NPar::TLocalExecutor* localExecutor
    ) {
        CB_ENSURE(
            !estimatedFeatures.empty(),
            "CreateTextProcessingCollection: Estimated feature shouldn't be empty"
        );

        TTextCollectionBuilder textCollectionBuilder(
            featureEstimators,
            textDigitizers,
            localExecutor
        );

        TVector<TEstimatedFeature> usedEstimatedFeatures;
        TGuid lastCalcerId = estimatedFeatures[0].CalcerId;

        for (const auto& estimatedFeature: estimatedFeatures) {
            const TGuid currentCalcerIndex = estimatedFeature.CalcerId;
            if (lastCalcerId != currentCalcerIndex) {
                textCollectionBuilder.AddFeatureEstimator(lastCalcerId, usedEstimatedFeatures);
                lastCalcerId = currentCalcerIndex;
                usedEstimatedFeatures.clear();
            }
            usedEstimatedFeatures.push_back(estimatedFeature);
        }
        textCollectionBuilder.AddFeatureEstimator(lastCalcerId, usedEstimatedFeatures);
        textCollectionBuilder.Build(textProcessingCollection, reorderedEstimatedFeatures);
    };

    void ExportFullModel(
        const TFullModel& fullModel,
        const TString& fullModelPath,
        const TMaybe<const TObjectsDataProvider*> allLearnObjectsData,
        TConstArrayRef<EModelType> formats,
        bool addFileFormatExtension
    ) {
        const TConstArrayRef<TFloatFeature> floatFeatures = fullModel.ModelTrees->GetFloatFeatures();
        const TConstArrayRef<TCatFeature> catFeatures = fullModel.ModelTrees->GetCatFeatures();
        TFeaturesLayout featuresLayout(
            TVector<TFloatFeature>(floatFeatures.begin(), floatFeatures.end()),
            TVector<TCatFeature>(catFeatures.begin(), catFeatures.end())
        );
        TVector<TString> featureIds = featuresLayout.GetExternalFeatureIds();

        THashMap<ui32, TString> catFeaturesHashToString;
        if (fullModel.GetUsedCatFeaturesCount()) {
            const bool anyExportFormatRequiresCatFeaturesHashToString = AnyOf(
                formats,
                [] (EModelType format) {
                    return format == EModelType::Python ||
                        format == EModelType::Cpp ||
                        format == EModelType::Json;
                }
            );
            if (anyExportFormatRequiresCatFeaturesHashToString) {
                CB_ENSURE(
                    allLearnObjectsData,
                    "Some of the specified model export formats require all categorical features values data"
                    " which is not available (probably due to the training continuation on a different dataset)"
                );
                catFeaturesHashToString = MergeCatFeaturesHashToString(**allLearnObjectsData);
            }
        }

        for (const auto& format: formats) {
            NCB::ExportModel(
                fullModel,
                fullModelPath,
                format,
                "",
                addFileFormatExtension,
                &featureIds,
                &catFeaturesHashToString
            );
        }
    }
}
