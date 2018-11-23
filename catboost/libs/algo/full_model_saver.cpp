#include "full_model_saver.h"

#include "online_ctr.h"
#include "quantization.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data_new/features_layout.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/model.h>

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
        const TVector<float>& targets,
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

    static void GetFeatureCombinationsToProjection(
        const TFullModel& coreModel,
        THashMap<TFeatureCombination, TProjection>* featureCombinationToProjection
    ) {
        featureCombinationToProjection->clear();

        THashMap<TFloatSplit, int> floatSplitToBinIndex;
        for (const auto& floatFeature: coreModel.ObliviousTrees.FloatFeatures) {
            for (int binIndex : xrange(floatFeature.Borders.size())) {
                floatSplitToBinIndex.emplace(
                    TFloatSplit(floatFeature.FeatureIndex, floatFeature.Borders[binIndex]),
                    binIndex
                );
            }
        }

        THashMap<TOneHotSplit, int> oneHotSplitToBinIndex;
        for (const auto& oneHotFeature: coreModel.ObliviousTrees.OneHotFeatures) {
            for (int binIndex : xrange(oneHotFeature.Values.size())) {
                oneHotSplitToBinIndex.emplace(
                    TOneHotSplit(oneHotFeature.CatFeatureIndex, oneHotFeature.Values[binIndex]),
                    binIndex
                );
            }
        }

        for (const auto& ctrFeature : coreModel.ObliviousTrees.CtrFeatures) {
            const TFeatureCombination& featureCombination = ctrFeature.Ctr.Base.Projection;

            TProjection projection;
            projection.CatFeatures = featureCombination.CatFeatures;

            projection.BinFeatures.reserve(featureCombination.BinFeatures.size());
            for (const auto& floatSplit : featureCombination.BinFeatures) {
                projection.BinFeatures.emplace_back(
                    floatSplit.FloatFeature,
                    floatSplitToBinIndex[floatSplit]
                );
            }

            projection.OneHotFeatures.reserve(featureCombination.OneHotFeatures.size());
            for (const auto& oneHotSplit : featureCombination.OneHotFeatures) {
                projection.OneHotFeatures.emplace_back(
                    oneHotSplit.CatFeatureIdx,
                    oneHotSplitToBinIndex[oneHotSplit]
                );
            }

            featureCombinationToProjection->emplace(featureCombination, std::move(projection));
        }
    }


    namespace {
        class TQuantizedPools {
        public:
            TQuantizedPools(
                const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                const TVector<TString>& classNames,
                const TVector<TTargetClassifier>& targetClassifiers,
                ECounterCalc counterCalcMethod,
                size_t oneHotMaxSize,
                ui32 numThreads
            )
                : TargetClassifiers(targetClassifiers)
                , OneHotMaxSize(oneHotMaxSize)
                , NumThreads(numThreads)
            {
                OwnedPools.Reset(new TTrainPools);
                NCB::TTargetConverter targetConverter = NCB::MakeTargetConverter(classNames);
                NCB::ReadTrainPools(
                    poolLoadOptions,
                    counterCalcMethod != ECounterCalc::SkipTest,
                    NumThreads,
                    &targetConverter,
                    Nothing(),
                    OwnedPools.Get()
                );
                Pools = TClearablePoolPtrs(*OwnedPools, true);
            }

            TQuantizedPools(
                const TClearablePoolPtrs& pools,
                const TVector<TTargetClassifier>& targetClassifiers,
                ECounterCalc counterCalcMethod,
                size_t oneHotMaxSize,
                ui32 numThreads
            )
                : Pools(pools)
                , TargetClassifiers(targetClassifiers)
                , OneHotMaxSize(oneHotMaxSize)
                , NumThreads(numThreads)
            {
                if (counterCalcMethod == ECounterCalc::SkipTest) {
                    Pools.Test.clear();
                }
            }

            void operator()(
                const TFullModel& coreModel,
                TDatasetDataForFinalCtrs* outDatasetDataForFinalCtrs,
                const THashMap<TFeatureCombination, TProjection>** outFeatureCombinationToProjection
            ) {
                NPar::TLocalExecutor localExecutor;
                localExecutor.RunAdditionalThreads(NumThreads - 1);

                if (NeedTargetClasses(coreModel)) {
                    CreateTargetClasses(
                        localExecutor,
                        Pools.Learn->Docs.Target,
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

                THashSet<int> usedFeatures;

                /*
                 *  need to have vector for all float features in pool,
                 *   but only used in model will be binarized
                 */
                TVector<TFloatFeature> floatFeatures(
                    Pools.Learn->Docs.GetEffectiveFactorCount() - Pools.Learn->CatFeatures.size()
                );

                for (const auto& modelFloatFeature : coreModel.ObliviousTrees.FloatFeatures) {
                    floatFeatures.at(modelFloatFeature.FeatureIndex) = modelFloatFeature;
                    usedFeatures.insert(modelFloatFeature.FlatFeatureIndex);
                }
                for (const auto& modelCatFeature : coreModel.ObliviousTrees.CatFeatures) {
                    usedFeatures.insert(modelCatFeature.FlatFeatureIndex);
                }

                TVector<int> ignoredFeatures;

                for (auto flatFeatureIndex : xrange(Pools.Learn->Docs.GetEffectiveFactorCount())) {
                    if (!usedFeatures.has(flatFeatureIndex)) {
                        ignoredFeatures.push_back(flatFeatureIndex);
                    }
                }

                LearnDataset = BuildDataset(*Pools.Learn);
                TestDatasets.clear();
                for (const TPool* testPoolPtr : Pools.Test) {
                    TestDatasets.push_back(BuildDataset(*testPoolPtr));
                }

                QuantizeTrainPools(
                    Pools,
                    floatFeatures,
                    &coreModel.ObliviousTrees.OneHotFeatures,
                    ignoredFeatures,
                    OneHotMaxSize,
                    localExecutor,
                    &LearnDataset,
                    &TestDatasets
                );
                TestDataPtrs = GetConstPointers(TestDatasets);

                outDatasetDataForFinalCtrs->LearnData = &LearnDataset;
                outDatasetDataForFinalCtrs->TestDataPtrs = &TestDataPtrs;
                outDatasetDataForFinalCtrs->LearnPermutation = Nothing();
                outDatasetDataForFinalCtrs->Targets = &Pools.Learn->Docs.Target;

                GetFeatureCombinationsToProjection(coreModel, &FeatureCombinationToProjection);
                *outFeatureCombinationToProjection = &FeatureCombinationToProjection;
            }

        private:
            /* used only if read,
               sharedPtr because of possible copying to GetBinarizedDataFunc after construction
            */
            TAtomicSharedPtr<TTrainPools> OwnedPools;

            // used if inited from external data
            TClearablePoolPtrs Pools;

            const TVector<TTargetClassifier>& TargetClassifiers;
            size_t OneHotMaxSize;
            ui32 NumThreads;

            TDataset LearnDataset;
            TVector<TDataset> TestDatasets;
            TDatasetPtrs TestDataPtrs; // for interface compatibility

            TVector<TVector<int>> LearnTargetClasses;
            TVector<int> TargetClassesCount;

            THashMap<TFeatureCombination, TProjection> FeatureCombinationToProjection;
        };
    }

    TCoreModelToFullModelConverter::TCoreModelToFullModelConverter(
        ui32 numThreads,
        EFinalCtrComputationMode finalCtrComputationMode,
        ui64 cpuRamLimit,
        ui64 ctrLeafCountLimit,
        bool storeAllSimpleCtrs,
        const NCatboostOptions::TCatFeatureParams& catFeatureParams
    )
        : NumThreads(numThreads)
        , FinalCtrComputationMode(finalCtrComputationMode)
        , CpuRamLimit(cpuRamLimit)
        , CtrLeafCountLimit(ctrLeafCountLimit)
        , StoreAllSimpleCtrs(storeAllSimpleCtrs)
        , CatFeatureParams(catFeatureParams)
    {}

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithCoreModelFrom(
         TFullModel* coreModel
    ) {
        GetCoreModelFunc = [coreModel]() -> TFullModel& { return *coreModel; };
        return *this;
    }

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithCoreModelFrom(
        const TString& coreModelPath
    ) {
        GetCoreModelFunc = [coreModelPath, coreModel = TFullModel()]() mutable -> TFullModel& {
            TIFStream modelInput(coreModelPath);
            coreModel.Load(&modelInput);
            return coreModel;
        };
        return *this;
    }

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithBinarizedDataComputedFrom(
         const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
         const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjection
    ) {
        if (FinalCtrComputationMode != EFinalCtrComputationMode::Skip) {
            GetBinarizedDataFunc = [
                &datasetDataForFinalCtrs = datasetDataForFinalCtrs,
                &featureCombinationToProjection = featureCombinationToProjection
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
        const TClearablePoolPtrs& pools,
        const TVector<TTargetClassifier>& targetClassifiers
    ) {
        if (FinalCtrComputationMode != EFinalCtrComputationMode::Skip) {
            GetBinarizedDataFunc = TQuantizedPools(
                pools,
                targetClassifiers,
                CatFeatureParams.CounterCalcMethod,
                CatFeatureParams.OneHotMaxSize,
                NumThreads
            );
        }
        return *this;
    }

    TCoreModelToFullModelConverter& TCoreModelToFullModelConverter::WithBinarizedDataComputedFrom(
        const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
        const TVector<TString>& classNames,
        const TVector<TTargetClassifier>& targetClassifiers
    ){
        if (FinalCtrComputationMode != EFinalCtrComputationMode::Skip) {
            GetBinarizedDataFunc = TQuantizedPools(
                poolLoadOptions,
                classNames,
                targetClassifiers,
                CatFeatureParams.CounterCalcMethod,
                CatFeatureParams.OneHotMaxSize,
                NumThreads
            );
        }
        return *this;
    }

    void TCoreModelToFullModelConverter::Do(TFullModel* dstModel, bool requiresStaticCtrProvider) {
        TFullModel& coreModel = GetCoreModelFunc();
        if (&coreModel != dstModel) {
            *dstModel = std::move(coreModel);
        }
        dstModel->ModelInfo["train_finish_time"] = TInstant::Now().ToStringUpToSeconds();
        dstModel->ModelInfo["catboost_version_info"] = GetProgramSvnVersion();
        if (FinalCtrComputationMode == EFinalCtrComputationMode::Skip) {
            return;
        }
        if (dstModel->HasValidCtrProvider()) {
            // ModelBase apparently has valid ctrs table
            // TODO(kirillovs): add here smart check for ctrprovider serialization ability
            // after implementing non-storing ctr providers
            return;
        }
        CB_ENSURE(GetBinarizedDataFunc, "Need dataset data specified for final CTR calculation");

        if (requiresStaticCtrProvider) {
            dstModel->CtrProvider = new TStaticCtrProvider;

            TMutex lock;

            CalcFinalCtrs(
                *dstModel,
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
                coreModel.ObliviousTrees.GetUsedModelCtrBases(),
                [this, &coreModel] (
                    const TVector<TModelCtrBase>& ctrBases,
                    TCtrDataStreamWriter* streamWriter
                ) {
                    CalcFinalCtrs(
                        coreModel,
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

    void TCoreModelToFullModelConverter::Do(
        const TString& fullModelPath,
        const TVector<EModelType>& formats,
        bool addFileFormatExtension,
        const TVector<TString>* featureId,
        const THashMap<int, TString>* catFeaturesHashToString
    ) {
        TFullModel& model = GetCoreModelFunc();
        Do(&model, false);
        for (const auto& format: formats) {
            ExportModel(model, fullModelPath, format, "", addFileFormatExtension, featureId, catFeaturesHashToString);
        }
        model.CtrProvider.Reset();
    }

    void TCoreModelToFullModelConverter::CalcFinalCtrs(
        const TFullModel& coreModel,
        const TVector<TModelCtrBase>& ctrBases,
        std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback
    ) {
        TDatasetDataForFinalCtrs datasetDataForFinalCtrs;
        const THashMap<TFeatureCombination, TProjection>* featureCombinationToProjectionMap;
        GetBinarizedDataFunc(coreModel, &datasetDataForFinalCtrs, &featureCombinationToProjectionMap);

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(NumThreads - 1);

        CalcFinalCtrsAndSaveToModel(
            CpuRamLimit,
            localExecutor,
            *featureCombinationToProjectionMap,
            datasetDataForFinalCtrs,
            CtrLeafCountLimit,
            StoreAllSimpleCtrs,
            CatFeatureParams.CounterCalcMethod,
            NCB::TFeaturesLayout(coreModel.ObliviousTrees.FloatFeatures, coreModel.ObliviousTrees.CatFeatures),
            ctrBases,
            std::move(asyncCtrValueTableCallback)
        );
    };
}
