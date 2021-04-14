#include "ctrs.h"

#include "quantized_features_info.h"

#include <catboost/private/libs/algo/index_hash_calcer.h>
#include <catboost/private/libs/algo_helpers/scratch_cache.h>
#include <catboost/private/libs/options/defaults_helper.h>
#include <catboost/private/libs/options/system_options.h>

#include <catboost/libs/helpers/vector_helpers.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>
#include <util/stream/input.h>
#include <util/system/mktemp.h>


using namespace NCB;

TCtrHelper GetCtrHelper(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    const NCB::TFeaturesLayout& layout,
    TConstArrayRef<float> learnTarget
) throw (yexception) {
    auto lossFunction = catBoostOptions.LossFunctionDescription->GetLossFunction();
    NCatboostOptions::TCatFeatureParams updatedCatFeatureParams = catBoostOptions.CatFeatureParams.Get();

    TMaybeData<TConstArrayRef<TConstArrayRef<float>>> targetsParam;
    if (!learnTarget.empty()) {
        UpdateCtrsTargetBordersOption(
            lossFunction,
            /*approxDimension*/ 1,
            &updatedCatFeatureParams
        );
        targetsParam = TConstArrayRef<TConstArrayRef<float>>(&learnTarget, 1);
    }

    TCtrHelper ctrHelper;
    ctrHelper.InitCtrHelper(
        updatedCatFeatureParams,
        layout,
        targetsParam,
        lossFunction,
        /*objectiveDescriptor*/ Nothing(),
        catBoostOptions.DataProcessingOptions->AllowConstLabel
    );
    return ctrHelper;
}


TTargetStatsForCtrs ComputeTargetStatsForCtrs(
    const TCtrHelper& ctrHelper,
    TConstArrayRef<float> learnTarget,
    NPar::TLocalExecutor* localExecutor
) throw (yexception) {
    TTargetStatsForCtrs targetStatsForCtrs;

    const auto& targetClassifiers = ctrHelper.GetTargetClassifiers();
    int ctrCount = targetClassifiers.ysize();
    AllocateRank2(ctrCount, learnTarget.size(), targetStatsForCtrs.LearnTargetClass);
    targetStatsForCtrs.TargetClassesCount.resize(ctrCount);
    for (int ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
        // Spark supports only 1-dimensional targets for now
        Y_ASSERT(targetClassifiers[ctrIdx].GetTargetId() == 0);
        NPar::ParallelFor(
            *localExecutor,
            0,
            SafeIntegerCast<ui32>(learnTarget.size()),
            [&] (ui32 z) {
                targetStatsForCtrs.LearnTargetClass[ctrIdx][z]
                    = targetClassifiers[ctrIdx].GetTargetClass(learnTarget[z]);
            }
        );
        targetStatsForCtrs.TargetClassesCount[ctrIdx] = targetClassifiers[ctrIdx].GetClassesCount();
    }
    return targetStatsForCtrs;
}


static TTrainingDataProviderPtr MakeTrainingDataProvider(
    const NCB::TQuantizedObjectsDataProviderPtr objectsData
) {
    TDataMetaInfo metaInfo;
    metaInfo.ObjectCount = objectsData->GetObjectCount();
    metaInfo.FeaturesLayout = objectsData->GetFeaturesLayout();
    return MakeIntrusive<TTrainingDataProvider>(
        objectsData->GetFeaturesLayout(),
        std::move(metaInfo),
        objectsData->GetObjectsGrouping(),
        objectsData,
        /*targetData*/ nullptr
    );
}

struct TOnlineCtrPerProjectionPerDatasetData {
    NCB::TOnlineCtrUniqValuesCounts UniqValuesCounts; // [catFeatureIdx]

    // [ctrIdx][targetBorderIdx][priorIdx][datasetIdx]. Stores TVector<ui64> to be compatible with TCompressedArray
    TVector<TArray2D<TVector<TVector<ui64>>>> Data;
};

struct TEstimatedColumnsDataWriter final : public IOnlineCtrProjectionDataWriter {
    TEstimatedColumnsDataWriter(const TVector<size_t>& datasetSizes)
        : DatasetSizes(datasetSizes)
    {
        for (auto size : datasetSizes) {
            DatasetUi64Sizes.push_back(CeilDiv(size, sizeof(ui64)));
        }
    }

    void SetCurrentCatFeatureIdx(ui32 currentCatFeatureIdx) {
        Data.resize(currentCatFeatureIdx + 1);
    }

    void SetUniqValuesCounts(const NCB::TOnlineCtrUniqValuesCounts& uniqValuesCounts) override {
        Data.back().UniqValuesCounts = uniqValuesCounts;
    }

    void AllocateData(size_t ctrCount) override {
        Data.back().Data.resize(ctrCount);
    }

    /* call after AllocateData has been called
      it must be thread-safe to call concurrently for different ctrIdx
     */
    void AllocateCtrData(size_t ctrIdx, size_t targetBorderCount, size_t priorCount) override {
        auto& dataPart = Data.back().Data[ctrIdx];
        dataPart.SetSizes(priorCount, targetBorderCount);
        for (auto targetBorderIdx : xrange(targetBorderCount)) {
            for (auto priorIdx : xrange(priorCount)) {
                auto& datasetsPart = dataPart[targetBorderIdx][priorIdx];
                datasetsPart.resize(DatasetSizes.size());
                for (auto datasetIdx : xrange(datasetsPart.size())) {
                    datasetsPart[datasetIdx].yresize(DatasetUi64Sizes[datasetIdx]);
                }
            }
        }
    }

    TArrayRef<ui8> GetDataBuffer(int ctrIdx, int targetBorderIdx, int priorIdx, int datasetIdx) override {
        return TArrayRef<ui8>(
            (ui8*)Data.back().Data[ctrIdx][targetBorderIdx][priorIdx][datasetIdx].data(),
            DatasetSizes[datasetIdx]
        );
    }

    TVector<TOnlineCtrPerProjectionPerDatasetData> GetData() {
        return std::move(Data);
    }

private:
    TVector<size_t> DatasetSizes;
    TVector<size_t> DatasetUi64Sizes;
    TVector<TOnlineCtrPerProjectionPerDatasetData> Data;
};


static TQuantizedObjectsDataProviderPtr CreateEstimatedObjectsDataProvider(
    ui32 objectCount,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TVector<TVector<ui64>>&& data // [featureIdx]
) {
    TCommonObjectsData commonObjectsData;
    commonObjectsData.FeaturesLayout = quantizedFeaturesInfo->GetFeaturesLayout();
    commonObjectsData.SubsetIndexing
        = MakeAtomicShared<TArraySubsetIndexing<ui32>>(TFullSubset<ui32>(objectCount));

    TQuantizedObjectsData dstData;
    dstData.QuantizedFeaturesInfo = std::move(quantizedFeaturesInfo);
    dstData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex.resize(data.size());
    dstData.ExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart.resize(data.size());
    dstData.FeaturesGroupsData.FlatFeatureIndexToGroupPart.resize(data.size());

    for (auto featureIdx : xrange<ui32>(data.size())) {
        dstData.FloatFeatures.push_back(
            MakeHolder<TQuantizedFloatValuesHolder>(
                featureIdx,
                TCompressedArray(objectCount, /*bitsPerKey*/ 8, std::move(data[featureIdx])),
                commonObjectsData.SubsetIndexing.Get()
            )
        );
    }

    return MakeIntrusive<TQuantizedObjectsDataProvider>(
        MakeIntrusive<TObjectsGrouping>(objectCount),
        std::move(commonObjectsData),
        std::move(dstData),
        /*skipCheck*/ true,
        Nothing()
    );
}


void ComputeEstimatedCtrFeatures(
    const TCtrHelper& ctrHelper,
    const NCatboostOptions::TCatBoostOptions& catBoostOptions, // actually only catFeatureParams is used
    const TTargetStatsForCtrs& targetStats,
    const TQuantizedObjectsDataProviderPtr& learnData,
    const TVector<TQuantizedObjectsDataProviderPtr>& testData,
    NPar::TLocalExecutor* localExecutor,
    TEstimatedForCPUObjectsDataProviders* outputData,
    TPrecomputedOnlineCtrMetaData* outputMeta
) throw (yexception) {
    TTrainingDataProviders trainingDataProviders;
    TVector<size_t> dataSizes;
    trainingDataProviders.Learn = MakeTrainingDataProvider(learnData);
    dataSizes.push_back(learnData->GetObjectCount());
    for (const auto& testDataPart : testData) {
        trainingDataProviders.Test.push_back(MakeTrainingDataProvider(testDataPart));
        dataSizes.push_back(testDataPart->GetObjectCount());
    }

    TFeaturesArraySubsetIndexing learnSubsetIndexing(TFullSubset<ui32>(learnData->GetObjectCount()));

    TEstimatedColumnsDataWriter dataWriter(dataSizes);

    NCB::TScratchCache scratchCache;

    const auto& featuresLayout = *(learnData->GetFeaturesLayout());
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            dataWriter.SetCurrentCatFeatureIdx(*catFeatureIdx);
            TProjection proj;
            proj.AddCatFeature(SafeIntegerCast<int>(*catFeatureIdx));
            ComputeOnlineCTRs(
                trainingDataProviders,
                proj,
                ctrHelper,
                learnSubsetIndexing,
                targetStats.LearnTargetClass,
                targetStats.TargetClassesCount,
                catBoostOptions.CatFeatureParams,
                localExecutor,
                &scratchCache,
                &dataWriter
            );
        }
    );

    TVector<TOnlineCtrPerProjectionPerDatasetData> data = dataWriter.GetData();

    TVector<TVector<TVector<ui64>>> dstData(dataSizes.size()); // [datasetIdx][dstFeatureIdx] -> storage

    size_t dstFeatureIdx = 0;
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            outputMeta->ValuesCounts.emplace(*catFeatureIdx, data[*catFeatureIdx].UniqValuesCounts);
            auto& perFeatureData = data[*catFeatureIdx].Data;
            for (auto ctrIdx : xrange<i16>(perFeatureData.size())) {
                auto& perCtrData = perFeatureData[ctrIdx];
                for (auto targetBorderIdx : xrange<i16>(perCtrData.GetYSize())) {
                    auto perTargetBorderData = perCtrData[targetBorderIdx];
                    for (auto priorIdx : xrange<i16>(perCtrData.GetXSize())) {
                        auto& perPriorData = perTargetBorderData[priorIdx];

                        outputMeta->OnlineCtrIdxToFeatureIdx.emplace(
                            TOnlineCtrIdx{
                                SafeIntegerCast<i32>(*catFeatureIdx),
                                ctrIdx,
                                targetBorderIdx,
                                priorIdx
                            },
                            dstFeatureIdx
                        );

                        for (auto datasetIdx : xrange(dataSizes.size())) {
                            dstData[datasetIdx].push_back(std::move(perPriorData[datasetIdx]));
                        }

                        ++dstFeatureIdx;
                    }
                }
            }
        }
    );

    auto estimatedQuantizedFeaturesInfo = MakeEstimatedQuantizedFeaturesInfo(
        SafeIntegerCast<i32>(dstFeatureIdx)
    );

    outputData->Learn = CreateEstimatedObjectsDataProvider(
        dataSizes[0],
        estimatedQuantizedFeaturesInfo,
        std::move(dstData[0])
    );
    outputData->Test.clear();
    for (auto testIdx : xrange(dataSizes.size() - 1)) {
        outputData->Test.push_back(
            CreateEstimatedObjectsDataProvider(
                dataSizes[testIdx + 1],
                estimatedQuantizedFeaturesInfo,
                std::move(dstData[testIdx + 1])
            )
        );
    }
}


TFinalCtrsCalcer::TFinalCtrsCalcer(
    TFullModel* modelWithoutCtrData, // moved into
    const NCatboostOptions::TCatBoostOptions* catBoostOptions,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    TConstArrayRef<float> learnTarget,
    TTargetStatsForCtrs* targetStatsForCtrs, // moved into
    const TCtrHelper& ctrHelper,
    NPar::TLocalExecutor* localExecutor
) throw(yexception)
    : Model(std::move(*modelWithoutCtrData))
    , FeaturesLayout(quantizedFeaturesInfo.GetFeaturesLayout())
    , TargetStatsForCtrs(std::move(*targetStatsForCtrs))
    , LocalExecutor(localExecutor)
    , CtrDataFile(MakeTempName(nullptr, "ctr_data_file"))
    , CtrDataFileStream(new TOFStream(CtrDataFile.Name()))
    , CatBoostOptions(catBoostOptions)
    , CpuRamLimit(ParseMemorySizeDescription(catBoostOptions->SystemOptions->CpuUsedRamLimit.Get()))
{
    TVector<TModelCtrBase> modelCtrBases = Model.ModelTrees->GetApplyData()->GetUsedModelCtrBases();
    StreamWriter.Reset(new TCtrDataStreamWriter(CtrDataFileStream.Get(), modelCtrBases.size()));
    for (const auto& modelCtrBase : modelCtrBases) {
        auto flatCatFeatureIdx = FeaturesLayout->GetExternalFeatureIdx(
            SafeIntegerCast<ui32>(modelCtrBase.Projection.CatFeatures[0]),
            EFeatureType::Categorical
        );
        CatFeatureFlatIndexToModelCtrsBases[SafeIntegerCast<i32>(flatCatFeatureIdx)].push_back(modelCtrBase);
    }

    DatasetDataForFinalCtrs.Targets = TVector<TConstArrayRef<float>>(1, learnTarget);
    DatasetDataForFinalCtrs.LearnTargetClass = &TargetStatsForCtrs.LearnTargetClass;
    DatasetDataForFinalCtrs.TargetClassesCount = &TargetStatsForCtrs.TargetClassesCount;
    DatasetDataForFinalCtrs.TargetClassifiers = &ctrHelper.GetTargetClassifiers();
}

TVector<i32> TFinalCtrsCalcer::GetCatFeatureFlatIndicesUsedForCtrs() const throw(yexception) {
    TVector<i32> result;
    for (const auto& [key, value] : CatFeatureFlatIndexToModelCtrsBases) {
        result.push_back(key);
    }
    return result;
}

void TFinalCtrsCalcer::ProcessForFeature(
   i32 catFeatureFlatIdx,
   const NCB::TQuantizedObjectsDataProviderPtr& learnData,
   const TVector<NCB::TQuantizedObjectsDataProviderPtr>& testData
) throw(yexception) {
    DatasetDataForFinalCtrs.Data.Learn = MakeTrainingDataProvider(learnData);
    for (const auto& testDataPart : testData) {
        DatasetDataForFinalCtrs.Data.Test.push_back(MakeTrainingDataProvider(testDataPart));
    }

    ui32 catFeatureIdx = FeaturesLayout->GetInternalFeatureIdx(catFeatureFlatIdx);

    TFeatureCombination featureCombination;
    featureCombination.CatFeatures.push_back(catFeatureIdx);
    TProjection projection;
    projection.CatFeatures.push_back(catFeatureIdx);

    THashMap<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
    featureCombinationToProjectionMap.emplace(featureCombination, projection);

    CalcFinalCtrsAndSaveToModel(
        CpuRamLimit,
        featureCombinationToProjectionMap,
        DatasetDataForFinalCtrs,
        PerfectHashedToHashedCatValuesMap,
        CatBoostOptions->CatFeatureParams->CtrLeafCountLimit,
        CatBoostOptions->CatFeatureParams->StoreAllSimpleCtrs,
        CatBoostOptions->CatFeatureParams->CounterCalcMethod,
        CatFeatureFlatIndexToModelCtrsBases.at(catFeatureFlatIdx),
        [&](TCtrValueTable&& table) {
            // there's lock inside, so it is thread-safe
            StreamWriter->SaveOneCtr(table);
        },
        LocalExecutor
    );

    DatasetDataForFinalCtrs.Data.Learn = nullptr;
    DatasetDataForFinalCtrs.Data.Test.clear();
}

class TFromFileCtrProvider: public ICtrProvider {
public:
    TFromFileCtrProvider(const TString& fileName)
        : FileName(fileName)
    {}

    bool HasNeededCtrs(const TConstArrayRef<TModelCtr>) const override {
        return false;
    }

    void CalcCtrs(
        const TConstArrayRef<TModelCtr>,
        const TConstArrayRef<ui8>,
        const TConstArrayRef<ui32>,
        size_t,
        TArrayRef<float>
    ) override {
        ythrow TCatBoostException()
            << "TFromFileCtrProvider is for streamed serialization only";
    }

    void SetupBinFeatureIndexes(
        const TConstArrayRef<TFloatFeature>,
        const TConstArrayRef<TOneHotFeature>,
        const TConstArrayRef<TCatFeature>
    ) override {
        ythrow TCatBoostException()
            << "TFromFileCtrProvider is for streamed serialization only";
    }
    bool IsSerializable() const override {
        return true;
    }
    void AddCtrCalcerData(TCtrValueTable&& ) override {
        ythrow TCatBoostException()
            << "TFromFileCtrProvider is for streamed serialization only";
    }

    void DropUnusedTables(TConstArrayRef<TModelCtrBase>) override {
        ythrow TCatBoostException()
            << "TFromFileCtrProvider is for streamed serialization only";
    }

    void Save(IOutputStream* out) const override {
        TIFStream in(FileName);
        TransferData(&in, out);
    }

    void Load(IInputStream*) override {
        ythrow TCatBoostException()
            << "TFromFileCtrProvider is for streamed serialization only";
    }

    TString ModelPartIdentifier() const override {
        return "static_provider_v1";
    }

private:
    TString FileName;
};


TFullModel TFinalCtrsCalcer::GetModelWithCtrData() throw(yexception) {
    StreamWriter.Destroy();
    CtrDataFileStream.Destroy();
    Model.CtrProvider = new TFromFileCtrProvider(CtrDataFile.Name());

    TTempFile tmpModelFile(MakeTempName(nullptr, "catboost_model"));
    {
        TOFStream out(tmpModelFile.Name());
        Model.Save(&out);
    }
    {
        TIFStream in(tmpModelFile.Name());
        Model.Load(&in);
    }
    return std::move(Model);
}
