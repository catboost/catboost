#include "tree_ctrs.h"
#include "batch_feature_tensor_builder.h"
#include "ctr_from_tensor_calcer.h"
#include <catboost/cuda/gpu_data/feature_layout_single.h>
#include <catboost/cuda/gpu_data/compressed_index_builder.h>

THolder<NCatboostCuda::TTreeCtrDataSetBuilder::TCompressedIndex> NCatboostCuda::TTreeCtrDataSetBuilder::CreateCompressedIndex(NCudaLib::TSingleMapping docsMapping) {
    THolder<TCompressedIndex> dataSet = MakeHolder<TCompressedIndex>();
    const TVector<TCtr>& ctrs = TreeCtrDataSet.GetCtrs();

    const ui32 featureCount = static_cast<ui32>(ctrs.size());
    TVector<ui32> featureIds;
    featureIds.resize(featureCount);
    std::iota(featureIds.begin(), featureIds.end(), 0);

    TBinarizationInfoProvider binarizationInfoProvider(ctrs,
                                                       TreeCtrDataSet.FeaturesManager);
    TDataSetDescription description = {};
    description.Name = "TreeCtrs compressed dataset";

    using TBuilder = TSharedCompressedIndexBuilder<TSingleDevLayout>;
    ui32 id = TBuilder::AddDataSetToCompressedIndex(binarizationInfoProvider,
                                                    description,
                                                    docsMapping,
                                                    featureIds,
                                                    dataSet.Get());

    CB_ENSURE(id == 0);

    return dataSet;
}

NCatboostCuda::TTreeCtrDataSetBuilder::TConstVec NCatboostCuda::TTreeCtrDataSetBuilder::GetBorders(const NCatboostCuda::TCtr& ctr,
                                                                                                   const NCatboostCuda::TTreeCtrDataSetBuilder::TVec& floatCtr,
                                                                                                   ui32 stream) {
    CB_ENSURE(TreeCtrDataSet.InverseCtrIndex.contains(ctr));
    const ui32 featureId = TreeCtrDataSet.InverseCtrIndex[ctr];
    const auto& bordersSlice = TreeCtrDataSet.CtrBorderSlices[featureId];

    if (TreeCtrDataSet.AreCtrBordersComputed[featureId] == false) {
        const auto& binarizationDescription = TreeCtrDataSet.FeaturesManager.GetCtrBinarization(ctr);
        TCudaBuffer<float, NCudaLib::TSingleMapping> bordersVecSlice = TreeCtrDataSet.CtrBorders.SliceView(bordersSlice);
        ComputeCtrBorders(floatCtr,
                          binarizationDescription,
                          stream,
                          bordersVecSlice);
        TreeCtrDataSet.AreCtrBordersComputed[featureId] = true;
    }
    return TreeCtrDataSet.CtrBorders.SliceView(bordersSlice).AsConstBuf();
}

void NCatboostCuda::TTreeCtrDataSetBuilder::ComputeCtrBorders(const NCatboostCuda::TTreeCtrDataSetBuilder::TVec& ctr,
                                                              const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                                                              ui32 stream,
                                                              NCatboostCuda::TTreeCtrDataSetBuilder::TVec& dst) {
    auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Build ctr borders");
    CB_ENSURE(dst.GetMapping().GetObjectsSlice().Size() == binarizationDescription.BorderCount + 1);
    ComputeBordersOnDevice(ctr,
                           binarizationDescription,
                           dst,
                           stream);
}

void NCatboostCuda::TTreeCtrDataSetBuilder::operator()(const NCatboostCuda::TCtr& ctr,
                                                       const NCatboostCuda::TTreeCtrDataSetBuilder::TVec& floatCtr,
                                                       ui32 stream) {
    const ui32 featureId = TreeCtrDataSet.InverseCtrIndex[ctr];
    const auto borders = GetBorders(ctr, floatCtr, stream);
    auto profileGuard = NCudaLib::GetCudaManager().GetProfiler().Profile("binarizeOnDevice");
    BinarizeOnDevice(floatCtr,
                     borders,
                     TreeCtrDataSet.CompressedIndex->GetDataSet(0u).GetTCFeature(featureId),
                     TreeCtrDataSet.CompressedIndex->FlatStorage,
                     StreamParallelCtrVisits /* atomic update for 2 stream */,
                     IsIdentityPermutation ? nullptr : &GatherIndices,
                     stream);
}

NCatboostCuda::TTreeCtrDataSetBuilder::TTreeCtrDataSetBuilder(const TCudaBuffer<const ui32, NCudaLib::TSingleMapping>& indices,
                                                              NCatboostCuda::TTreeCtrDataSet& ctrDataSet,
                                                              bool streamParallelCtrVisits,
                                                              bool isIdentityPermutation)
    : TreeCtrDataSet(ctrDataSet)
    , GatherIndices(indices.ConstCopyView())
    , StreamParallelCtrVisits(streamParallelCtrVisits)
    , IsIdentityPermutation(isIdentityPermutation)
{
    CB_ENSURE(!TreeCtrDataSet.HasCompressedIndex(), "Error: Compressed dataset index already exists");
    using TBuilder = TSharedCompressedIndexBuilder<TSingleDevLayout>;

    if (TreeCtrDataSet.CompressedIndex == nullptr) {
        TreeCtrDataSet.CompressedIndex = CreateCompressedIndex(indices.GetMapping());
    }
    TBuilder::ResetStorage(TreeCtrDataSet.CompressedIndex.Get());
}

NCatboostCuda::TTreeCtrDataSetsHelper::TTreeCtrDataSetsHelper(const NCatboostCuda::TFeatureParallelDataSet& dataSet,
                                                              const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
                                                              ui32 maxDepth, ui32 foldCount,
                                                              NCatboostCuda::TFeatureTensorTracker& emptyTracker)
    : DataSet(dataSet)
    , FeaturesManager(featuresManager)
    , EmptyTracker(emptyTracker.Copy())
    , PureTreeCtrTensorTracker(emptyTracker.Copy())
    , MaxDepth(maxDepth)
    , FoldCount(foldCount)
{
    DepthPermutations.resize(1);
    if (LevelBasedCompressedIndex) {
        DepthPermutations[0] = dataSet.GetIndices().ConstCopyView();
    } else {
        auto tmp = TMirrorBuffer<ui32>::CopyMapping(dataSet.GetIndices());
        MakeSequence(tmp, 0u);
        DepthPermutations[0] = tmp.ConstCopyView();
    }
    const auto& manager = NCudaLib::GetCudaManager();
    const ui32 devCount = manager.GetDeviceCount();
    DataSets.resize(devCount);
    PureTreeCtrDataSets.resize(devCount);
    PackSizeEstimators.resize(devCount);
    NCudaLib::GetCudaManager().WaitComplete();
    for (ui32 dev = 0; dev < devCount; ++dev) {
        auto freeMemory = manager.FreeMemoryMb(dev, false);
        PackSizeEstimators[dev] = (MakeHolder<TTreeCtrDataSetMemoryUsageEstimator>(featuresManager,
                                                                           freeMemory,
                                                                           dataSet.GetCatFeatures().GetFeatureCount(dev),
                                                                           FoldCount,
                                                                           MaxDepth,
                                                                           static_cast<const ui32>(dataSet.GetDataProvider().GetObjectCount()),
                                                                           dataSet.GetCatFeatures().GetStorageType() == EGpuCatFeaturesStorage::GpuRam
                                                                               ? NCudaLib::EPtrType::CudaDevice
                                                                               : NCudaLib::EPtrType::CudaHost));
    }
}

void NCatboostCuda::TTreeCtrDataSetsHelper::AddSplit(const NCatboostCuda::TBinarySplit& split,
                                                     const TMirrorBuffer<ui32>& docBins) {
    if (DataSet.GetCatFeatures().GetFeatureCount() == 0) {
        return;
    }
    auto profileGuard = NCudaLib::GetCudaManager().GetProfiler().Profile("addSplitToTreeCtrsHelper");
    ++CurrentDepth;

    if (FeaturesManager.IsCtr(split.FeatureId)) {
        TFeatureTensor newTensor = CurrentTensor;
        newTensor.AddTensor(FeaturesManager.GetCtr(split.FeatureId).FeatureTensor);
        if (newTensor == CurrentTensor || (!FeaturesManager.UseAsBaseTensorForTreeCtr(newTensor))) {
            return;
        }
        CurrentTensor = newTensor;
        AddNewDataSets(newTensor);
    } else {
        UpdatePureTreeCtrTensor(split);
    }

    //need more memory (sizeof(ui32) * docCount * MaxDepth addition memory),
    // slower cindex building,
    // need to gather docIndices,
    // much faster histograms calc
    if (LevelBasedCompressedIndex) {
        AssignDepthForDataSetsWithoutCompressedIndex(CurrentDepth);
        UpdateUsedPermutations();
        ClearUnusedPermutations();
        if (UsedPermutations.contains(CurrentDepth)) {
            CachePermutation(docBins, CurrentDepth);
        }
        //if we don't have enough memory, we don't need to cache first-level permutations
    } else {
        AssignDepthForDataSetsWithoutCompressedIndex(0);
        UsedPermutations = {0};
    }
    SortDataSetsByCompressedIndexLevelAndSize();
}

void NCatboostCuda::TTreeCtrDataSetsHelper::AddDataSets(
    const TVector<NCatboostCuda::TTreeCtrDataSetsHelper::TTreeCtrDataSetPtr>& dataSets, ui32 permutationId,
    bool withCompressedIndexFlag, TVector<NCatboostCuda::TTreeCtrDataSet*>& dst) {
    for (ui32 i = 0; i < dataSets.size(); ++i) {
        if (dataSets[i]->GetCompressedIndexPermutationKey() == permutationId) {
            if (dataSets[i]->HasCompressedIndex() == withCompressedIndexFlag) {
                dst.push_back(dataSets[i].Get());
            }
        }
    }
}

void NCatboostCuda::TTreeCtrDataSetsHelper::AddNewDataSets(const NCatboostCuda::TFeatureTensor& tensor) {
    auto& manager = NCudaLib::GetCudaManager();
    const ui32 devCount = manager.GetDeviceCount();
    TensorTrackers[tensor] = CreateTensorTracker(tensor);

    for (ui32 dev = 0; dev < devCount; ++dev) {
        AddDataSetPacks(tensor,
                        TensorTrackers[tensor].GetIndices().DeviceView(dev).AsConstBuf(),
                        dev,
                        DataSets[dev]);
    }
}

void NCatboostCuda::TTreeCtrDataSetsHelper::UpdatePureTreeCtrTensor(const NCatboostCuda::TBinarySplit& split) {
    PureTreeCtrTensorTracker.AddBinarySplit(split);
    PureTreeCtrDataSets.clear();

    auto& manager = NCudaLib::GetCudaManager();
    const ui32 devCount = manager.GetDeviceCount();
    PureTreeCtrDataSets.resize(devCount);

    for (ui32 dev = 0; dev < devCount; ++dev) {
        AddDataSetPacks(PureTreeCtrTensorTracker.GetCurrentTensor(),
                        PureTreeCtrTensorTracker.GetIndices().DeviceView(dev).AsConstBuf(),
                        dev,
                        PureTreeCtrDataSets[dev]);
    }
}

bool NCatboostCuda::TTreeCtrDataSetsHelper::FreeMemoryForDataSet(const NCatboostCuda::TTreeCtrDataSet& dataSet,
                                                                 TVector<NCatboostCuda::TTreeCtrDataSetsHelper::TTreeCtrDataSetPtr>& dataSets) {
    const ui32 deviceId = dataSet.GetDeviceId();
    double freeMemory = GetFreeMemory(deviceId);
    double memoryForDataSet = PackSizeEstimators[deviceId]->MemoryForDataSet(dataSet);

    //drop should be in reverse order, so we not trigger defragmentation
    for (i32 dataSetId = (dataSets.size() - 1); dataSetId >= 0; --dataSetId) {
        if (freeMemory >= memoryForDataSet) {
            freeMemory = GetFreeMemory(deviceId);
        }
        if (freeMemory < memoryForDataSet) {
            if (dataSets[dataSetId].Get() != &dataSet && dataSets[dataSetId]->HasCompressedIndex()) {
                freeMemory += PackSizeEstimators[deviceId]->MemoryForDataSet(*dataSets[dataSetId]);
                TTreeCtrDataSetBuilder::DropCache(*dataSets[dataSetId]);
            }
        } else {
            return true;
        }
    }
    return false;
}

void NCatboostCuda::TTreeCtrDataSetsHelper::CachePermutation(const TMirrorBuffer<ui32>& currentBins, ui32 depth) {
    if (DepthPermutations.size() <= depth) {
        DepthPermutations.resize(depth + 1);
    }
    TCudaBuffer<ui32, NCudaLib::TMirrorMapping> permutation;
    permutation.Reset(currentBins.GetMapping());
    MakeSequence(permutation);
    auto tmpBins = TMirrorBuffer<ui32>::CopyMapping(currentBins);
    tmpBins.Copy(currentBins);
    ReorderBins(tmpBins, permutation, 0, depth);
    DepthPermutations[depth] = permutation.ConstCopyView();
    UsedPermutations.insert(depth);
}

void NCatboostCuda::TTreeCtrDataSetsHelper::SortDataSetsByCompressedIndexLevelAndSize() {
    auto comparator = [&](const TTreeCtrDataSetPtr& left, const TTreeCtrDataSetPtr& right) -> bool {
        return (left->GetCompressedIndexPermutationKey() < right->GetCompressedIndexPermutationKey()) ||
               (left->GetCompressedIndexPermutationKey() == right->GetCompressedIndexPermutationKey() &&
                left->Ctrs.size() > right->Ctrs.size());
    };

    for (auto& devDataSets : DataSets) {
        Sort(devDataSets.begin(), devDataSets.end(), comparator);
    }
    for (auto& devDataSets : PureTreeCtrDataSets) {
        Sort(devDataSets.begin(), devDataSets.end(), comparator);
    }
}

void NCatboostCuda::TTreeCtrDataSetsHelper::UpdateForPack(const TVector<TVector<NCatboostCuda::TTreeCtrDataSetsHelper::TTreeCtrDataSetPtr>>& dataSets,
                                                          TSet<ui32>& usedPermutations) {
    for (auto& devDataSets : dataSets) {
        for (auto& ds : devDataSets) {
            usedPermutations.insert(ds->GetCompressedIndexPermutationKey());
        }
    }
}

void NCatboostCuda::TTreeCtrDataSetsHelper::UpdateUsedPermutations() {
    TSet<ui32> usedPermutations;
    UpdateForPack(DataSets, usedPermutations);
    UpdateForPack(PureTreeCtrDataSets, usedPermutations);
    UsedPermutations = usedPermutations;
}

void NCatboostCuda::TTreeCtrDataSetsHelper::ClearUnusedPermutations() {
    for (ui32 i = 0; i < DepthPermutations.size(); ++i) {
        if (UsedPermutations.count(i) == 0) {
            DepthPermutations[i].Clear();
        }
    }
}

bool NCatboostCuda::TTreeCtrDataSetsHelper::AssignForPack(TVector<TVector<NCatboostCuda::TTreeCtrDataSetsHelper::TTreeCtrDataSetPtr>>& dataSets, ui32 depth) {
    bool assigned = false;
    for (auto& devDataSets : dataSets) {
        for (auto& dataSet : devDataSets) {
            if (dataSet->HasCompressedIndex()) {
                continue;
            }
            dataSet->SetPermutationKey(depth);
            assigned = true;
        }
    }
    return assigned;
}

void NCatboostCuda::TTreeCtrDataSetsHelper::AddDataSetPacks(const NCatboostCuda::TFeatureTensor& baseTensor,
                                                            const TSingleBuffer<const ui32>& baseTensorIndices,
                                                            ui32 deviceId,
                                                            TVector<NCatboostCuda::TTreeCtrDataSetsHelper::TTreeCtrDataSetPtr>& dst) {
    const auto& catFeatures = DataSet.GetCatFeatures();
    auto& devFeatures = catFeatures.GetDeviceFeatures(deviceId);
    if (devFeatures.size() == 0) {
        return;
    }
    const ui32 maxPackSize = PackSizeEstimators[deviceId]->GetMaxPackSize();
    CB_ENSURE(maxPackSize, "Error: not enough memory for building ctrs");

    const ui32 currentDstSize = static_cast<const ui32>(dst.size());
    dst.push_back(MakeHolder<TTreeCtrDataSet>(FeaturesManager,
                                              baseTensor,
                                              baseTensorIndices));

    ui32 packSize = 0;
    for (auto feature : devFeatures) {
        auto& nextDataSet = dst.back();
        auto tensor = baseTensor;
        tensor.AddCatFeature(feature);
        if (tensor == baseTensor || !FeaturesManager.UseForTreeCtr(tensor)) {
            continue;
        }
        nextDataSet->AddCatFeature(feature);
        ++packSize;

        if (packSize >= maxPackSize) {
            dst.push_back(MakeHolder<TTreeCtrDataSet>(FeaturesManager,
                                                      baseTensor,
                                                      baseTensorIndices));
            packSize = 0;
        }
    }

    if (dst.back()->CatFeatures.size() == 0) {
        dst.pop_back();
    }
    for (ui32 i = currentDstSize; i < dst.size(); ++i) {
        dst[i]->BuildFeatureIndex();
    }
}

NCatboostCuda::TFeatureTensorTracker NCatboostCuda::TTreeCtrDataSetsHelper::CreateEmptyTrackerForTensor(const NCatboostCuda::TFeatureTensor& tensor) {
    ui64 maxSize = 0;
    TFeatureTensor bestTensor;
    if (PureTreeCtrTensorTracker.GetCurrentTensor().IsSubset(tensor)) {
        maxSize = PureTreeCtrTensorTracker.GetCurrentTensor().Size();
        bestTensor = PureTreeCtrTensorTracker.GetCurrentTensor();
    }

    for (auto& entry : TensorTrackers) {
        auto& tracker = entry.second;
        if (tracker.GetCurrentTensor().IsSubset(tensor) && tracker.GetCurrentTensor().Size() > maxSize) {
            bestTensor = tracker.GetCurrentTensor();
            maxSize = tracker.GetCurrentTensor().Size();
        }
    }

    if (maxSize == 0) {
        return EmptyTracker.Copy();
    }

    if (bestTensor == PureTreeCtrTensorTracker.GetCurrentTensor()) {
        return PureTreeCtrTensorTracker.Copy();
    }

    return TensorTrackers[bestTensor]
        .Copy();
}

void NCatboostCuda::TTreeCtrDataSetsHelper::VisitPermutationDataSets(ui32 permutationId,
                                                                     NCatboostCuda::TTreeCtrDataSetsHelper::TDataSetVisitor& visitor) {
    NCudaLib::RunPerDeviceSubtasks([&](ui32 device) {
        {
            const ui32 catFeatureCount = DataSet.GetCatFeatures().GetDeviceFeatures(device).size();
            if (catFeatureCount == 0) {
                return;
            }
            //this visit order should be best for cache hit
            TVector<TTreeCtrDataSet*> cachedDataSets;
            TVector<TTreeCtrDataSet*> withoutCachedIndexDataSets;

            //cached dataSets doesn't need recalc.
            AddDataSets(DataSets[device], permutationId, true, cachedDataSets);
            AddDataSets(PureTreeCtrDataSets[device], permutationId, true,
                        cachedDataSets);
            //this dataSets need to be rebuild. dataSets withouth cindex should have permutationId >= permutationId of any dataSet with cindex for max performance
            AddDataSets(PureTreeCtrDataSets[device], permutationId, false,
                        withoutCachedIndexDataSets);
            AddDataSets(DataSets[device], permutationId, false,
                        withoutCachedIndexDataSets);

            ProceedDataSets(permutationId, cachedDataSets, visitor);
            ProceedDataSets(permutationId, withoutCachedIndexDataSets, visitor);
        }
    });
}

ui32 NCatboostCuda::TTreeCtrDataSetsHelper::GetMaxUniqueValues() const {
    ui32 maxUniqueValues = 1;
    NCudaLib::RunPerDeviceSubtasks([&](ui32 device) {
        for (const auto& dataSetPtr : DataSets[device]) {
            const auto& ctrs = dataSetPtr->GetCtrs();
            for (const TCtr& ctr : ctrs) {
                if (!FeaturesManager.IsKnown(ctr) || !FeaturesManager.IsUsedCtr(FeaturesManager.GetId(ctr))) {
                    maxUniqueValues = Max(maxUniqueValues, FeaturesManager.GetMaxCtrUniqueValues(ctr));
                }
            }
        }
        for (const auto& dataSetPtr : PureTreeCtrDataSets[device]) {
            const auto& ctrs = dataSetPtr->GetCtrs();
            for (const TCtr& ctr : ctrs) {
                if (!FeaturesManager.IsKnown(ctr) || !FeaturesManager.IsUsedCtr(FeaturesManager.GetId(ctr))) {
                    maxUniqueValues = Max(maxUniqueValues, FeaturesManager.GetMaxCtrUniqueValues(ctr));
                }
            }
        }
    });
    return maxUniqueValues;
}

void NCatboostCuda::TTreeCtrDataSetsHelper::ProceedDataSets(const ui32 dataSetPermutationId,
                                                            const TVector<NCatboostCuda::TTreeCtrDataSet*>& dataSets,
                                                            NCatboostCuda::TTreeCtrDataSetsHelper::TDataSetVisitor& visitor) {
    for (auto dataSetPtr : dataSets) {
        auto& dataSet = *dataSetPtr;
        if (dataSetPtr->GetCompressedIndexPermutationKey() != dataSetPermutationId) {
            continue;
        }
        if (PackSizeEstimators[dataSet.GetDeviceId()]->NotEnoughMemoryForDataSet(dataSet, CurrentDepth)) {
            FreeMemoryForDataSet(dataSet);
        }
        ProceedDataSet(dataSetPermutationId, dataSet, visitor);
    }
}

void NCatboostCuda::TTreeCtrDataSetsHelper::ProceedDataSets(ui32 dataSetPermutationId,
                                                            const TVector<NCatboostCuda::TTreeCtrDataSet*>& dataSets,
                                                            bool withCompressedIndex,
                                                            NCatboostCuda::TTreeCtrDataSetsHelper::TDataSetVisitor& visitor) {
    TVector<ui32> dataSetIds;
    TVector<TTreeCtrDataSet*> rest;

    for (ui32 dataSetId = 0; dataSetId < dataSets.size(); ++dataSetId) {
        if (dataSets[dataSetId]->GetCompressedIndexPermutationKey() == dataSetPermutationId) {
            if (dataSets[dataSetId]->HasCompressedIndex() == withCompressedIndex) {
                dataSetIds.push_back(dataSetId);
            }
        }
    }

    for (auto idx : dataSetIds) {
        const auto& dataSet = *dataSets[idx];
        if (!withCompressedIndex && PackSizeEstimators[dataSet.GetDeviceId()]->NotEnoughMemoryForDataSet(dataSet, CurrentDepth)) {
            FreeMemoryForDataSet(dataSet);
        }
        ProceedDataSet(dataSetPermutationId, *dataSets[idx], visitor);
    }
}

void NCatboostCuda::TTreeCtrDataSetsHelper::ProceedDataSet(ui32 dataSetPermutationKey,
                                                           NCatboostCuda::TTreeCtrDataSet& dataSet,
                                                           NCatboostCuda::TTreeCtrDataSetsHelper::TDataSetVisitor& visitor) {
    const ui32 deviceId = dataSet.GetDeviceId();
    auto ctrTargets = DeviceView(DataSet.GetCtrTargets(), deviceId);

    if (!dataSet.HasCompressedIndex()) {
        NCudaLib::GetCudaManager().WaitComplete();

        auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile(
            TStringBuilder() << "Build  #" << dataSet.GetCtrs().size() << " ctrs dataset");

        using TTensorBuilder = TBatchFeatureTensorBuilder;
        const ui32 tensorBuilderStreams = PackSizeEstimators[deviceId]->GetStreamCountForCtrCalculation();

        TTreeCtrDataSetBuilder builder(DepthPermutations[dataSetPermutationKey].DeviceView(deviceId),
                                       dataSet,
                                       tensorBuilderStreams > 1,
                                       !LevelBasedCompressedIndex);

        NCudaLib::GetCudaManager().WaitComplete();

        TTensorBuilder batchFeatureTensorBuilder(FeaturesManager,
                                                 DataSet.GetCatFeatures(),
                                                 tensorBuilderStreams);

        TVector<ui32> catFeatureIds(dataSet.GetCatFeatures().begin(),
                                    dataSet.GetCatFeatures().end());
        TCtrFromTensorCalcer ctrFromTensorCalcer(builder,
                                                 dataSet.GetCtrConfigs(),
                                                 ctrTargets);

        TBatchFeatureTensorBuilder::TFeatureTensorVisitor ctrFromTensorCalcerFunc = [&](const TFeatureTensor& tensor, TCtrBinBuilder<NCudaLib::TSingleMapping>& builder) {
            ctrFromTensorCalcer(tensor, builder);
        };
        batchFeatureTensorBuilder.VisitCtrBinBuilders(dataSet.GetBaseTensorIndices(),
                                                      dataSet.GetBaseTensor(),
                                                      catFeatureIds,
                                                      ctrFromTensorCalcerFunc);
        NCudaLib::GetCudaManager().WaitComplete();
    }

    visitor(dataSet);

    if (NeedToDropDataSetAfterVisit(deviceId)) {
        TTreeCtrDataSetBuilder::DropCache(dataSet);
    }
}

bool NCatboostCuda::TTreeCtrDataSetsHelper::NeedToDropDataSetAfterVisit(ui32 deviceId) const {
    if (IsLastLevel()) {
        return true;
    }
    auto freeMemory = GetFreeMemory(deviceId);

    if (freeMemory < (MinFreeMemory + DataSet.GetDataProvider().GetObjectCount() * 12.0 / 1024 / 1024)) {
        return true;
    }
    return false;
}

void NCatboostCuda::TTreeCtrDataSetsHelper::FreeMemoryForDataSet(const NCatboostCuda::TTreeCtrDataSet& dataSet) {
    bool isDone = FreeMemoryForDataSet(dataSet, PureTreeCtrDataSets[dataSet.GetDeviceId()]);
    if (!isDone) {
        FreeMemoryForDataSet(dataSet, DataSets[dataSet.GetDeviceId()]);
    }
}
