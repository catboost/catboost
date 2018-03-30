#pragma once

#include "tree_ctr_memory_usage_estimator.h"
#include "batch_feature_tensor_builder.h"
#include "histograms_helper.h"
#include "tree_ctrs_dataset.h"
#include "ctr_from_tensor_calcer.h"
#include <catboost/cuda/utils/countdown_latch.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>

#include <util/generic/map.h>
#include <util/generic/hash.h>
#include <util/generic/set.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

namespace NCatboostCuda {
    class TTreeCtrDataSetBuilder {
    public:
        using TVec = TCudaBuffer<float, NCudaLib::TSingleMapping>;
        using TConstVec = TCudaBuffer<const float, NCudaLib::TSingleMapping>;
        using TCompressedIndex = TTreeCtrDataSet::TCompressedIndex;

        template <class TUi32>
        TTreeCtrDataSetBuilder(const TCudaBuffer<TUi32, NCudaLib::TSingleMapping>& indices,
                               TTreeCtrDataSet& ctrDataSet,
                               bool streamParallelCtrVisits,
                               bool isIdentityPermutation = false)

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

        void operator()(const TCtr& ctr,
                        const TVec& floatCtr,
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

        static void DropCache(TTreeCtrDataSet& dataSet) {
            dataSet.CacheHolder.Reset(new TScopedCacheHolder);
            if (dataSet.CompressedIndex) {
                dataSet.CompressedIndex->FlatStorage.Clear();
            }
        }

    private:
        //damn proxy for learn set one-hots
        class TBinarizationInfoProvider {
        public:
            ui32 GetFoldsCount(ui32 featureId) const {
                return FeaturesManager.GetCtrBinarization(Ctrs[featureId]).BorderCount;
            }

            double GetGroupingLevel(ui32 featureId) const {
                return GetFoldsCount(featureId) * 1.0 / 256;
            }

            bool IsOneHot(ui32) const {
                return false;
            }

            TBinarizationInfoProvider(const TVector<TCtr>& ctrs,
                                      const TBinarizedFeaturesManager& featuresManager)
                : Ctrs(ctrs)
                , FeaturesManager(featuresManager)
            {
            }

        private:
            const TVector<TCtr>& Ctrs;
            const TBinarizedFeaturesManager& FeaturesManager;
        };

        THolder<TCompressedIndex> CreateCompressedIndex(NCudaLib::TSingleMapping docsMapping) {
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

        TConstVec GetBorders(const TCtr& ctr,
                             const TVec& floatCtr,
                             ui32 stream) {
            CB_ENSURE(TreeCtrDataSet.InverseCtrIndex.has(ctr));
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
            return TreeCtrDataSet.CtrBorders.SliceView(bordersSlice);
        }

        void ComputeCtrBorders(const TVec& ctr,
                               const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                               ui32 stream,
                               TVec& dst) {
            auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Build ctr borders");
            CB_ENSURE(dst.GetMapping().GetObjectsSlice().Size() == binarizationDescription.BorderCount + 1);
            ComputeBordersOnDevice(ctr,
                                   binarizationDescription,
                                   dst,
                                   stream);
        }

    private:
        TTreeCtrDataSet& TreeCtrDataSet;
        TCudaBuffer<const ui32, NCudaLib::TSingleMapping> GatherIndices;
        bool StreamParallelCtrVisits;
        bool IsIdentityPermutation;
    };

    template <NCudaLib::EPtrType CatFeaturesStoragePtrType>
    class TTreeCtrDataSetsHelper {
    public:
        using TTreeCtrDataSetPtr = THolder<TTreeCtrDataSet>;

    public:
        TTreeCtrDataSetsHelper(const TFeatureParallelDataSet<CatFeaturesStoragePtrType>& dataSet,
                               const TBinarizedFeaturesManager& featuresManager,
                               ui32 maxDepth, ui32 foldCount,
                               TFeatureTensorTracker<CatFeaturesStoragePtrType>& emptyTracker)
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
                PackSizeEstimators[dev] = (new TTreeCtrDataSetMemoryUsageEstimator(featuresManager,
                                                                                   freeMemory,
                                                                                   dataSet.GetCatFeatures().GetFeatureCount(dev),
                                                                                   FoldCount,
                                                                                   MaxDepth,
                                                                                   static_cast<const ui32>(dataSet.GetDataProvider().GetSampleCount()),
                                                                                   CatFeaturesStoragePtrType));
            }
        }

        const TCudaBuffer<const ui32, NCudaLib::TMirrorMapping>& GetPermutationIndices(ui32 depth) {
            CB_ENSURE(DepthPermutations[depth].GetObjectsSlice().Size());
            return DepthPermutations[depth];
        };

        void AddSplit(const TBinarySplit& split,
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
                if (UsedPermutations.has(CurrentDepth)) {
                    CachePermutation(docBins, CurrentDepth);
                }
                //if we don't have enough memory, we don't need to cache first-level permutations
            } else {
                AssignDepthForDataSetsWithoutCompressedIndex(0);
                UsedPermutations = {0};
            }
            SortDataSetsByCompressedIndexLevelAndSize();
        }

        const TSet<ui32>& GetUsedPermutations() const {
            return UsedPermutations;
        }

        template <class TCtrDataSetVisitor>
        void VisitPermutationDataSets(ui32 permutationId,
                                      TCtrDataSetVisitor&& visitor) {
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

    private:
        void AddDataSets(const TVector<TTreeCtrDataSetPtr>& dataSets, ui32 permutationId, bool withCompressedIndexFlag,
                         TVector<TTreeCtrDataSet*>& dst) {
            for (ui32 i = 0; i < dataSets.size(); ++i) {
                if (dataSets[i]->GetCompressedIndexPermutationKey() == permutationId) {
                    if (dataSets[i]->HasCompressedIndex() == withCompressedIndexFlag) {
                        dst.push_back(dataSets[i].Get());
                    }
                }
            }
        }

        void AddNewDataSets(const TFeatureTensor& tensor) {
            auto& manager = NCudaLib::GetCudaManager();
            const ui32 devCount = manager.GetDeviceCount();
            TensorTrackers[tensor] = CreateTensorTracker(tensor);

            for (ui32 dev = 0; dev < devCount; ++dev) {
                AddDataSetPacks(tensor,
                                TensorTrackers[tensor].GetIndices().DeviceView(dev),
                                dev,
                                DataSets[dev]);
            }
        }

        void UpdatePureTreeCtrTensor(const TBinarySplit& split) {
            PureTreeCtrTensorTracker.AddBinarySplit(split);
            PureTreeCtrDataSets.clear();

            auto& manager = NCudaLib::GetCudaManager();
            const ui32 devCount = manager.GetDeviceCount();
            PureTreeCtrDataSets.resize(devCount);

            for (ui32 dev = 0; dev < devCount; ++dev) {
                AddDataSetPacks(PureTreeCtrTensorTracker.GetCurrentTensor(),
                                PureTreeCtrTensorTracker.GetIndices().DeviceView(dev),
                                dev,
                                PureTreeCtrDataSets[dev]);
            }
        }

        template <class TVisitor>
        void ProceedDataSets(const ui32 dataSetPermutationId,
                             const TVector<TTreeCtrDataSet*>& dataSets,
                             TVisitor& visitor) {
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

        template <class TVisitor>
        TVector<TTreeCtrDataSet*> ProceedDataSets(ui32 dataSetPermutationId,
                                                  const TVector<TTreeCtrDataSet*>& dataSets,
                                                  bool withCompressedIndex,
                                                  TVisitor& visitor) {
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

        bool FreeMemoryForDataSet(const TTreeCtrDataSet& dataSet,
                                  TVector<TTreeCtrDataSetPtr>& dataSets) {
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

        double GetFreeMemory(ui32 deviceId) const {
            auto& manager = NCudaLib::GetCudaManager();
            return manager.FreeMemoryMb(deviceId) - PackSizeEstimators[deviceId]->GetReserveMemory(CurrentDepth);
        }

        void FreeMemoryForDataSet(const TTreeCtrDataSet& dataSet) {
            bool isDone = FreeMemoryForDataSet(dataSet, PureTreeCtrDataSets[dataSet.GetDeviceId()]);
            if (!isDone) {
                FreeMemoryForDataSet(dataSet, DataSets[dataSet.GetDeviceId()]);
            }
        }

        template <class TVisitor>
        void ProceedDataSet(ui32 dataSetPermutationKey,
                            TTreeCtrDataSet& dataSet,
                            TVisitor& visitor) {
            const ui32 deviceId = dataSet.GetDeviceId();
            auto ctrTargets = DeviceView(DataSet.GetCtrTargets(), deviceId);

            if (!dataSet.HasCompressedIndex()) {
                NCudaLib::GetCudaManager().WaitComplete();

                auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile(
                    TStringBuilder() << "Build  #" << dataSet.GetCtrs().size() << " ctrs dataset");

                using TTensorBuilder = TBatchFeatureTensorBuilder<CatFeaturesStoragePtrType>;
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
                TCtrFromTensorCalcer<TTreeCtrDataSetBuilder> ctrFromTensorCalcer(builder,
                                                                                 dataSet.GetCtrConfigs(),
                                                                                 ctrTargets);

                batchFeatureTensorBuilder.VisitCtrBinBuilders(dataSet.GetBaseTensorIndices(),
                                                              dataSet.GetBaseTensor(),
                                                              catFeatureIds,
                                                              ctrFromTensorCalcer);
                NCudaLib::GetCudaManager().WaitComplete();
            }

            visitor(dataSet);

            if (NeedToDropDataSetAfterVisit(deviceId)) {
                TTreeCtrDataSetBuilder::DropCache(dataSet);
            }
        }

        void CachePermutation(const TMirrorBuffer<ui32>& currentBins,
                              ui32 depth) {
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

        void SortDataSetsByCompressedIndexLevelAndSize() {
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

        void UpdateForPack(const TVector<TVector<TTreeCtrDataSetPtr>>& dataSets, TSet<ui32>& usedPermutations) {
            for (auto& devDataSets : dataSets) {
                for (auto& ds : devDataSets) {
                    usedPermutations.insert(ds->GetCompressedIndexPermutationKey());
                }
            }
        }

        void UpdateUsedPermutations() {
            TSet<ui32> usedPermutations;
            UpdateForPack(DataSets, usedPermutations);
            UpdateForPack(PureTreeCtrDataSets, usedPermutations);
            UsedPermutations = usedPermutations;
        }

        void ClearUnusedPermutations() {
            for (ui32 i = 0; i < DepthPermutations.size(); ++i) {
                if (UsedPermutations.count(i) == 0) {
                    DepthPermutations[i].Clear();
                }
            }
        }

        bool AssignForPack(TVector<TVector<TTreeCtrDataSetPtr>>& dataSets, ui32 depth) {
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

        bool AssignDepthForDataSetsWithoutCompressedIndex(ui32 depth) {
            bool assignedDataSets = AssignForPack(DataSets, depth);
            bool assignedPureTreeCtrDataSets = AssignForPack(PureTreeCtrDataSets, depth);
            return assignedDataSets | assignedPureTreeCtrDataSets;
        }

        void AddDataSetPacks(const TFeatureTensor& baseTensor,
                             const TSingleBuffer<const ui32>& baseTensorIndices,
                             ui32 deviceId,
                             TVector<TTreeCtrDataSetPtr>& dst) {
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

        bool NeedToDropDataSetAfterVisit(ui32 deviceId) const {
            if (IsLastLevel()) {
                return true;
            }
            auto freeMemory = GetFreeMemory(deviceId);

            if (freeMemory < (MinFreeMemory + DataSet.GetDataProvider().GetSampleCount() * 12.0 / 1024 / 1024)) {
                return true;
            }
            return false;
        }

        TFeatureTensorTracker<CatFeaturesStoragePtrType> CreateEmptyTrackerForTensor(const TFeatureTensor& tensor) {
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

        TFeatureTensorTracker<CatFeaturesStoragePtrType> CreateTensorTracker(const TFeatureTensor& tensor) {
            auto tracker = CreateEmptyTrackerForTensor(tensor);
            tracker.AddFeatureTensor(tensor);
            return tracker;
        }

        bool IsLastLevel() const {
            return ((CurrentDepth + 1) == MaxDepth);
        }

    private:
        const TFeatureParallelDataSet<CatFeaturesStoragePtrType>& DataSet;
        const TBinarizedFeaturesManager& FeaturesManager;

        TVector<TVector<TTreeCtrDataSetPtr>> DataSets;
        TVector<THolder<TTreeCtrDataSetMemoryUsageEstimator>> PackSizeEstimators;
        TVector<TVector<TTreeCtrDataSetPtr>> PureTreeCtrDataSets;

        TVector<TCudaBuffer<const ui32, NCudaLib::TMirrorMapping>> DepthPermutations;
        TSet<ui32> UsedPermutations;

        TFeatureTensorTracker<CatFeaturesStoragePtrType> EmptyTracker;
        TMap<TFeatureTensor, TFeatureTensorTracker<CatFeaturesStoragePtrType>> TensorTrackers;

        TFeatureTensorTracker<CatFeaturesStoragePtrType> PureTreeCtrTensorTracker;
        TFeatureTensor CurrentTensor;

        double MinFreeMemory = 8;
        const ui32 MaxDepth;
        const ui32 FoldCount;
        ui32 CurrentDepth = 0;
        bool LevelBasedCompressedIndex = false;
    };
}
