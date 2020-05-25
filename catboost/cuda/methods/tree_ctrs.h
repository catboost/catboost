#pragma once

#include "tree_ctrs_dataset.h"
#include "tree_ctr_memory_usage_estimator.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/private/libs/options/binarization_options.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/data/feature.h>

namespace NCatboostCuda {
    class TTreeCtrDataSetBuilder {
    public:
        using TVec = TCudaBuffer<float, NCudaLib::TSingleMapping>;
        using TConstVec = TCudaBuffer<const float, NCudaLib::TSingleMapping>;
        using TCompressedIndex = TTreeCtrDataSet::TCompressedIndex;

        TTreeCtrDataSetBuilder(const TCudaBuffer<const ui32, NCudaLib::TSingleMapping>& indices,
                               TTreeCtrDataSet& ctrDataSet,
                               bool streamParallelCtrVisits,
                               bool isIdentityPermutation = false);

        void operator()(const TCtr& ctr,
                        const TVec& floatCtr,
                        ui32 stream);

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

            bool IsEffectivelyOneHot(ui32) const {
                return false;
            }
            bool SkipInSplitSearch(ui32) const {
                return false;
            }

            bool SkipFirstBucketInOneHot(ui32) const {
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

        THolder<TCompressedIndex> CreateCompressedIndex(NCudaLib::TSingleMapping docsMapping);

        TConstVec GetBorders(const TCtr& ctr,
                             const TVec& floatCtr,
                             ui32 stream);

        void ComputeCtrBorders(const TVec& ctr,
                               const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                               ui32 stream,
                               TVec& dst);

    private:
        TTreeCtrDataSet& TreeCtrDataSet;
        TCudaBuffer<const ui32, NCudaLib::TSingleMapping> GatherIndices;
        bool StreamParallelCtrVisits;
        bool IsIdentityPermutation;
    };

    class TTreeCtrDataSetsHelper {
    public:
        using TTreeCtrDataSetPtr = THolder<TTreeCtrDataSet>;
        using TDataSetVisitor = std::function<void(const TTreeCtrDataSet&)>;

    public:
        TTreeCtrDataSetsHelper(const TFeatureParallelDataSet& dataSet,
                               const TBinarizedFeaturesManager& featuresManager,
                               ui32 maxDepth, ui32 foldCount,
                               TFeatureTensorTracker& emptyTracker);

        const TCudaBuffer<const ui32, NCudaLib::TMirrorMapping>& GetPermutationIndices(ui32 depth) {
            CB_ENSURE(DepthPermutations[depth].GetObjectsSlice().Size());
            return DepthPermutations[depth];
        };

        void AddSplit(const TBinarySplit& split,
                      const TMirrorBuffer<ui32>& docBins);

        const TSet<ui32>& GetUsedPermutations() const {
            return UsedPermutations;
        }

        void VisitPermutationDataSets(ui32 permutationId,
                                      TDataSetVisitor& visitor);

        ui32 GetMaxUniqueValues() const;

    private:
        void AddDataSets(const TVector<TTreeCtrDataSetPtr>& dataSets, ui32 permutationId, bool withCompressedIndexFlag,
                         TVector<TTreeCtrDataSet*>& dst);

        void AddNewDataSets(const TFeatureTensor& tensor);

        void UpdatePureTreeCtrTensor(const TBinarySplit& split);

        void ProceedDataSets(const ui32 dataSetPermutationId,
                             const TVector<TTreeCtrDataSet*>& dataSets,
                             TDataSetVisitor& visitor);

        void ProceedDataSets(ui32 dataSetPermutationId,
                             const TVector<TTreeCtrDataSet*>& dataSets,
                             bool withCompressedIndex,
                             TDataSetVisitor& visitor);

        bool FreeMemoryForDataSet(const TTreeCtrDataSet& dataSet,
                                  TVector<TTreeCtrDataSetPtr>& dataSets);

        double GetFreeMemory(ui32 deviceId) const {
            auto& manager = NCudaLib::GetCudaManager();
            return manager.FreeMemoryMb(deviceId) - PackSizeEstimators[deviceId]->GetReserveMemory(CurrentDepth);
        }

        void FreeMemoryForDataSet(const TTreeCtrDataSet& dataSet);

        void ProceedDataSet(ui32 dataSetPermutationKey,
                            TTreeCtrDataSet& dataSet,
                            TDataSetVisitor& visitor);

        void CachePermutation(const TMirrorBuffer<ui32>& currentBins,
                              ui32 depth);

        void SortDataSetsByCompressedIndexLevelAndSize();

        void UpdateForPack(const TVector<TVector<TTreeCtrDataSetPtr>>& dataSets, TSet<ui32>& usedPermutations);

        void UpdateUsedPermutations();

        void ClearUnusedPermutations();

        bool AssignForPack(TVector<TVector<TTreeCtrDataSetPtr>>& dataSets, ui32 depth);

        bool AssignDepthForDataSetsWithoutCompressedIndex(ui32 depth) {
            bool assignedDataSets = AssignForPack(DataSets, depth);
            bool assignedPureTreeCtrDataSets = AssignForPack(PureTreeCtrDataSets, depth);
            return assignedDataSets | assignedPureTreeCtrDataSets;
        }

        void AddDataSetPacks(const TFeatureTensor& baseTensor,
                             const TSingleBuffer<const ui32>& baseTensorIndices,
                             ui32 deviceId,
                             TVector<TTreeCtrDataSetPtr>& dst);

        bool NeedToDropDataSetAfterVisit(ui32 deviceId) const;

        TFeatureTensorTracker CreateEmptyTrackerForTensor(const TFeatureTensor& tensor);

        TFeatureTensorTracker CreateTensorTracker(const TFeatureTensor& tensor) {
            auto tracker = CreateEmptyTrackerForTensor(tensor);
            tracker.AddFeatureTensor(tensor);
            return tracker;
        }

        bool IsLastLevel() const {
            return ((CurrentDepth + 1) == MaxDepth);
        }

    private:
        const TFeatureParallelDataSet& DataSet;
        const TBinarizedFeaturesManager& FeaturesManager;

        TVector<TVector<TTreeCtrDataSetPtr>> DataSets;
        TVector<THolder<TTreeCtrDataSetMemoryUsageEstimator>> PackSizeEstimators;
        TVector<TVector<TTreeCtrDataSetPtr>> PureTreeCtrDataSets;

        TVector<TCudaBuffer<const ui32, NCudaLib::TMirrorMapping>> DepthPermutations;
        TSet<ui32> UsedPermutations;

        TFeatureTensorTracker EmptyTracker;
        TMap<TFeatureTensor, TFeatureTensorTracker> TensorTrackers;

        TFeatureTensorTracker PureTreeCtrTensorTracker;
        TFeatureTensor CurrentTensor;

        double MinFreeMemory = 8;
        const ui32 MaxDepth;
        const ui32 FoldCount;
        ui32 CurrentDepth = 0;
        bool LevelBasedCompressedIndex = false;
    };

}
