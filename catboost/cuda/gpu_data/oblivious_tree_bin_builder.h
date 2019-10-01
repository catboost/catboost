#pragma once

#include "ctr_helper.h"
#include "splitter.h"
#include "feature_parallel_dataset.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/ctrs/ctr_bins_builder.h>
#include <catboost/cuda/ctrs/ctr_calcers.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/cat_features_dataset.h>

namespace NCatboostCuda {
    class IBinarySplitProvider {
    public:
        virtual ~IBinarySplitProvider() = default;

        virtual const TMirrorBuffer<ui64>& GetCompressedBits(const TBinarySplit& split) const = 0;

        virtual void Split(const TBinarySplit& split,
                           TMirrorBuffer<ui32>& bins,
                           ui32 depth) = 0;

        virtual void SplitByExternalComputedFeature(const TBinarySplit& split,
                                                    const TSingleBuffer<const ui64>& compressedBits,
                                                    TMirrorBuffer<ui32>& dst,
                                                    ui32 depth) = 0;
    };

    class TFeatureTensorTracker: public TMoveOnly {
    public:
        TFeatureTensorTracker() = default;

        TFeatureTensorTracker(TScopedCacheHolder& cacheHolder,
                              const TBinarizedFeaturesManager& featuresManager,
                              const TFeatureParallelDataSet& learnSet,
                              const IBinarySplitProvider& binarySplitProvider,
                              const TFeatureParallelDataSet* testSet = nullptr,
                              const IBinarySplitProvider* testBinarySplitProvider = nullptr,
                              ui32 stream = 0)
            : CacheHolder(&cacheHolder)
            , FeaturesManager(&featuresManager)
            , LearnDataSet(&learnSet)
            , LearnBinarySplitsProvider(&binarySplitProvider)
            , LinkedTest(testSet)
            , TestBinarySplitsProvider(testBinarySplitProvider)
            , Stream(stream)
        {
        }

        bool AddFeatureTensor(const TFeatureTensor& featureTensor);

        bool AddBinarySplit(const TBinarySplit& split) {
            TFeatureTensor tensor;
            tensor.AddBinarySplit(split);
            return AddFeatureTensor(tensor);
        }

        bool AddCatFeature(const ui32 catFeature) {
            TFeatureTensor tensor;
            tensor.AddCatFeature(catFeature);
            return AddFeatureTensor(tensor);
        }

        const TFeatureTensor& GetCurrentTensor() const {
            return CurrentTensor;
        }

        const TMirrorBuffer<ui32>& GetIndices() const {
            return Indices;
        }

        TFeatureTensorTracker Copy();

    private:
        TScopedCacheHolder* CacheHolder = nullptr;
        const TBinarizedFeaturesManager* FeaturesManager = nullptr;

        const TFeatureParallelDataSet* LearnDataSet = nullptr;
        const IBinarySplitProvider* LearnBinarySplitsProvider = nullptr;

        const TFeatureParallelDataSet* LinkedTest = nullptr;
        const IBinarySplitProvider* TestBinarySplitsProvider = nullptr;

        TFeatureTensor CurrentTensor;
        TMirrorBuffer<ui32> Indices;
        TSlice LearnSlice;
        ui32 Stream = 0;
    };

    class TTreeUpdater: public TNonCopyable {
    public:
        using TDataSet = TFeatureParallelDataSet;
        using TTensorTracker = TFeatureTensorTracker;

        TTreeUpdater(TScopedCacheHolder& cacheHolder,
                     const TBinarizedFeaturesManager& featuresManager,
                     const TCtrTargets<NCudaLib::TMirrorMapping>& ctrTargets,
                     const TDataSet& learnSet,
                     TMirrorBuffer<ui32>& learnBins,
                     const TDataSet* testSet = nullptr,
                     TMirrorBuffer<ui32>* testBins = nullptr);

        ~TTreeUpdater() {
        }

        TTreeUpdater& AddSplit(const TBinarySplit& split);

        TTreeUpdater& AddSplit(const TBinarySplit& split,
                               const TSingleBuffer<const ui64>& compressedBins);

        TObliviousTreeStructure GetCurrentStructure() const {
            return {BinarySplits};
        }

        template <class TBuilder>
        const TMirrorBuffer<ui64>& CacheTreeCtrSplit(const TDataSet& ds,
                                                     const TBinarySplit& split,
                                                     TBuilder&& builder) {
            auto& ctr = FeaturesManager.GetCtr(split.FeatureId);

            if (FeaturesManager.IsPermutationDependent(ctr)) {
                return CacheHolder.Cache(ds.GetPermutationDependentScope(),
                                         split,
                                         std::forward<TBuilder>(builder));
            } else {
                return CacheHolder.Cache(ds.GetPermutationIndependentScope(),
                                         split,
                                         std::forward<TBuilder>(builder));
            }
        }

        //assumes that all features are already cached, so it's faster to compute in mirror-mode, instead of single-dev + broadcast
        //not-optimal for last ctr split if we have only one permutaiton, otherwise it should be fastest way
        const TMirrorBuffer<ui64>& ComputeAndCacheCtrSplit(const TDataSet& dataSet,
                                                           const TBinarySplit& split);

        TMirrorBuffer<ui64> CreateSplit(const TMirrorBuffer<float>& ctr,
                                        const float border,
                                        TSlice slice);

        THolder<TTensorTracker> CreateEmptyTensorTracker() {
            return MakeHolder<TTensorTracker>(CacheHolder, FeaturesManager,
                                              LearnDataSet, *SplitHelper,
                                              LinkedTest, TestSplitHelper.Get());
        }

    private:
        bool CanContinueTensorTracker(const TFeatureTensor& newTensor) const {
            return TensorTracker && TensorTracker->GetCurrentTensor().IsSubset(newTensor);
        }

    private:
        THolder<TTensorTracker> TensorTracker;
        TVector<TBinarySplit> BinarySplits;

        const TBinarizedFeaturesManager& FeaturesManager;
        TScopedCacheHolder& CacheHolder;
        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;

        const TDataSet& LearnDataSet;
        const TDataSet* LinkedTest;

        THolder<IBinarySplitProvider> SplitHelper;
        THolder<IBinarySplitProvider> TestSplitHelper;

        TMirrorBuffer<ui32>& LearnBins;
        TMirrorBuffer<ui32>* TestBins;
    };
}
