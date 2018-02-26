#pragma once

#include "ctr_helper.h"
#include "splitter.h"
#include "feature_parallel_dataset.h"

#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/gpu_data/compressed_index_builder.cpp>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/ctrs/ctr_bins_builder.h>
#include <catboost/cuda/ctrs/ctr_calcers.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/cat_features_dataset.h>
#include <catboost/cuda/data/helpers.h>

namespace NCatboostCuda {
    class IBinarySplitProvider {
    public:
        virtual ~IBinarySplitProvider() = default;

        virtual const TMirrorBuffer<ui64>& GetCompressedBits(const TBinarySplit& split) const = 0;
    };

    template <NCudaLib::EPtrType CatFeaturesStoragePtrType>
    class TFeatureTensorTracker: public TMoveOnly {
    public:
        TFeatureTensorTracker() = default;

        TFeatureTensorTracker(TScopedCacheHolder& cacheHolder,
                              const TBinarizedFeaturesManager& featuresManager,
                              const TFeatureParallelDataSet<CatFeaturesStoragePtrType>& learnSet,
                              const IBinarySplitProvider& binarySplitProvider,
                              const TFeatureParallelDataSet<CatFeaturesStoragePtrType>* testSet = nullptr,
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

        bool AddFeatureTensor(const TFeatureTensor& featureTensor) {
            const TFeatureTensor prevTensor = CurrentTensor;
            CurrentTensor.AddTensor(featureTensor);
            if (prevTensor == CurrentTensor) {
                return false;
            }

            TCtrBinBuilder<NCudaLib::TMirrorMapping> binBuilder(Stream);

            if (LearnSlice.Size() == 0) {
                binBuilder.SetIndices(LearnDataSet->GetIndices(),
                                      LinkedTest ? &(LinkedTest->GetIndices()) : nullptr);
                LearnSlice = binBuilder.GetLearnSlice();
            } else {
                binBuilder.SetIndices(std::move(Indices),
                                      LearnSlice); //some ping-pong with buffer
            }

            auto catFeatures = RemoveExisting(CurrentTensor.GetCatFeatures(), prevTensor.GetCatFeatures());
            auto binarySplits = RemoveExisting(CurrentTensor.GetSplits(), prevTensor.GetSplits());

            TMirrorCatFeatureProvider<CatFeaturesStoragePtrType> broadcastedCatFeatures(LearnDataSet->GetCatFeatures(),
                                                                                        *CacheHolder);

            for (ui32 catFeature : catFeatures) {
                const auto& compressedFeatures = broadcastedCatFeatures.BroadcastFeature(catFeature, Stream)
                                                     .GetInStream(Stream);

                if (LinkedTest) {
                    TMirrorCatFeatureProvider<CatFeaturesStoragePtrType> broadcastedTestCatFeatures(
                        LinkedTest->GetCatFeatures(),
                        *CacheHolder);
                    const auto& compressedTest = broadcastedTestCatFeatures.BroadcastFeature(catFeature, Stream)
                                                     .GetInStream(Stream);
                    binBuilder.AddCompressedBins(compressedFeatures,
                                                 compressedTest,
                                                 FeaturesManager->GetBinCount(catFeature));
                } else {
                    binBuilder.AddCompressedBins(compressedFeatures,
                                                 FeaturesManager->GetBinCount(catFeature));
                }
            }

            //TODO(noxoomo): builder should merge several task in one
            for (const auto& binarySplit : binarySplits) {
                const auto& learnBits = LearnBinarySplitsProvider->GetCompressedBits(binarySplit);

                if (LinkedTest) {
                    const auto& testBits = TestBinarySplitsProvider->GetCompressedBits(binarySplit);

                    binBuilder.AddCompressedBins(learnBits,
                                                 testBits,
                                                 2);
                } else {
                    binBuilder.AddCompressedBins(learnBits,
                                                 2);
                }
            }

            Indices = std::move(binBuilder.MoveIndices());

            return true;
        }

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

        TFeatureTensorTracker Copy() {
            TFeatureTensorTracker other(*CacheHolder, *FeaturesManager,
                                        *LearnDataSet, *LearnBinarySplitsProvider,
                                        LinkedTest, TestBinarySplitsProvider,
                                        Stream);

            other.CurrentTensor = CurrentTensor;
            other.LearnSlice = LearnSlice;
            other.Indices.Reset(Indices.GetMapping());
            other.Indices.Copy(Indices, Stream);
            return other;
        }

    private:
        TScopedCacheHolder* CacheHolder = nullptr;
        const TBinarizedFeaturesManager* FeaturesManager = nullptr;

        const TFeatureParallelDataSet<CatFeaturesStoragePtrType>* LearnDataSet = nullptr;
        const IBinarySplitProvider* LearnBinarySplitsProvider = nullptr;

        const TFeatureParallelDataSet<CatFeaturesStoragePtrType>* LinkedTest = nullptr;
        const IBinarySplitProvider* TestBinarySplitsProvider = nullptr;

        TFeatureTensor CurrentTensor;
        TMirrorBuffer<ui32> Indices;
        TSlice LearnSlice;
        ui32 Stream = 0;
    };

    template <class TDataSet,
              class TCtrSplitBuilder>
    class TSplitHelper: public IBinarySplitProvider {
    public:
        TSplitHelper(TScopedCacheHolder& scopedCache,
                     TCtrSplitBuilder& builder,
                     const TBinarizedFeaturesManager& featuresManager,
                     const TDataSet& dataSet)
            : ScopedCache(scopedCache)
            , CtrSplitBuilder(builder)
            , FeaturesManager(featuresManager)
            , DataSet(dataSet)
        {
        }

        const TMirrorBuffer<ui64>& GetCompressedBits(const TBinarySplit& split) const override {
            const ui32 featureId = split.FeatureId;
            if (DataSet.HasFeatures() && DataSet.GetFeatures().HasFeature(featureId)) {
                return GetCompressedBitsFromGpuFeatures(DataSet.GetFeatures(), split, nullptr);
            } else if (DataSet.HasPermutationDependentFeatures() && DataSet.GetPermutationFeatures().HasFeature(featureId)) {
                return GetCompressedBitsFromGpuFeatures(DataSet.GetPermutationFeatures(),
                                                        split,
                                                        &DataSet.GetInverseIndices());
            } else if (FeaturesManager.IsTreeCtr(split.FeatureId)) {
                return CtrSplitBuilder.ComputeAndCacheCtrSplit(DataSet,
                                                               split);
            } else {
                ythrow TCatboostException() << "Error: unknown feature";
            }
        }

        void Split(const TBinarySplit& split,
                   TMirrorBuffer<ui32>& bins,
                   ui32 depth) {
            const auto& compressedBits = GetCompressedBits(split);
            UpdateBinFromCompressedBits(compressedBits,
                                        bins,
                                        depth);
        }

        template <class TUi64>
        void SplitByExternalComputedFeature(const TBinarySplit& split,
                                            const TSingleBuffer<TUi64>& compressedBits,
                                            TMirrorBuffer<ui32>& dst,
                                            ui32 depth) {
            CB_ENSURE(FeaturesManager.IsTreeCtr(split.FeatureId), "Feature id should be combinations ctr");

            const auto& ctr = FeaturesManager.GetCtr(split.FeatureId);

            const ui32 docCount = DataSet.GetSamplesMapping().GetObjectsSlice().Size();
            const ui32 compressedSize = CompressedSize<ui64>(docCount, 2);
            auto broadcastFunction = [&]() -> TMirrorBuffer<ui64> {
                TMirrorBuffer<ui64> broadcastedBits = TMirrorBuffer<ui64>::Create(NCudaLib::TMirrorMapping(compressedSize));
                Reshard(compressedBits, broadcastedBits);
                return broadcastedBits;
            };

            const auto& mirrorCompressedBits = [&]() -> const TMirrorBuffer<ui64>& {
                if (FeaturesManager.IsPermutationDependent(ctr)) {
                    return ScopedCache.Cache(DataSet.GetPermutationDependentScope(),
                                             split,
                                             broadcastFunction);
                } else {
                    return ScopedCache.Cache(DataSet.GetPermutationIndependentScope(),
                                             split,
                                             broadcastFunction);
                }
            }();

            UpdateBinFromCompressedBits(mirrorCompressedBits, dst, depth);
        }

    private:
        const TMirrorBuffer<ui64>& GetCompressedBitsFromGpuFeatures(const TCompressedDataSet<>& dataSet,
                                                                    const TBinarySplit& split,
                                                                    const TMirrorBuffer<ui32>* readIndices) const {
            const ui32 featureId = split.FeatureId;
            CB_ENSURE(dataSet.HasFeature(featureId), TStringBuilder() << "Error: can't get compressed bits for feature " << featureId);
            return ScopedCache.Cache(dataSet, split, [&]() -> TMirrorBuffer<ui64> {
                return BuildMirrorSplitForDataSet(dataSet, split, readIndices);
            });
        }

        //        template <class TBinarizedDataSet>
        TMirrorBuffer<ui64> BuildMirrorSplitForDataSet(const TCompressedDataSet<>& ds,
                                                       const TBinarySplit& split,
                                                       const TMirrorBuffer<ui32>* readIndices = nullptr) const {
            auto feature = ds.GetTCFeature(split.FeatureId);
            const ui32 docCount = ds.GetDocCount();

            const ui32 compressedSize = CompressedSize<ui64>(docCount, 2);
            TMirrorBuffer<ui64> broadcastedBits = TMirrorBuffer<ui64>::Create(NCudaLib::TMirrorMapping(compressedSize));

            const ui32 devCount = GetDeviceCount();

            for (ui32 dev = 0; dev < devCount; ++dev) {
                if (!feature.IsEmpty(dev)) {
                    TSingleBuffer<ui64> compressedBits = TSingleBuffer<ui64>::Create(NCudaLib::TSingleMapping(dev, compressedSize));
                    TSingleBuffer<const ui32> indices;
                    if (readIndices) {
                        indices = readIndices->DeviceView(dev);
                    }
                    CreateCompressedSplit(ds, feature, split.BinIdx, compressedBits, readIndices ? &indices : nullptr);
                    Reshard(compressedBits, broadcastedBits);
                    break;
                }
                CB_ENSURE(dev + 1 != devCount, TStringBuilder() << "Error : feature was not found " << split.FeatureId);
            }
            return broadcastedBits;
        }

        TScopedCacheHolder& ScopedCache;
        TCtrSplitBuilder& CtrSplitBuilder;
        const TBinarizedFeaturesManager& FeaturesManager;
        const TDataSet& DataSet;
    };

    template <class TDataSet>
    class TTreeUpdater: public TNonCopyable {
    public:
        static constexpr NCudaLib::EPtrType CatFeatureStoragePtrType = TDataSet::GetCatFeaturesStoragePtrType();
        using TTensorTracker = TFeatureTensorTracker<CatFeatureStoragePtrType>;

        TTreeUpdater(TScopedCacheHolder& cacheHolder,
                     const TBinarizedFeaturesManager& featuresManager,
                     const TCtrTargets<NCudaLib::TMirrorMapping>& ctrTargets,
                     const TDataSet& learnSet,
                     TMirrorBuffer<ui32>& learnBins,
                     const TDataSet* testSet = nullptr,
                     TMirrorBuffer<ui32>* testBins = nullptr)
            : FeaturesManager(featuresManager)
            , CacheHolder(cacheHolder)
            , CtrTargets(ctrTargets)
            , LearnDataSet(learnSet)
            , LinkedTest(testSet)
            , SplitHelper(CacheHolder, *this, featuresManager, LearnDataSet)
            , LearnBins(learnBins)
            , TestBins(testBins)
        {
            if (LinkedTest) {
                TestSplitHelper.Reset(new TSplitHelper<TDataSet, TTreeUpdater>(CacheHolder,
                                                                               *this,
                                                                               FeaturesManager,
                                                                               *LinkedTest));
            }

            FillBuffer(LearnBins, 0u);

            if (LinkedTest) {
                FillBuffer(*TestBins, 0u);
            }
        }

        ~TTreeUpdater() {
        }

        TTreeUpdater& AddSplit(const TBinarySplit& split) {
            //TODO(noxoomo): current caching logic is too complicated. It should be easier
            SplitHelper.Split(split, LearnBins, (ui32)BinarySplits.size());

            if (LinkedTest) {
                TestSplitHelper->Split(split, *TestBins, (ui32)BinarySplits.size());
            }

            BinarySplits.push_back(split);
            return *this;
        }

        template <class TUi64>
        TTreeUpdater& AddSplit(const TBinarySplit& split,
                               const TSingleBuffer<TUi64>& compressedBins) {
            CB_ENSURE(LinkedTest == nullptr, "Error: need test bins to add externally-computed split");

            SplitHelper.SplitByExternalComputedFeature(split,
                                                       compressedBins,
                                                       LearnBins,
                                                       (ui32)BinarySplits.size());

            if (LinkedTest) {
                TestSplitHelper->Split(split, *TestBins, (ui32)BinarySplits.size());
            }

            BinarySplits.push_back(split);
            return *this;
        }

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
                                                           const TBinarySplit& split) {
            CB_ENSURE(&dataSet == &LearnDataSet || (&dataSet == LinkedTest));
            using TCtrHelper = TCalcCtrHelper<NCudaLib::TMirrorMapping>;
            CB_ENSURE(FeaturesManager.IsCtr(split.FeatureId), "Error: split should be ctr");
            const auto& ctr = FeaturesManager.GetCtr(split.FeatureId);

            bool floatCtrBuilderWasCalledFlag = false;
            auto floatCtrBuilder = [&]() -> TMirrorBuffer<float> {
                if (!CanContinueTensorTracker(ctr.FeatureTensor)) {
                    TensorTracker = CreateEmptyTensorTracker();
                }

                TensorTracker->AddFeatureTensor(ctr.FeatureTensor);
                TCtrHelper helper(CtrTargets,
                                  TensorTracker->GetIndices());
                floatCtrBuilderWasCalledFlag = true;
                auto floatCtr = helper.ComputeCtr(ctr.Configuration);
                return floatCtr;
            };

            TMirrorBuffer<float> floatCtr;
            auto& borders = FeaturesManager.GetBorders(split.FeatureId);

            auto buildLearnBitsFunction = [&]() -> TMirrorBuffer<ui64> {
                if (!floatCtrBuilderWasCalledFlag) {
                    floatCtr = floatCtrBuilder();
                }
                return CreateSplit(floatCtr, borders[split.BinIdx], CtrTargets.LearnSlice);
            };
            auto buildTestBitsFunction = [&]() -> TMirrorBuffer<ui64> {
                if (!floatCtrBuilderWasCalledFlag) {
                    floatCtr = floatCtrBuilder();
                }
                return CreateSplit(floatCtr, borders[split.BinIdx], CtrTargets.TestSlice);
            };

            const auto& compressedSplitLearn = CacheTreeCtrSplit(LearnDataSet, split, buildLearnBitsFunction);
            const TMirrorBuffer<ui64>* compressedSplitTest = LinkedTest ? &CacheTreeCtrSplit(*LinkedTest, split,
                                                                                             buildTestBitsFunction)
                                                                        : nullptr;

            if (&dataSet == LinkedTest) {
                CB_ENSURE(compressedSplitTest);
                return *compressedSplitTest;
            }
            return compressedSplitLearn;
        }

        TMirrorBuffer<ui64> CreateSplit(const TMirrorBuffer<float>& ctr,
                                        const float border,
                                        TSlice slice) {
            auto compressedSplit = TMirrorBuffer<ui64>::Create(
                NCudaLib::TMirrorMapping(CompressedSize<ui64>(static_cast<ui32>(slice.Size()), 2)));
            CreateCompressedSplitFloat(ctr.SliceView(slice),
                                       border,
                                       compressedSplit);
            return compressedSplit;
        }

        THolder<TTensorTracker> CreateEmptyTensorTracker() {
            return MakeHolder<TTensorTracker>(CacheHolder, FeaturesManager,
                                              LearnDataSet, SplitHelper,
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

        TSplitHelper<TDataSet, TTreeUpdater> SplitHelper;
        THolder<TSplitHelper<TDataSet, TTreeUpdater>> TestSplitHelper;

        TMirrorBuffer<ui32>& LearnBins;
        TMirrorBuffer<ui32>* TestBins;
    };
}
