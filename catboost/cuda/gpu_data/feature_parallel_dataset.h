#pragma once

#include "cat_features_dataset.h"
#include "ctr_helper.h"
#include "samples_grouping.h"
#include "feature_layout_feature_parallel.h"
#include "compressed_index.h"
#include "dataset_base.h"

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/permutation.h>

namespace NCatboostCuda {
    class TPermutationScope: public TThreadSafeGuidHolder {
    };

    template <NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::EPtrType::CudaDevice>
    class TFeatureParallelDataSet: public TDataSetBase<TFeatureParallelLayout>, public TGuidHolder {
    public:
        using TCompressedIndex = TSharedCompressedIndex<TFeatureParallelLayout>;
        using TCompressedDataSet = typename TCompressedIndex::TCompressedDataSet;
        using TSampelsMapping = typename TFeatureParallelLayout::TSamplesMapping;
        using TParent = TDataSetBase<TFeatureParallelLayout>;

        static constexpr NCudaLib::EPtrType GetCatFeaturesStoragePtrType() {
            return CatFeaturesStoragePtrType;
        };

        //target-ctr_type
        const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& GetCatFeatures() const {
            return CatFeatures;
        }

        const IQueriesGrouping& GetSamplesGrouping() const {
            return *SamplesGrouping;
        }

        const TCtrTargets<NCudaLib::TMirrorMapping>& GetCtrTargets() const {
            return CtrTargets;
        }

        const TPermutationScope& GetPermutationIndependentScope() const {
            return *PermutationIndependentScope;
        }

        const TPermutationScope& GetPermutationDependentScope() const {
            return *PermutationDependentScope;
        }

        bool HasCtrHistoryDataSet() const {
            return LinkedHistoryForCtrs != nullptr;
        }

        const TFeatureParallelDataSet& LinkedHistoryForCtr() const {
            CB_ENSURE(HasCtrHistoryDataSet(), "No history dataset found");
            return *LinkedHistoryForCtrs;
        }

        //doc-indexing
        const TMirrorBuffer<const ui32>& GetIndices() const {
            return TParent::GetTarget().GetIndices();
        }

        //doc-indexing
        const TMirrorBuffer<ui32>& GetInverseIndices() const {
            return InverseIndices;
        }

    private:
        TFeatureParallelDataSet(const TDataProvider& dataProvider,
                                TAtomicSharedPtr<TCompressedIndex> compressedIndex,
                                TAtomicSharedPtr<TPermutationScope> permutationIndependentScope,
                                TAtomicSharedPtr<TPermutationScope> permutationDependentScope,
                                const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& catFeatures,
                                const TCtrTargets<NCudaLib::TMirrorMapping>& ctrTargets,
                                TTarget<NCudaLib::TMirrorMapping>&& target,
                                TMirrorBuffer<ui32>&& inverseIndices,
                                TDataPermutation&& permutation)
            : TDataSetBase<TFeatureParallelLayout>(dataProvider,
                                                   compressedIndex,
                                                   std::move(permutation),
                                                   std::move(target))
            , PermutationDependentScope(permutationDependentScope)
            , PermutationIndependentScope(permutationIndependentScope)
            , InverseIndices(std::move(inverseIndices))
            , CtrTargets(ctrTargets)
            , CatFeatures(catFeatures)
        {
            if (dataProvider.HasQueries()) {
                const auto& ctrEstimationPermutation = TParent::GetCtrsEstimationPermutation();
                auto permutedQids = ctrEstimationPermutation.Gather(dataProvider.GetQueryIds());
                auto samplesGrouping = MakeHolder<TQueriesGrouping>(std::move(permutedQids),
                                                                    dataProvider.GetPairs());
                if (dataProvider.HasSubgroupIds()) {
                    auto permutedGids = ctrEstimationPermutation.Gather(dataProvider.GetSubgroupIds());
                    samplesGrouping->SetSubgroupIds(std::move(permutedGids));
                }
                SamplesGrouping = std::move(samplesGrouping);
            } else {
                SamplesGrouping.Reset(new TWithoutQueriesGrouping(dataProvider.GetSampleCount()));
            }
        }

    private:
        TAtomicSharedPtr<TPermutationScope> PermutationDependentScope;
        TAtomicSharedPtr<TPermutationScope> PermutationIndependentScope;

        TMirrorBuffer<ui32> InverseIndices;

        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;
        const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& CatFeatures;

        TFeatureParallelDataSet* LinkedHistoryForCtrs = nullptr;
        THolder<IQueriesGrouping> SamplesGrouping;

        template <NCudaLib::EPtrType CatFeatureStoragePtrType>
        friend class TFeatureParallelDataSetHoldersBuilder;
    };

    template <NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::EPtrType::CudaDevice>
    class TFeatureParallelDataSetsHolder: public TGuidHolder {
    public:
        using TCompressedDataSet = typename TSharedCompressedIndex<TFeatureParallelLayout>::TCompressedDataSet;

        const TFeatureParallelDataSet<CatFeaturesStoragePtrType>& GetDataSetForPermutation(ui32 permutationId) const {
            const auto* dataSetPtr = PermutationDataSets.at(permutationId).Get();
            CB_ENSURE(dataSetPtr);
            return *dataSetPtr;
        }

        const TDataProvider& GetDataProvider() const {
            CB_ENSURE(DataProvider);
            return *DataProvider;
        }

        const TBinarizedFeaturesManager& GetFeaturesManger() const {
            CB_ENSURE(FeaturesManager);
            return *FeaturesManager;
        }

        const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& GetCatFeatures() const {
            return *LearnCatFeaturesDataSet;
        }

        const TCompressedDataSet& GetPermutationIndependentFeatures() const {
            CB_ENSURE(PermutationDataSets[0]);
            return PermutationDataSets[0]->GetFeatures();
        }

        const TCompressedDataSet& GetPermutationDependentFeatures(ui32 permutationId) const {
            CB_ENSURE(PermutationDataSets[permutationId]);
            return PermutationDataSets[permutationId]->GetPermutationFeatures();
        }

        ui32 PermutationsCount() const {
            return (const ui32)PermutationDataSets.size();
        }

        const TDataPermutation& GetPermutation(ui32 permutationId) const {
            return PermutationDataSets[permutationId]->GetCtrsEstimationPermutation();
        }

        TFeatureParallelDataSetsHolder() = default;

        TFeatureParallelDataSetsHolder(const TDataProvider& dataProvider,
                                       const TBinarizedFeaturesManager& featuresManager)
            : DataProvider(&dataProvider)
            , FeaturesManager(&featuresManager)
        {
            CompressedIndex = new TSharedCompressedIndex<TFeatureParallelLayout>();
        }

        bool HasTestDataSet() const {
            return TestDataSet != nullptr;
        }

        const TFeatureParallelDataSet<CatFeaturesStoragePtrType>& GetTestDataSet() const {
            CB_ENSURE(HasTestDataSet());
            return *TestDataSet;
        }

        const TCtrTargets<NCudaLib::TMirrorMapping>& GetCtrTargets() const {
            return *CtrTargets;
        }

        bool IsEmpty() const {
            return FeaturesManager == nullptr;
        }

    private:
        const TDataProvider* DataProvider = nullptr;
        const TBinarizedFeaturesManager* FeaturesManager = nullptr;

        //learn target and weights
        TMirrorBuffer<float> DirectTarget;
        TMirrorBuffer<float> DirectWeights;
        //For tree-ctrs
        THolder<TCtrTargets<NCudaLib::TMirrorMapping>> CtrTargets;
        THolder<TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>> LearnCatFeaturesDataSet;
        THolder<TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>> TestCatFeaturesDataSet;
        //for float features
        TAtomicSharedPtr<TSharedCompressedIndex<TFeatureParallelLayout>> CompressedIndex;

        TVector<THolder<TFeatureParallelDataSet<CatFeaturesStoragePtrType>>> PermutationDataSets;
        THolder<TFeatureParallelDataSet<CatFeaturesStoragePtrType>> TestDataSet;

        template <NCudaLib::EPtrType CatFeatureStoragePtrType>
        friend class TFeatureParallelDataSetHoldersBuilder;
    };
}
