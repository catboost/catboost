#pragma once

#include "cat_features_dataset.h"
#include "samples_grouping.h"
#include "compressed_index.h"
#include "dataset_base.h"

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/permutation.h>
#include <util/system/env.h>

namespace NCatboostCuda {
    struct TDocParallelSplit {
        TDataPermutation Permutation;
        //wrt samples permutation
        TAtomicSharedPtr<IQueriesGrouping> SamplesGrouping;
        NCudaLib::TStripeMapping Mapping;

        explicit TDocParallelSplit(const TDataProvider& dataProvider,
                                   const TDataPermutation& permutation)
            : Permutation(permutation)
        {
            if (dataProvider.HasQueries()) {
                auto permutedQids = Permutation.Gather(dataProvider.GetQueryIds());
                auto samplesGrouping = MakeHolder<TQueriesGrouping>(std::move(permutedQids),
                                                                    dataProvider.GetPairs());
                if (dataProvider.HasSubgroupIds()) {
                    auto permutedGids = permutation.Gather(dataProvider.GetSubgroupIds());
                    samplesGrouping->SetSubgroupIds(std::move(permutedGids));
                }
                SamplesGrouping = std::move(samplesGrouping);
            } else {
                SamplesGrouping.Reset(new TWithoutQueriesGrouping(dataProvider.GetSampleCount()));
            }

            auto queriesSplit = NCudaLib::TStripeMapping::SplitBetweenDevices((ui64)SamplesGrouping->GetQueryCount());
            Mapping = queriesSplit.Transform([&](const TSlice queriesSlice) {
                return SamplesGrouping->GetQueryOffset(queriesSlice.Right) - SamplesGrouping->GetQueryOffset(queriesSlice.Left);
            });
        }
    };

    class TDocParallelDataSet: public TDataSetBase<TDocParallelLayout> {
    public:
        using TParent = TDataSetBase<TDocParallelLayout>;
        using TLayout = TDocParallelLayout;
        using TCompressedIndex = typename TParent::TCompressedIndex;
        using TCompressedDataSet = typename TParent::TCompressedDataSet;
        using TDocsMapping = typename TDocParallelLayout::TSamplesMapping;

        const IQueriesGrouping& GetSamplesGrouping() const {
            return *SamplesGrouping;
        }

        //permutation used to shuffle docs between devices
        const TDataPermutation& GetLoadBalancingPermutation() const {
            return LoadBalancingPermutation;
        }

    private:
        TDocParallelDataSet(const TDataProvider& dataProvider,
                            TAtomicSharedPtr<TCompressedIndex> compressedIndex,
                            const TDataPermutation& ctrsEstimationPermutation,
                            const TDataPermutation& loadBalancingPermutation,
                            TAtomicSharedPtr<IQueriesGrouping> grouping,
                            TTarget<NCudaLib::TStripeMapping>&& target)
            : TDataSetBase<TDocParallelLayout>(dataProvider,
                                               compressedIndex,
                                               ctrsEstimationPermutation,
                                               std::move(target))
            , SamplesGrouping(grouping)
            , LoadBalancingPermutation(loadBalancingPermutation)
        {
        }

    private:
        TAtomicSharedPtr<IQueriesGrouping> SamplesGrouping;
        TDataPermutation LoadBalancingPermutation;
        friend class TDocParallelDataSetBuilder;
    };

    class TDocParallelDataSetsHolder: public TGuidHolder {
    public:
        using TCompressedIndex = TSharedCompressedIndex<TDocParallelLayout>;
        using TCompressedDataSet = typename TCompressedIndex::TCompressedDataSet;

        const TDocParallelDataSet& GetDataSetForPermutation(ui32 permutationId) const {
            const auto* dataSetPtr = PermutationDataSets.at(permutationId).Get();
            CB_ENSURE(dataSetPtr);
            return *dataSetPtr;
        }

        const TDataProvider& GetDataProvider() const {
            CB_ENSURE(DataProvider);
            return *DataProvider;
        }

        const TDataProvider& GetTestDataProvider() const {
            CB_ENSURE(TestDataProvider);
            return *TestDataProvider;
        }

        const TBinarizedFeaturesManager& GetFeaturesManger() const {
            CB_ENSURE(FeaturesManager);
            return *FeaturesManager;
        }

        ui32 PermutationsCount() const {
            return (const ui32)PermutationDataSets.size();
        }

        const TDataPermutation& GetCtrEstimationPermutation(ui32 permutationId) const {
            return PermutationDataSets[permutationId]->GetCtrsEstimationPermutation();
        }

        const TDataPermutation& GetLoadBalancingPermutation() const {
            return LearnDocPerDevicesSplit->Permutation;
        }

        const TDataPermutation& GetTestLoadBalancingPermutation() const {
            CB_ENSURE(HasTestDataSet());
            return TestDocPerDevicesSplit->Permutation;
        }

        TDocParallelDataSetsHolder() = default;

        TDocParallelDataSetsHolder(const TDataProvider& dataProvider,
                                   const TBinarizedFeaturesManager& featuresManager,
                                   const TDataProvider* testProvider = nullptr)
            : DataProvider(&dataProvider)
            , TestDataProvider(testProvider)
            , FeaturesManager(&featuresManager)
        {
            CompressedIndex = new TCompressedIndex();
            //            const ui32 loadBalancingPermutationId = 42;
            const ui32 loadBalancingPermutationId = FromString<ui32>(GetEnv("CB_LOAD_BALANCE_PERMUTATION", "42"));

            LearnDocPerDevicesSplit = new TDocParallelSplit(*DataProvider,
                                                            GetPermutation(dataProvider,
                                                                           loadBalancingPermutationId));
            if (TestDataProvider) {
                TestDocPerDevicesSplit = new TDocParallelSplit(*TestDataProvider,
                                                               GetPermutation(*TestDataProvider,
                                                                              loadBalancingPermutationId));
            }
        }

        bool HasTestDataSet() const {
            return TestDataSet != nullptr;
        }

        const TDocParallelDataSet& GetTestDataSet() const {
            CB_ENSURE(HasTestDataSet());
            return *TestDataSet;
        }

        bool IsEmpty() const {
            return FeaturesManager == nullptr;
        }

    private:
        const TDataProvider* DataProvider = nullptr;
        const TDataProvider* TestDataProvider = nullptr;
        const TBinarizedFeaturesManager* FeaturesManager = nullptr;
        //learn target and weights
        TStripeBuffer<float> Target;
        TStripeBuffer<float> Weights;
        TAtomicSharedPtr<TCompressedIndex> CompressedIndex;

        TVector<THolder<TDocParallelDataSet>> PermutationDataSets;
        THolder<TDocParallelDataSet> TestDataSet;

        THolder<TDocParallelSplit> LearnDocPerDevicesSplit;
        THolder<TDocParallelSplit> TestDocPerDevicesSplit;

        friend class TDocParallelDataSetBuilder;
    };

}
