#pragma once

#include "cat_features_dataset.h"
#include "samples_grouping.h"
#include "compressed_index.h"
#include "dataset_base.h"

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/data/permutation.h>

#include <catboost/libs/data/data_provider.h>

#include <util/system/env.h>

namespace NCatboostCuda {
    struct TDocParallelSplit {
        TDataPermutation Permutation;
        //wrt samples permutation
        TAtomicSharedPtr<IQueriesGrouping> SamplesGrouping;
        NCudaLib::TStripeMapping Mapping;

        explicit TDocParallelSplit(const NCB::TTrainingDataProvider& dataProvider,
                                   const TDataPermutation& permutation)
            : Permutation(permutation)
        {
            if (dataProvider.MetaInfo.HasGroupId && dataProvider.ObjectsGrouping->GetGroupCount() < dataProvider.ObjectsGrouping->GetObjectCount()) {
                auto samplesGrouping = MakeHolder<TQueriesGrouping>(Permutation,
                                                                    *dataProvider.TargetData->GetGroupInfo(),
                                                                    dataProvider.MetaInfo.HasPairs);
                if (dataProvider.MetaInfo.HasSubgroupIds) {
                    auto permutedGids = permutation.Gather(*dataProvider.ObjectsData->GetSubgroupIds());
                    samplesGrouping->SetSubgroupIds(std::move(permutedGids));
                }
                SamplesGrouping = std::move(samplesGrouping);
            } else {
                SamplesGrouping.Reset(new TWithoutQueriesGrouping(dataProvider.GetObjectCount()));
            }

            auto queriesSplit = NCudaLib::TStripeMapping::SplitBetweenDevices((ui64)SamplesGrouping->GetQueryCount());
            Mapping = queriesSplit.Transform([&](const TSlice queriesSlice) {
                return SamplesGrouping->GetQueryOffset(queriesSlice.Right) - SamplesGrouping->GetQueryOffset(queriesSlice.Left);
            });
        }
    };

    class TDocParallelDataSet: public TDataSetBase<TDocParallelLayout>, public TGuidHolder {
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

        template <class TKey, class TBuilder>
        auto Cache(const TKey& key, TBuilder&& builder) const -> decltype(GetCompressedIndex().GetCache().Cache(*this,
                                                                                                                key,
                                                                                                                std::forward<TBuilder>(builder))) {
            return GetCompressedIndex().GetCache().Cache(*this,
                                                         key,
                                                         std::forward<TBuilder>(builder));
        };

    private:
        TDocParallelDataSet(const NCB::TTrainingDataProvider& dataProvider,
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

        const NCB::TTrainingDataProvider& GetDataProvider() const {
            CB_ENSURE(DataProvider);
            return *DataProvider;
        }

        const NCB::TTrainingDataProvider& GetTestDataProvider() const {
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

        TDocParallelDataSetsHolder(const NCB::TTrainingDataProvider& dataProvider,
                                   const TBinarizedFeaturesManager& featuresManager,
                                   const NCB::TTrainingDataProvider* testProvider = nullptr)
            : DataProvider(&dataProvider)
            , TestDataProvider(testProvider)
            , FeaturesManager(&featuresManager)
        {
            CompressedIndex = new TCompressedIndex();

            const auto getPermutation = [&] (const auto& dataProvider) {
                if (FeaturesManager->UseShuffle() && dataProvider.MetaInfo.HasGroupId) {
                    const ui32 loadBalancingPermutationId = FromString<ui32>(GetEnv("CB_LOAD_BALANCE_PERMUTATION", "42"));
                    return GetPermutation(dataProvider, loadBalancingPermutationId);
                } else {
                    return GetPermutation(dataProvider, TDataPermutation::IdentityPermutationId());
                }
            };
            LearnDocPerDevicesSplit = MakeHolder<TDocParallelSplit>(*DataProvider, getPermutation(*DataProvider));

            if (TestDataProvider) {
                TestDocPerDevicesSplit = MakeHolder<TDocParallelSplit>(*TestDataProvider, getPermutation(*TestDataProvider));
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
        const NCB::TTrainingDataProvider* DataProvider = nullptr;
        const NCB::TTrainingDataProvider* TestDataProvider = nullptr;
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
