#pragma once

#include "cat_features_dataset.h"
#include "samples_grouping.h"
#include "compressed_index.h"

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/data/permutation.h>

#include <catboost/libs/data/data_provider.h>

namespace NCatboostCuda {
    template <class TSamplesMapping>
    class TTarget {
    public:
        explicit TTarget(NCudaLib::TCudaBuffer<const float, TSamplesMapping>&& targets,
                         NCudaLib::TCudaBuffer<const float, TSamplesMapping>&& weights,
                         bool isPairWeights)
            : Targets(std::move(targets))
            , Weights(std::move(weights))
            , HasIndicesFlag(false)
            , IndicesOffsets(CreateDistributedObject<ui32>(0))
            , StorePairWeights(isPairWeights)
        {
        }

        explicit TTarget(NCudaLib::TCudaBuffer<const float, TSamplesMapping>&& targets,
                         NCudaLib::TCudaBuffer<const float, TSamplesMapping>&& weights,
                         NCudaLib::TCudaBuffer<const ui32, TSamplesMapping>&& indices,
                         bool isPairWeights)
            : Targets(std::move(targets))
            , Weights(std::move(weights))
            , Indices(std::move(indices))
            , HasIndicesFlag(true)
            , IndicesOffsets(CreateDistributedObject<ui32>(0))
            , StorePairWeights(isPairWeights)
        {
        }

        TTarget(const TTarget& other)
            : Targets(other.Targets.ConstCopyView())
            , Weights(other.Weights.ConstCopyView())
            , IndicesOffsets(CreateDistributedObject<ui32>(0))
            , StorePairWeights(other.StorePairWeights)
        {
            if (other.HasIndicesFlag) {
                HasIndicesFlag = true;
                Indices = other.Indices.ConstCopyView();
            } else {
                IndicesOffsets = other.IndicesOffsets;
                HasIndicesFlag = false;
            }
        }

        TTarget(TTarget&& other) = default;

        void WriteIndices(NCudaLib::TCudaBuffer<ui32, TSamplesMapping>& dst,
                          ui32 stream = 0) const {
            if (HasIndicesFlag) {
                CB_ENSURE(dst.GetObjectsSlice() == Targets.GetObjectsSlice());
                dst.Copy(Indices, stream);
            } else {
                dst.Reset(Targets.GetMapping());
                MakeSequenceWithOffset(dst,
                                       IndicesOffsets,
                                       stream);
            }
        }

        bool HasIndices() const {
            return HasIndicesFlag;
        }

        const TCudaBuffer<const float, TSamplesMapping>& GetTargets() const {
            return Targets;
        };

        const TCudaBuffer<const float, TSamplesMapping>& GetWeights() const {
            return Weights;
        };

        bool HasPairWeights() const {
            return StorePairWeights;
        }

        const TCudaBuffer<const ui32, TSamplesMapping>& GetIndices() const {
            CB_ENSURE(HasIndicesFlag);
            return Indices;
        };

        const TSamplesMapping& GetSamplesMapping() const {
            return Targets.GetMapping();
        }

    private:
        template <class>
        friend class TTargetHelper;

        template <class>
        friend class TQuerywiseTargetsImpl;

    private:
        TCudaBuffer<const float, TSamplesMapping> Targets;
        TCudaBuffer<const float, TSamplesMapping> Weights;
        TCudaBuffer<const ui32, TSamplesMapping> Indices;

        bool HasIndicesFlag = false;
        NCudaLib::TDistributedObject<ui32> IndicesOffsets;
        bool StorePairWeights;
    };

    template <class TMapping>
    class TTargetHelper {
    public:
        static TTarget<TMapping> Slice(const TTarget<TMapping>&,
                                       const TSlice&) {
            CB_ENSURE(false);
        }

        static TTarget<NCudaLib::TStripeMapping> StripeView(const TTarget<TMapping>&,
                                                            const TMapping&) {
            CB_ENSURE(false);
        }
    };

    template <>
    class TTargetHelper<NCudaLib::TMirrorMapping> {
    public:
        static TTarget<NCudaLib::TMirrorMapping> Slice(const TTarget<NCudaLib::TMirrorMapping>& target,
                                                       const TSlice& slice) {
            if (target.HasIndicesFlag) {
                return TTarget<NCudaLib::TMirrorMapping>(target.Targets.SliceView(slice),
                                                         target.Weights.SliceView(slice),
                                                         target.Indices.SliceView(slice),
                                                         target.StorePairWeights);
            } else {
                auto result = TTarget<NCudaLib::TMirrorMapping>(target.Targets.SliceView(slice),
                                                                target.Weights.SliceView(slice),
                                                                target.StorePairWeights);

                auto offsets = CreateDistributedObject<ui32>(0u);
                for (ui32 dev = 0; dev < target.IndicesOffsets.DeviceCount(); ++dev) {
                    ui32 devOffset = slice.Left + target.IndicesOffsets.At(dev);
                    offsets.Set(dev, devOffset);
                }
                result.IndicesOffsets = offsets;
                return result;
            }
        }

        static TTarget<NCudaLib::TStripeMapping> StripeView(const TTarget<NCudaLib::TMirrorMapping>& target,
                                                            const NCudaLib::TStripeMapping& stripeMapping) {
            if (target.HasIndicesFlag) {
                return TTarget<NCudaLib::TStripeMapping>(NCudaLib::StripeView(target.Targets, stripeMapping),
                                                         NCudaLib::StripeView(target.Weights, stripeMapping),
                                                         NCudaLib::StripeView(target.Indices, stripeMapping),
                                                         target.StorePairWeights);
            } else {
                auto result = TTarget<NCudaLib::TStripeMapping>(NCudaLib::StripeView(target.Targets, stripeMapping),
                                                                NCudaLib::StripeView(target.Weights, stripeMapping),
                                                                target.StorePairWeights);

                auto offsets = CreateDistributedObject<ui32>(0u);
                for (ui32 dev = 0; dev < target.IndicesOffsets.DeviceCount(); ++dev) {
                    TSlice devSlice = stripeMapping.DeviceSlice(dev);
                    ui32 devOffset = devSlice.Left + target.IndicesOffsets.At(dev);
                    offsets.Set(dev, devOffset);
                }
                result.IndicesOffsets = offsets;
                result.HasIndicesFlag = true;
                return result;
            }
        }
    };

    template <class TMapping>
    inline TTarget<TMapping> SliceTarget(const TTarget<TMapping>& target, const TSlice slice) {
        return TTargetHelper<TMapping>::Slice(target, slice);
    }

    template <class TLayout>
    class TDataSetBase {
    public:
        using TSamplesMapping = typename TLayout::TSamplesMapping;
        using TCompressedIndex = TSharedCompressedIndex<TLayout>;
        using TCompressedDataSet = typename TCompressedIndex::TCompressedDataSet;

        bool HasFeatures() const {
            return PermutationIndependentFeatures != static_cast<ui32>(-1);
        }

        bool HasPermutationDependentFeatures() const {
            return PermutationDependentFeatures != static_cast<ui32>(-1);
        }

        const TCompressedDataSet& GetFeatures() const {
            CB_ENSURE(HasFeatures());
            return CompressedIndex->GetDataSet(PermutationIndependentFeatures);
        }

        const TCompressedIndex& GetCompressedIndex() const {
            CB_ENSURE(CompressedIndex);
            return *CompressedIndex;
        }

        const TCompressedDataSet& GetPermutationFeatures() const {
            CB_ENSURE(HasPermutationDependentFeatures());
            return CompressedIndex->GetDataSet(PermutationDependentFeatures);
        }

        bool HasFeature(ui32 featureId) const {
            if (HasFeatures() && GetFeatures().HasFeature(featureId)) {
                return true;
            } else if (HasPermutationDependentFeatures()) {
                return GetPermutationFeatures().HasFeature(featureId);
            } else {
                return false;
            }
        }

        const NCudaLib::TDistributedObject<TCFeature>& GetTCFeature(ui32 featureId) const {
            CB_ENSURE(HasFeature(featureId));
            if (HasFeatures() && GetFeatures().HasFeature(featureId)) {
                return GetFeatures().GetTCFeature(featureId);
            } else if (HasPermutationDependentFeatures()) {
                return GetPermutationFeatures().GetTCFeature(featureId);
            } else {
                CB_ENSURE(false);
            }
            Y_UNREACHABLE();
        }

        bool IsOneHot(ui32 featureId) const {
            CB_ENSURE(HasFeature(featureId));
            if (HasFeatures() && GetFeatures().HasFeature(featureId)) {
                return GetFeatures().IsOneHot(featureId);
            } else if (HasPermutationDependentFeatures()) {
                return GetPermutationFeatures().IsOneHot(featureId);
            } else {
                CB_ENSURE(false);
            }
            Y_UNREACHABLE();
        }

        const NCB::TTrainingDataProvider& GetDataProvider() const {
            return DataProvider;
        }

        const TDataPermutation& GetCtrsEstimationPermutation() const {
            return CtrsEstimationPermutation;
        };

        const TTarget<TSamplesMapping>& GetTarget() const {
            return TargetsBuffer;
        }

        const TSamplesMapping& GetSamplesMapping() const {
            return TargetsBuffer.GetTargets().GetMapping();
        }

        TDataSetBase(const NCB::TTrainingDataProvider& dataProvider,
                     TAtomicSharedPtr<TCompressedIndex> compressedIndex,
                     const TDataPermutation& ctrPermutation,
                     TTarget<TSamplesMapping>&& target)
            : TargetsBuffer(std::move(target))
            , DataProvider(dataProvider)
            , CompressedIndex(compressedIndex)
            , CtrsEstimationPermutation(ctrPermutation)
        {
        }

    private:
        //direct indexing. we will gather them anyway in gradient descent and tree searching,
        // so let's save some memory
        TTarget<TSamplesMapping> TargetsBuffer;

        const NCB::TTrainingDataProvider& DataProvider;
        TAtomicSharedPtr<TCompressedIndex> CompressedIndex;
        TDataPermutation CtrsEstimationPermutation;

        ui32 PermutationDependentFeatures = -1;
        ui32 PermutationIndependentFeatures = -1;

        friend class TFeatureParallelDataSetHoldersBuilder;
        friend class TDocParallelDataSetBuilder;
    };

}
