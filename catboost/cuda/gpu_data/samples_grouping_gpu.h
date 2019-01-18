#pragma once

#include "samples_grouping.h"
#include "feature_parallel_dataset.h"
#include "doc_parallel_dataset.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <util/system/types.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    //TODO(noxoomo): account to group sizes during splitting.
    //TODO(noxoomo): write only gids on GPU (and gids pairs/pairSize/pairCount), all index should be made on GPU for efficien sampling scheme support
    template <class TMapping>
    class TGpuSamplesGrouping: public TGuidHolder {
    public:
        /*
         * Biased offset are query offset in single mapping
         * offsets bias if first query offset on device
         * query size are just size
         * introduction of offsetsBias allows to reuse mirror-buffers in stripe leaves estimation
         * Pairs are also global index
         */
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& GetBiasedOffsets() const {
            return Offsets;
        };

        const NCudaLib::TDistributedObject<ui32>& GetOffsetsBias() const {
            return OffsetBiases;
        }

        const NCudaLib::TCudaBuffer<const ui32, TMapping>& GetSizes() const {
            return Sizes;
        };

        const NCudaLib::TCudaBuffer<const uint2, TMapping>& GetPairs() const {
            return Pairs;
        };

        const NCudaLib::TCudaBuffer<const float, TMapping>& GetPairsWeights() const {
            return PairsWeights;
        };

        //TODO(noxoomo): get rid of this, we need sampling for this structure without CPU structs rebuilding
        ui32 GetQuerySize(ui32 localQueryId) const {
            return Grouping->GetQuerySize(GetQid(localQueryId));
        }

        ui32 GetQueryOffset(ui32 localQueryId) const {
            return Grouping->GetQueryOffset(GetQid(localQueryId)) - Grouping->GetQueryOffset(GetQid(0));
        }

        ui32 GetQueryCount() const {
            return Grouping->GetQueryId(CurrentDocsSlice.Right) - Grouping->GetQueryId(CurrentDocsSlice.Left);
        }

        bool HasSubgroupIds() const {
            const auto queriesGrouping = dynamic_cast<const TQueriesGrouping*>(Grouping);
            return queriesGrouping && queriesGrouping->HasSubgroupIds();
        }

        bool HasPairs() const {
            return Pairs.GetObjectsSlice().Size();
        }

        TVector<TVector<TCompetitor>> CreateQueryCompetitors(ui32 localQid) const {
            const auto queriesGrouping = dynamic_cast<const TQueriesGrouping*>(Grouping);
            CB_ENSURE(queriesGrouping && queriesGrouping->GetFlatQueryPairs().size());
            const ui32 querySize = GetQuerySize(localQid);

            TVector<TVector<TCompetitor>> competitors(querySize);
            const uint2* pairs = queriesGrouping->GetFlatQueryPairs().data();
            const float* pairWeights = queriesGrouping->GetQueryPairWeights().data();
            const ui32 queryOffset = Grouping->GetQueryOffset(GetQid(localQid));

            const ui32 firstPair = GetQueryPairOffset(localQid);
            const ui32 lastPair = GetQueryPairOffset(localQid + 1);

            for (ui32 pairId = firstPair; pairId < lastPair; ++pairId) {
                uint2 pair = pairs[pairId];
                TCompetitor competitor(pair.y - queryOffset,
                                       pairWeights[pairId]);
                competitor.SampleWeight = 0;
                competitors[pair.x - queryOffset].push_back(competitor);
            }
            return competitors;
        };

        const ui32* GetSubgroupIds(ui32 localQueryId) const {
            const auto queriesGrouping = dynamic_cast<const TQueriesGrouping*>(Grouping);
            CB_ENSURE(queriesGrouping && queriesGrouping->HasSubgroupIds());
            return queriesGrouping->GetSubgroupIds(GetQid(localQueryId));
        }

        TGpuSamplesGrouping CopyView() const {
            TGpuSamplesGrouping copy;
            copy.Grouping = Grouping;
            copy.CurrentDocsSlice = CurrentDocsSlice;
            copy.Offsets = Offsets.ConstCopyView();
            copy.Sizes = Sizes.ConstCopyView();
            copy.OffsetBiases = OffsetBiases;
            if (Pairs.GetObjectsSlice().Size()) {
                copy.Pairs = Pairs.ConstCopyView();
                copy.PairsWeights = PairsWeights.ConstCopyView();
            }
            return copy;
        }

        const IQueriesGrouping& GetOwner() const {
            return *Grouping;
        }

    protected:
        TGpuSamplesGrouping(const IQueriesGrouping* owner,
                            TSlice docSlices,
                            NCudaLib::TCudaBuffer<const ui32, TMapping>&& offsets,
                            NCudaLib::TCudaBuffer<const ui32, TMapping>&& sizes,
                            NCudaLib::TDistributedObject<ui32>&& biases)
            : Grouping(owner)
            , CurrentDocsSlice(docSlices)
            , Offsets(std::move(offsets))
            , Sizes(std::move(sizes))
            , OffsetBiases(std::move(biases))
        {
        }

        TGpuSamplesGrouping()
            : OffsetBiases(NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(0))
        {
        }

        ui32 GetQid(ui32 localQueryId) const {
            return Grouping->GetQueryOffset(CurrentDocsSlice.Left) + localQueryId;
        }

        ui32 GetQueryPairOffset(ui32 localQueryId) const {
            if (dynamic_cast<const TQueriesGrouping*>(Grouping) != nullptr) {
                return dynamic_cast<const TQueriesGrouping*>(Grouping)->GetQueryPairOffset(GetQid(localQueryId));
            } else {
                CB_ENSURE(false, "Error: don't have pairs thus pairwise metrics/learning can't be used");
            }
        }

    private:
        template <class>
        friend class TGpuSamplesGroupingHelper;

        const IQueriesGrouping* Grouping;
        TSlice CurrentDocsSlice;

        NCudaLib::TCudaBuffer<const ui32, TMapping> Offsets;
        NCudaLib::TCudaBuffer<const ui32, TMapping> Sizes;
        NCudaLib::TDistributedObject<ui32> OffsetBiases;

        //if we have them
        NCudaLib::TCudaBuffer<const uint2, TMapping> Pairs;
        NCudaLib::TCudaBuffer<const float, TMapping> PairsWeights;
    };

    template <class>
    class TGpuSamplesGroupingHelper;

    template <>
    class TGpuSamplesGroupingHelper<NCudaLib::TMirrorMapping> {
    public:
        static TGpuSamplesGrouping<NCudaLib::TMirrorMapping> CreateGpuGrouping(const TFeatureParallelDataSet& dataSet,
                                                                               const TSlice& slice);

        static TGpuSamplesGrouping<NCudaLib::TMirrorMapping> SliceGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& grouping,
                                                                           const TSlice& localSlice);

        static TGpuSamplesGrouping<NCudaLib::TStripeMapping> MakeStripeGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& mirrorMapping,
                                                                                const TCudaBuffer<const ui32, NCudaLib::TStripeMapping>& indices);
    };

    template <>
    class TGpuSamplesGroupingHelper<NCudaLib::TStripeMapping> {
    public:
        static TGpuSamplesGrouping<NCudaLib::TStripeMapping> CreateGpuGrouping(const TDocParallelDataSet& dataSet);
    };

    extern template class TGpuSamplesGroupingHelper<NCudaLib::TStripeMapping>;
    extern template class TGpuSamplesGroupingHelper<NCudaLib::TMirrorMapping>;

    template <class TDataSet>
    inline TGpuSamplesGrouping<NCudaLib::TStripeMapping> CreateGpuGrouping(const TDataSet& dataSet) {
        return TGpuSamplesGroupingHelper<NCudaLib::TStripeMapping>::CreateGpuGrouping(dataSet);
    }

    template <class TDataSet>
    inline TGpuSamplesGrouping<NCudaLib::TMirrorMapping> CreateGpuGrouping(const TDataSet& dataSet,
                                                                           const TSlice& slice) {
        return TGpuSamplesGroupingHelper<NCudaLib::TMirrorMapping>::CreateGpuGrouping(dataSet, slice);
    }

    inline TGpuSamplesGrouping<NCudaLib::TMirrorMapping> SliceGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& grouping,
                                                                       const TSlice& localSlice) {
        return TGpuSamplesGroupingHelper<NCudaLib::TMirrorMapping>::SliceGrouping(grouping, localSlice);
    }

    inline TGpuSamplesGrouping<NCudaLib::TStripeMapping> MakeStripeGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& mirrorMapping,
                                                                            const TCudaBuffer<const ui32, NCudaLib::TStripeMapping>& indices) {
        return TGpuSamplesGroupingHelper<NCudaLib::TMirrorMapping>::MakeStripeGrouping(mirrorMapping, indices);
    }
}
