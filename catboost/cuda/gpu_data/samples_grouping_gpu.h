#pragma once

#include "samples_grouping.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <util/system/types.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    //TODO(noxoomo): account to group sizes during splitting.
    template <class TMapping>
    class TGpuSamplesGrouping {
    public:
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

        ui32 GetQuerySize(ui32 localQueryId) const {
            return Grouping->GetQuerySize(GetQid(localQueryId));
        }

        ui32 GetQueryOffset(ui32 localQueryId) const {
            return Grouping->GetQueryOffset(GetQid(localQueryId)) - Grouping->GetQueryOffset(GetQid(0));
        }

        ui32 GetQueryCount() const {
            return Grouping->GetQueryId(CurrentDocsSlice.Right) - Grouping->GetQueryId(CurrentDocsSlice.Left);
        }

        const ui32* GetSubgroupIds(ui32 localQueryId) const {
            const auto queriesGrouping = dynamic_cast<const TQueriesGrouping*>(Grouping);
            CB_ENSURE(queriesGrouping && queriesGrouping->HasSubgroupIds());
            return queriesGrouping->GetSubgroupIds(GetQid(localQueryId));
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
            : OffsetBiases(NCudaLib::GetCudaManager().CreateDistributedObject<ui32>())
        {
        }

        ui32 GetQid(ui32 localQueryId) const {
            return Grouping->GetQueryOffset(CurrentDocsSlice.Left) + localQueryId;
        }

    private:
        friend TGpuSamplesGrouping<NCudaLib::TStripeMapping> MakeStripeGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& mirrorMapping,
                                                                                const TCudaBuffer<const ui32, NCudaLib::TStripeMapping>& indices);

        friend TGpuSamplesGrouping<NCudaLib::TMirrorMapping> SliceGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& grouping, const TSlice& localSlice);

        template <class TDataSet>
        friend TGpuSamplesGrouping<NCudaLib::TMirrorMapping> CreateGpuGrouping(const TDataSet& dataSet, const TSlice& slice);

        const IQueriesGrouping* Grouping;
        TSlice CurrentDocsSlice;

        NCudaLib::TCudaBuffer<const ui32, TMapping> Offsets;
        NCudaLib::TCudaBuffer<const ui32, TMapping> Sizes;
        NCudaLib::TDistributedObject<ui32> OffsetBiases;

        //if we have them
        NCudaLib::TCudaBuffer<const uint2, TMapping> Pairs;
        NCudaLib::TCudaBuffer<const float, TMapping> PairsWeights;
    };

    template <class TDataSet>
    inline TGpuSamplesGrouping<NCudaLib::TMirrorMapping> CreateGpuGrouping(const TDataSet& dataSet, const TSlice& slice) {
        const IQueriesGrouping& grouping = dataSet.GetSamplesGrouping();

        const ui32 rightSubgroupId = grouping.GetQueryId(slice.Right);
        const ui32 leftSubgroupId = grouping.GetQueryId(slice.Left);
        const ui32 groupCount = rightSubgroupId - leftSubgroupId;
        TVector<ui32> offsets(groupCount);
        TVector<ui32> sizes(groupCount);
        const ui32 offset = grouping.GetQueryOffset(leftSubgroupId);

        for (ui32 i = leftSubgroupId; i < rightSubgroupId; ++i) {
            offsets[i - leftSubgroupId] = grouping.GetQueryOffset(i) - offset;
            sizes[i - leftSubgroupId] = grouping.GetQuerySize(i);
        }
        TMirrorBuffer<ui32> offsetsGpu = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(offsets.size()));
        TMirrorBuffer<ui32> sizesGpu = TMirrorBuffer<ui32>::CopyMapping(offsetsGpu);
        offsetsGpu.Write(offsets);
        sizesGpu.Write(sizes);
        TGpuSamplesGrouping<NCudaLib::TMirrorMapping> samplesGrouping;

        samplesGrouping.Grouping = &grouping;
        samplesGrouping.CurrentDocsSlice = slice;
        samplesGrouping.Offsets = offsetsGpu.ConstCopyView();
        samplesGrouping.Sizes = sizesGpu.ConstCopyView();
        samplesGrouping.OffsetBiases = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>();

        {
            const TQueriesGrouping* pointwiseSamplesGrouping = dynamic_cast<const TQueriesGrouping*>(&grouping);
            if (pointwiseSamplesGrouping && pointwiseSamplesGrouping->GetFlatQueryPairs().size()) {
                const auto& pairs = pointwiseSamplesGrouping->GetFlatQueryPairs();
                const auto& pairsWeights = pointwiseSamplesGrouping->GetQueryPairWeights();

                auto pairsGpu = TMirrorBuffer<uint2>::Create(NCudaLib::TMirrorMapping(pairs.size()));
                auto pairsWeightsGpu = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(pairs.size()));

                pairsGpu.Write(pairs);
                pairsWeightsGpu.Write(pairsWeights);

                samplesGrouping.Pairs = pairsGpu.ConstCopyView();
                samplesGrouping.PairsWeights = pairsWeightsGpu.ConstCopyView();
            }
        }
        return samplesGrouping;
    }

    inline TGpuSamplesGrouping<NCudaLib::TMirrorMapping> SliceGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& grouping,
                                                                       const TSlice& localSlice) {
        CB_ENSURE(localSlice.Size() <= grouping.CurrentDocsSlice.Size());
        TSlice globalSlice;
        globalSlice.Left = localSlice.Left + grouping.CurrentDocsSlice.Left;
        globalSlice.Right = localSlice.Right + grouping.CurrentDocsSlice.Left;

        const ui32 firstSubgroupId = grouping.Grouping->GetQueryId(globalSlice.Left);
        const ui32 lastSubgroupId = grouping.Grouping->GetQueryId(globalSlice.Right);

        const ui32 firstGroupDoc = grouping.Grouping->GetQueryOffset(firstSubgroupId);
        const ui32 lastGroupDoc = grouping.Grouping->GetQueryOffset(lastSubgroupId);

        CB_ENSURE(firstGroupDoc == globalSlice.Left, "Error: slice should be group-consistent");
        CB_ENSURE(lastGroupDoc == globalSlice.Right, "Error: slice should be group-consistent");

        auto biases = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>();
        for (ui32 dev = 0; dev < biases.DeviceCount(); ++dev) {
            biases.Set(dev, firstGroupDoc);
        }

        TSlice groupsSlice;
        const ui32 groupOffset = grouping.Grouping->GetQueryId(grouping.CurrentDocsSlice.Left);
        CB_ENSURE(firstSubgroupId >= groupOffset);
        groupsSlice.Left = firstSubgroupId - groupOffset;
        groupsSlice.Right = lastSubgroupId - groupOffset;
        CB_ENSURE(grouping.Offsets.GetObjectsSlice() == TSlice(0, grouping.Grouping->GetQueryId(
                                                                      grouping.CurrentDocsSlice.Right) -
                                                                      groupOffset));

        TGpuSamplesGrouping<NCudaLib::TMirrorMapping> sliceGrouping(grouping.Grouping,
                                                                    globalSlice,
                                                                    grouping.Offsets.SliceView(groupsSlice),
                                                                    grouping.Sizes.SliceView(groupsSlice),
                                                                    std::move(biases));
        {
            const TQueriesGrouping* pointwiseSamplesGrouping = dynamic_cast<const TQueriesGrouping*>(grouping.Grouping);
            if (pointwiseSamplesGrouping && pointwiseSamplesGrouping->GetFlatQueryPairs().size()) {
                TSlice pairsSlice;
                CB_ENSURE(firstSubgroupId >= groupOffset);
                const ui32 shift = pointwiseSamplesGrouping->GetQueryPairOffset(groupOffset);
                pairsSlice.Left = pointwiseSamplesGrouping->GetQueryPairOffset(firstSubgroupId) - shift;
                pairsSlice.Right = pointwiseSamplesGrouping->GetQueryPairOffset(lastSubgroupId) - shift;

                sliceGrouping.Pairs = grouping.Pairs.SliceView(pairsSlice);
                sliceGrouping.PairsWeights = grouping.PairsWeights.SliceView(pairsSlice);
            }
        }

        return sliceGrouping;
    }

    inline TGpuSamplesGrouping<NCudaLib::TStripeMapping> MakeStripeGrouping(const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& mirrorMapping,
                                                                            const TCudaBuffer<const ui32, NCudaLib::TStripeMapping>& indices) {
        CB_ENSURE(indices.GetObjectsSlice() == mirrorMapping.CurrentDocsSlice);
        TSlice docsSlice = indices.GetObjectsSlice();
        docsSlice.Left += mirrorMapping.CurrentDocsSlice.Left;
        docsSlice.Right += mirrorMapping.CurrentDocsSlice.Left;
        const IQueriesGrouping& grouping = *mirrorMapping.Grouping;
        const ui32 queryIdOffset = grouping.GetQueryId(docsSlice.Left);

        const NCudaLib::TDistributedObject<ui32>& baseBias = mirrorMapping.GetOffsetsBias();
        NCudaLib::TDistributedObject<ui32> deviceOffsetsBiases = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>();

        TVector<TSlice> groupMetaSlices(deviceOffsetsBiases.DeviceCount());

        ui32 offset = 0;
        for (ui32 dev = 0; dev < deviceOffsetsBiases.DeviceCount(); ++dev) {
            TSlice deviceSlice = indices.GetMapping().DeviceSlice(dev);
            const ui32 localBias = baseBias.At(dev) + offset;
            deviceOffsetsBiases.Set(dev, localBias);
            ui32 firstSubgroupId = grouping.GetQueryId(docsSlice.Left + offset) - queryIdOffset;
            ui32 lastSubgroupId = grouping.GetQueryId(docsSlice.Left + offset + deviceSlice.Size()) - queryIdOffset;
            offset += deviceSlice.Size();
            groupMetaSlices[dev] = TSlice(firstSubgroupId, lastSubgroupId);
        }

        NCudaLib::TStripeMapping groupMetaMapping(std::move(groupMetaSlices));
        CB_ENSURE(groupMetaMapping.GetObjectsSlice() == TSlice(0, grouping.GetQueryId(docsSlice.Right) - queryIdOffset));
        NCudaLib::TCudaBuffer<const ui32, NCudaLib::TStripeMapping> offsets = NCudaLib::StripeView(mirrorMapping.Offsets, groupMetaMapping);
        NCudaLib::TCudaBuffer<const ui32, NCudaLib::TStripeMapping> sizes = NCudaLib::StripeView(mirrorMapping.Sizes, groupMetaMapping);

        TGpuSamplesGrouping<NCudaLib::TStripeMapping> samplesGrouping(&grouping,
                                                                      mirrorMapping.CurrentDocsSlice,
                                                                      std::move(offsets),
                                                                      std::move(sizes),
                                                                      std::move(deviceOffsetsBiases));

        {
            const TQueriesGrouping* pointwiseSamplesGrouping = dynamic_cast<const TQueriesGrouping*>(&grouping);
            if (pointwiseSamplesGrouping && pointwiseSamplesGrouping->GetFlatQueryPairs().size()) {
                NCudaLib::TStripeMapping pairsMapping = groupMetaMapping.Transform([&](const TSlice& groupsSlice) -> ui64 {
                    ui32 firstSubgroupId = groupsSlice.Left + queryIdOffset;
                    ui32 lastSubgroupId = groupsSlice.Right + queryIdOffset;
                    return pointwiseSamplesGrouping->GetQueryPairOffset(lastSubgroupId) - pointwiseSamplesGrouping->GetQueryPairOffset(firstSubgroupId);
                });
                samplesGrouping.Pairs = NCudaLib::StripeView(mirrorMapping.Pairs, pairsMapping);
                samplesGrouping.PairsWeights = NCudaLib::StripeView(mirrorMapping.PairsWeights, pairsMapping);
            }
        }
        return samplesGrouping;
    }
}
