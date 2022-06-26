#include "querywise_helper.h"

void NCatboostCuda::TQuerywiseSampler::SampleQueries(TGpuAwareRandom& random, const double querywiseSampleRate,
                                                     const double docwiseSampleRate, const ui32 maxQuerySize,
                                                     const TCudaBuffer<ui32, NCatboostCuda::TQuerywiseSampler::TMapping>& qids,
                                                     const TCudaBuffer<const ui32, NCatboostCuda::TQuerywiseSampler::TMapping>& queryOffsets,
                                                     const NCudaLib::TDistributedObject<ui32>& offsetsBias,
                                                     const TCudaBuffer<const ui32, NCatboostCuda::TQuerywiseSampler::TMapping>& querySizes,
                                                     TCudaBuffer<ui32, NCatboostCuda::TQuerywiseSampler::TMapping>* sampledIndices) {
    auto& seeds = random.GetGpuSeeds<TMapping>();
    auto& indices = *sampledIndices;
    indices.Reset(qids.GetMapping());

    auto sampledWeight = TCudaBuffer<float, TMapping>::CopyMapping(qids);
    MakeSequence(indices);

    auto takenQueryMasks = TCudaBuffer<float, TMapping>::CopyMapping(queryOffsets);
    FillBuffer(takenQueryMasks, 1.0f);
    if (querywiseSampleRate < 1.0) {
        UniformBootstrap(seeds, takenQueryMasks, querywiseSampleRate);
    }

    auto keys = TCudaBuffer<ui64, TMapping>::CopyMapping(indices);
    CreateShuffleKeys(seeds, qids, &keys);
    RadixSort(keys,
              indices);

    CreateTakenDocsMask(takenQueryMasks,
                        qids,
                        indices,
                        queryOffsets,
                        offsetsBias,
                        querySizes,
                        docwiseSampleRate,
                        maxQuerySize,
                        &sampledWeight);

    {
        auto nzElements = TCudaBuffer<ui32, TMapping>::CopyMapping(indices);
        auto docs = TCudaBuffer<ui32, TMapping>::CopyMapping(indices);
        docs.Copy(indices);

        FilterZeroEntries(&sampledWeight,
                          &nzElements);

        indices.Reset(sampledWeight.GetMapping());
        Gather(indices, docs, nzElements);
    }
    RadixSort(indices);
}

const TCudaBuffer<ui32, NCatboostCuda::TQuerywiseSampler::TMapping>& NCatboostCuda::TQuerywiseSampler::GetPerDocQids(const NCatboostCuda::TGpuSamplesGrouping<NCatboostCuda::TQuerywiseSampler::TMapping>& samplesGrouping) {
    return CacheHolder.Cache(samplesGrouping, 0, [&]() -> TCudaBuffer<ui32, TMapping> {
        auto docsMapping = samplesGrouping.GetSizes().GetMapping().Transform([&](const TSlice& queriesSlice) -> ui64 {
            return samplesGrouping.GetQueryOffset(queriesSlice.Right) - samplesGrouping.GetQueryOffset(queriesSlice.Left);
        });
        auto ids = TCudaBuffer<ui32, TMapping>::Create(docsMapping);
        ComputeQueryIds(samplesGrouping.GetSizes(),
                        samplesGrouping.GetBiasedOffsets(),
                        samplesGrouping.GetOffsetsBias(),
                        &ids);
        return ids;
    });
}

void NCatboostCuda::TQuerywiseSampler::SampleQueries(TGpuAwareRandom& random, const double querywiseSampleRate,
                                                     const double docwiseSampleRate, const ui32 maxQuerySize,
                                                     const NCatboostCuda::TGpuSamplesGrouping<NCatboostCuda::TQuerywiseSampler::TMapping>& samplesGrouping,
                                                     TCudaBuffer<ui32, NCatboostCuda::TQuerywiseSampler::TMapping>* sampledIndices) {
    const TCudaBuffer<ui32, TMapping>& qids = GetPerDocQids(samplesGrouping);

    SampleQueries(random,
                  querywiseSampleRate,
                  docwiseSampleRate,
                  maxQuerySize,
                  qids,
                  samplesGrouping.GetBiasedOffsets(),
                  samplesGrouping.GetOffsetsBias(),
                  samplesGrouping.GetSizes(),
                  sampledIndices);
}

void NCatboostCuda::ComputeQueryOffsets(const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& origQids,
                                        const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& sampledDocs,
                                        TCudaBuffer<ui32, NCudaLib::TStripeMapping>* docQids,
                                        TCudaBuffer<ui32, NCudaLib::TStripeMapping>* queryOffsets) {
    CB_ENSURE(sampledDocs.GetObjectsSlice().Size(), "Object slice is empty");
    auto tempFlags = TCudaBuffer<ui32, NCudaLib::TStripeMapping>::CopyMapping(sampledDocs);
    docQids->Reset(sampledDocs.GetMapping());
    FillQueryEndMasks(origQids, sampledDocs, &tempFlags);
    ScanVector(tempFlags, *docQids, false);

    queryOffsets->Reset(CreateMappingFromTail(*docQids, /* tail is lastBinIdx, +1 for size, +1 for fake bin */ 2));
    UpdatePartitionOffsets(*docQids, *queryOffsets);
}
