#pragma once

#include "non_zero_filter.h"
#include "samples_grouping_gpu.h"
#include "kernels.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/filter.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/scan.h>
#include <catboost/cuda/cuda_util/partitions.h>

namespace NCatboostCuda {
    inline void ComputeQueryOffsets(const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& origQids,
                                    const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& sampledDocs,
                                    TCudaBuffer<ui32, NCudaLib::TStripeMapping>* docQids,
                                    TCudaBuffer<ui32, NCudaLib::TStripeMapping>* queryOffsets) {
        Y_VERIFY(sampledDocs.GetObjectsSlice().Size());
        auto tempFlags = TCudaBuffer<ui32, NCudaLib::TStripeMapping>::CopyMapping(sampledDocs);
        docQids->Reset(sampledDocs.GetMapping());
        FillQueryEndMasks(origQids, sampledDocs, &tempFlags);
        ScanVector(tempFlags, *docQids, false);

        queryOffsets->Reset(CreateMappingFromTail(*docQids, /* tail is lastBinIdx, +1 for size, +1 for fake bin */ 2));
        UpdatePartitionOffsets(*docQids, *queryOffsets);
    }

    class TQuerywiseSampler {
    public:
        using TMapping = NCudaLib::TStripeMapping;

        TQuerywiseSampler() {
        }

        const TCudaBuffer<ui32, TMapping>& GetPerDocQids(const NCatboostCuda::TGpuSamplesGrouping<TMapping>& samplesGrouping) {
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
        };

        void SampleQueries(TGpuAwareRandom& random,
                           const double querywiseSampleRate,
                           const double docwiseSampleRate,
                           const ui32 maxQuerySize,
                           const NCatboostCuda::TGpuSamplesGrouping<TMapping>& samplesGrouping,
                           TCudaBuffer<ui32, TMapping>* sampledIndices) {
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

        void SampleQueries(TGpuAwareRandom& random,
                           const double querywiseSampleRate,
                           const double docwiseSampleRate,
                           const ui32 maxQuerySize,
                           const TCudaBuffer<ui32, TMapping>& qids,
                           const TCudaBuffer<const ui32, TMapping>& queryOffsets,
                           const NCudaLib::TDistributedObject<ui32>& offsetsBias,
                           const TCudaBuffer<const ui32, TMapping>& querySizes,
                           TCudaBuffer<ui32, TMapping>* sampledIndices) {
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

            FilterZeroEntries(&sampledWeight,
                              &indices);
            RadixSort(indices);
        }

    private:
        TScopedCacheHolder CacheHolder;
    };
}
