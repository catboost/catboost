#pragma once

#include "non_zero_filter.h"
#include "samples_grouping_gpu.h"
#include "kernels.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/libs/helpers/cpu_random.h>
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
    void ComputeQueryOffsets(const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& origQids,
                             const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& sampledDocs,
                             TCudaBuffer<ui32, NCudaLib::TStripeMapping>* docQids,
                             TCudaBuffer<ui32, NCudaLib::TStripeMapping>* queryOffsets);

    class TQuerywiseSampler {
    public:
        using TMapping = NCudaLib::TStripeMapping;

        TQuerywiseSampler() {
        }

        const TCudaBuffer<ui32, TMapping>& GetPerDocQids(const NCatboostCuda::TGpuSamplesGrouping<TMapping>& samplesGrouping);

        void SampleQueries(TGpuAwareRandom& random,
                           const double querywiseSampleRate,
                           const double docwiseSampleRate,
                           const ui32 maxQuerySize,
                           const NCatboostCuda::TGpuSamplesGrouping<TMapping>& samplesGrouping,
                           TCudaBuffer<ui32, TMapping>* sampledIndices);

        void SampleQueries(TGpuAwareRandom& random,
                           const double querywiseSampleRate,
                           const double docwiseSampleRate,
                           const ui32 maxQuerySize,
                           const TCudaBuffer<ui32, TMapping>& qids,
                           const TCudaBuffer<const ui32, TMapping>& queryOffsets,
                           const NCudaLib::TDistributedObject<ui32>& offsetsBias,
                           const TCudaBuffer<const ui32, TMapping>& querySizes,
                           TCudaBuffer<ui32, TMapping>* sampledIndices);

    private:
        TScopedCacheHolder CacheHolder;
    };
}
