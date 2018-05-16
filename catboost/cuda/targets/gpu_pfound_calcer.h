#pragma once

#include "quality_metric_helpers.h"
#include <catboost/libs/metrics/pfound.h>
#include <catboost/cuda/gpu_data/samples_grouping_gpu.h>

namespace NCatboostCuda {
    template <class TMapping>
    class TGpuPFoundCalcer {
    public:
        using TConstVec = TCudaBuffer<const float, TMapping>;
        TGpuPFoundCalcer(TCudaBuffer<const float, TMapping>&& target,
                         const TGpuSamplesGrouping<TMapping>& samplesGrouping)
            : Target(std::move(target))
            , SamplesGrouping(samplesGrouping)
        {
        }

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            TVector<float> pointCpu;
            point.Read(pointCpu);

            if (TargetCpu.size() == 0) {
                Target.Read(TargetCpu);
                SamplesGrouping.GetBiasedOffsets().Read(QueryOffsets);
                SamplesGrouping.GetSizes().Read(QuerySizes);
            }

            const ui32 queryCount = SamplesGrouping.GetQueryCount();

            auto& executor = NPar::LocalExecutor();
            const ui32 threadCount = 1 + executor.GetThreadCount();

            TVector<TPFoundCalcer> calcers(threadCount);
            const ui32 queriesPerThread = (queryCount + threadCount - 1) / threadCount;

            NPar::ParallelFor(executor, 0, threadCount, [&](ui32 tid) {
                auto& localCalcer = calcers[tid];
                const ui32 firstQuery = tid * queriesPerThread;
                const ui32 lastQuery = Min<ui32>((tid + 1) * queriesPerThread, queryCount);
                for (ui32 query = firstQuery; query < lastQuery; ++query) {
                    ui32 offset = SamplesGrouping.GetQueryOffset(query);
                    ui32 querySize = SamplesGrouping.GetQuerySize(query);
                    const ui32* subgroupIds = SamplesGrouping.GetSubgroupIds(query);
                    localCalcer.AddQuery(~TargetCpu + offset, ~pointCpu + offset, subgroupIds, querySize);
                }
            });
            TMetricHolder result;
            for (auto& localCalcer : calcers) {
                result.Add(localCalcer.GetMetric());
            }
            return TAdditiveStatistic(result.Stats[0], result.Stats[1]);
        }

    private:
        TCudaBuffer<const float, TMapping> Target;
        const TGpuSamplesGrouping<TMapping>& SamplesGrouping;
        mutable TVector<float> TargetCpu;
        mutable TVector<ui32> QueryOffsets;
        mutable TVector<ui32> QuerySizes;
    };
}
