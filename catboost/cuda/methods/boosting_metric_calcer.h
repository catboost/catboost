#pragma once

#include "approx_helper.h"
#include <catboost/libs/metrics/metric.h>
#include <catboost/cuda/targets/gpu_metrics.h>

#include <library/cpp/threading/local_executor/local_executor.h>

namespace NCatboostCuda {
    class IMetricCalcer {
    public:
        virtual ~IMetricCalcer() {
        }

        virtual TMetricHolder Compute(const IGpuMetric* metric) = 0;
    };

    template <class TTarget>
    class TMetricCalcer: public IMetricCalcer {
    public:
        using TTargetMapping = typename TTarget::TMapping;
        using TConstVec = typename TTarget::TConstVec;

        TMetricCalcer(const TTarget& target, NPar::ILocalExecutor* localExecutor)
            : Target(target)
            , LocalExecutor(localExecutor)
        {
        }

        void SetPoint(TConstVec&& point) {
            Point = std::move(point);
            PointOnCpuCached = false;
            Cache = MakeHolder<TScopedCacheHolder>();
        }

        TMetricHolder Compute(const IGpuMetric* metric) final {
            CB_ENSURE(Point.GetObjectsSlice().Size(), "Set point first");
            auto targets = Target.GetTarget().GetTargets().ConstCopyView();
            TConstVec weights;
            if ((metric->GetCpuMetric().UseWeights.IsIgnored() || metric->GetCpuMetric().UseWeights) &&
                !Target.GetTarget().HasPairWeights()
            ) {
                weights = Target.GetTarget().GetWeights().ConstCopyView();
            } else {
                using TVec = typename TTarget::TVec;
                auto tmp = TVec::CopyMapping(targets);
                FillBuffer(tmp, 1.0f);
                weights = tmp.ConstCopyView();
            }

            if (dynamic_cast<const IGpuPointwiseMetric*>(metric)) {
                return dynamic_cast<const IGpuPointwiseMetric*>(metric)->Eval(targets,
                                                                              weights,
                                                                              Point,
                                                                              Cache.Get());

            } else if (dynamic_cast<const IGpuQuerywiseMetric*>(metric)) {
                return dynamic_cast<const IGpuQuerywiseMetric*>(metric)->Eval(targets,
                                                                              weights,
                                                                              Target.GetSamplesGrouping(),
                                                                              Point);

            } else if (dynamic_cast<const TTargetFallbackMetric*>(metric)) {
                return dynamic_cast<const TTargetFallbackMetric*>(metric)->Eval(Target,
                                                                                Point);

            } else if (dynamic_cast<const TCpuFallbackMetric*>(metric)) {
                CachePointOnCpu();
                CacheCpuTargetAndWeight();
                if (metric->GetCpuMetric().GetErrorType() != EErrorType::PerObjectError) {
                    CacheQueryInfo(Target.GetSamplesGrouping());
                }
                return dynamic_cast<const TCpuFallbackMetric*>(metric)->Eval(PointOnCpu,
                                                                             CpuTarget,
                                                                             CpuWeights,
                                                                             QueryInfo,
                                                                             LocalExecutor);
            } else if (dynamic_cast<const TGpuCustomMetric*>(metric)) {
                return dynamic_cast<const TGpuCustomMetric*>(metric)->Eval(targets,
                                                                            weights,
                                                                            Point,
                                                                            Cache.Get());
            } else {
                CB_ENSURE(false, "Can't compute metric " << metric->GetCpuMetric().GetDescription() << " during GPU learning");
            }
        }

    private:
        void CachePointOnCpu() {
            if (!PointOnCpuCached) {
                const bool removeLastApproxOptimization = Target.GetType() == ELossFunction::MultiClass;
                ReadApproxInCpuFormat(Point, removeLastApproxOptimization, &PointOnCpu);
                PointOnCpuCached = true;
            }
        }

        void CacheCpuTargetAndWeight() {
            if (CpuTarget.size() == 0) {
                Target.GetTarget().GetTargets().Read(CpuTarget);
            }
            if (CpuWeights.size() == 0) {
                if (!Target.GetTarget().HasPairWeights()) {
                    Target.GetTarget().GetWeights().Read(CpuWeights);
                } else {
                    CpuWeights.resize(CpuTarget.size(), 1);
                }
            }
        }

        void CacheQueryInfo(const TGpuSamplesGrouping<TTargetMapping>& samplesGrouping) {
            if (QueryInfo.size() == 0) {
                CacheCpuTargetAndWeight();

                const ui32 qidCount = samplesGrouping.GetQueryCount();
                ui32 cursor = 0;

                for (ui32 qid = 0; qid < qidCount; ++qid) {
                    const ui32 querySize = samplesGrouping.GetQuerySize(qid);
                    TQueryInfo queryInfo;
                    queryInfo.Begin = cursor;
                    queryInfo.End = cursor + querySize;
                    queryInfo.Weight = CpuWeights[cursor];

                    if (samplesGrouping.HasSubgroupIds()) {
                        const ui32* subgroupIds = samplesGrouping.GetSubgroupIds(qid);
                        for (ui32 i = 0; i < querySize; ++i) {
                            queryInfo.SubgroupId.push_back(subgroupIds[i]);
                        }
                    }

                    if (samplesGrouping.HasPairs()) {
                        queryInfo.Competitors = samplesGrouping.CreateQueryCompetitors(qid);
                    }
                    QueryInfo.push_back(queryInfo);
                    cursor += querySize;
                }
            }
        }

    private:
        const TTarget& Target;
        TConstVec Point;
        TVector<TVector<double>> PointOnCpu;
        bool PointOnCpuCached = false;
        THolder<TScopedCacheHolder> Cache;

        TVector<float> CpuTarget;
        TVector<float> CpuWeights;
        TVector<TQueryInfo> QueryInfo;

        NPar::ILocalExecutor* LocalExecutor;
    };

}
