#pragma once

#include "quality_metric_helpers.h"
#include "target_base.h"
#include "kernel.h"
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/metrics/pfound.h>

namespace NCatboostCuda {
    template <class TDocLayout,
              class TDataSet>
    class TYetiRank: public TQuerywiseTarget<TDocLayout, TDataSet> {
    public:
        using TParent = TQuerywiseTarget<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TYetiRank(const TDataSet& dataSet,
                  TRandom& random,
                  TSlice slice,
                  const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::YetiRank);
            const auto& options = targetOptions.GetLossParams();

            if (options.has("PermutationCount")) {
                PermutationCount = FromString<ui32>(options.at("PermutationCount"));
            }
            const auto& grouping = TParent::GetSamplesGrouping();
            for (ui32 qid = 0; qid < grouping.GetQueryCount(); ++qid) {
                const auto querySize = grouping.GetQuerySize(qid);
                CB_ENSURE(querySize <= 1023, "Error: max query size supported on GPU is 1023, got " << querySize);
            }
        }

        TYetiRank(const TYetiRank& target,
                  const TSlice& slice)
            : TParent(target,
                      slice)
            , PermutationCount(target.GetPermutationCount())
        {
        }

        template <class TLayout>
        TYetiRank(const TYetiRank<TLayout, TDataSet>& basedOn,
                  TCudaBuffer<const float, TMapping>&& target,
                  TCudaBuffer<const float, TMapping>&& weights,
                  TCudaBuffer<const ui32, TMapping>&& indices)
            : TParent(basedOn,
                      std::move(target),
                      std::move(weights),
                      std::move(indices))
            , PermutationCount(basedOn.GetPermutationCount())
        {
        }

        TYetiRank(TYetiRank&& other)
            : TParent(std::move(other))
            , PermutationCount(other.PermutationCount)
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;
        using TParent::GetWeights;

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            TVector<float> pointCpu;
            point.Read(pointCpu);

            if (TargetCpu.size() == 0) {
                GetTarget().Read(TargetCpu);
            }

            const auto& samplesGrouping = TParent::GetSamplesGrouping();
            const ui32 queryCount = samplesGrouping.GetQueryCount();

            TPFoundCalcer calcer;
            for (ui32 query = 0; query < queryCount; ++query) {
                ui32 offset = samplesGrouping.GetQueryOffset(query);
                ui32 querySize = samplesGrouping.GetQuerySize(query);
                const ui32* groupIds = samplesGrouping.GetGroupIds(query);
                calcer.AddQuery(~TargetCpu + offset, ~pointCpu + offset, groupIds, querySize);
            }

            auto metricHolder = calcer.GetMetric();
            return TAdditiveStatistic(metricHolder.Error, metricHolder.Weight);
        }

        static double Score(const TAdditiveStatistic& score) {
            return score.Sum / score.Weight;
        }

        double Score(const TConstVec& point) {
            return Score(ComputeStats(point));
        }

        void GradientAt(const TConstVec& point,
                        TVec& dst,
                        TVec& weights,
                        ui32 stream = 0) const {
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &dst,
                                      &weights,
                                      stream);
        }

        void ApproximateForPermutation(const TConstVec& point,
                                       const TBuffer<ui32>* indices,
                                       TVec* value,
                                       TVec* der,
                                       TVec* der2,
                                       ui32 stream = 0) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();

            ApproximateYetiRank(TParent::GetRandom().NextUniformL(), PermutationCount,
                                samplesGrouping.GetSizes(),
                                samplesGrouping.GetBiasedOffsets(),
                                samplesGrouping.GetOffsetsBias(),
                                GetTarget(),
                                point,
                                indices,
                                value,
                                der,
                                der2,
                                stream);
        }

        static constexpr bool IsMinOptimal() {
            return false;
        }

        static constexpr TStringBuf TargetName() {
            return "YetiRank";
        }

        ui32 GetPermutationCount() const {
            return PermutationCount;
        }

    private:
        mutable TVector<float> TargetCpu;
        ui32 PermutationCount = 10;
    };

}
