#pragma once

#include "target_func.h"
#include "kernel.h"
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/metrics/pfound.h>
#include <catboost/cuda/gpu_data/dataset_base.h>

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
                  TGpuAwareRandom& random,
                  TSlice slice,
                  const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            Init(targetOptions);
        }

        TYetiRank(const TDataSet& dataSet,
                  TGpuAwareRandom& random,
                  const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            Init(targetOptions);
        }

        TYetiRank(const TYetiRank& target,
                  const TSlice& slice)
            : TParent(target,
                      slice)
            , PermutationCount(target.GetPermutationCount())
        {
        }

        TYetiRank(const TYetiRank& target)
            : TParent(target)
            , PermutationCount(target.GetPermutationCount())
        {
        }

        template <class TLayout>
        TYetiRank(const TYetiRank<TLayout, TDataSet>& basedOn,
                  TTarget<TMapping>&& target)
            : TParent(basedOn,
                      std::move(target))
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

        TAdditiveStatistic ComputeStats(const TConstVec& point, const TMap<TString, TString> params = TMap<TString, TString>()) const {
            Y_UNUSED(point);
            Y_UNUSED(params);
            CB_ENSURE(false, "Unimplemented");
        }

        static double Score(const TAdditiveStatistic& score) {
            return score.Stats[0] / score.Stats[1];
        }

        double Score(const TConstVec& point) const {
            return Score(ComputeStats(point));
        }

        void GradientAt(const TConstVec& point,
                        TVec& weightedDer,
                        TVec& weights,
                        ui32 stream = 0) const {
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &weightedDer,
                                      &weights,
                                      stream);
        }

        //For YetiRank Newton approximation is meaningless
        void NewtonAt(const TConstVec& point,
                      TVec& weightedDer,
                      TVec& weights,
                      ui32 stream = 0) const {
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &weightedDer,
                                      &weights,
                                      stream);
        }

        void ApproximateForPermutation(const TConstVec& point,
                                       const TBuffer<ui32>* indices,
                                       TVec* value,
                                       TVec* der,
                                       TVec* weights,
                                       ui32 stream = 0) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();

            ApproximateYetiRank(TParent::GetRandom().NextUniformL(),
                                PermutationCount,
                                samplesGrouping.GetSizes(),
                                samplesGrouping.GetBiasedOffsets(),
                                samplesGrouping.GetOffsetsBias(),
                                GetTarget().GetTargets(),
                                point,
                                indices,
                                value,
                                der,
                                weights,
                                stream);
        }

        static constexpr bool IsMinOptimal() {
            return false;
        }

        static constexpr TStringBuf ScoreMetricName() {
            return "PFound";
        }

        ELossFunction GetScoreMetricType() const {
            return ELossFunction::PFound;
        }

        //TODO(noxoomo): rename it. not dynamic boosting/ctrs permutations :) how many queries we'll generate
        ui32 GetPermutationCount() const {
            return PermutationCount;
        }

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::YetiRank);

            PermutationCount = NCatboostOptions::GetYetiRankPermutations(targetOptions);
            const auto& grouping = TParent::GetSamplesGrouping();
            for (ui32 qid = 0; qid < grouping.GetQueryCount(); ++qid) {
                const auto querySize = grouping.GetQuerySize(qid);
                CB_ENSURE(querySize <= 1023, "Error: max query size supported on GPU is 1023, got " << querySize);
            }
        }
    private:
        ui32 PermutationCount = 10;
    };

}
