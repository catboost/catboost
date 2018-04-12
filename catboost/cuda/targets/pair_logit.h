#pragma once

#include "quality_metric_helpers.h"
#include "target_func.h"
#include "kernel.h"
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>

namespace NCatboostCuda {
    template <class TDocLayout,
              class TDataSet>
    class TPairLogit: public TQuerywiseTarget<TDocLayout, TDataSet> {
    public:
        using TParent = TQuerywiseTarget<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TPairLogit(const TDataSet& dataSet,
                   TRandom& random,
                   TSlice slice,
                   const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            Init(targetOptions);
        }

        TPairLogit(const TDataSet& dataSet,
                   TRandom& random,
                   const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            Init(targetOptions);
        }

        TPairLogit(const TPairLogit& target,
                   const TSlice& slice)
            : TParent(target,
                      slice) {
        }

        TPairLogit(const TPairLogit& target)
            : TParent(target)
        {
        }

        template <class TLayout>
        TPairLogit(const TPairLogit<TLayout, TDataSet>& basedOn,
                   TTarget<TMapping>&& target)
            : TParent(basedOn,
                      std::move(target)) {
        }

        TPairLogit(TPairLogit&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            const double weight = GetPairsTotalWeight();
            TVector<float> result;
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

            ApproximateForPermutation(point, /*indices*/ nullptr, &tmp, nullptr, nullptr);
            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            return TAdditiveStatistic(result[0], weight);
        }

        static double Score(const TAdditiveStatistic& score) {
            return -score.Sum / score.Weight;
        }

        double Score(const TConstVec& point) {
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
                                      nullptr,
                                      stream);
            weights.Copy(GetTarget().GetWeights());
        }

        void NewtonAt(const TConstVec& point,
                      TVec& weightedDer,
                      TVec& weightedDer2,
                      ui32 stream = 0) const {
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &weightedDer,
                                      &weightedDer2,
                                      stream);
        }

        void ApproximateForPermutation(const TConstVec& point,
                                       const TBuffer<ui32>* indices,
                                       TVec* value,
                                       TVec* der,
                                       TVec* der2,
                                       ui32 stream = 0) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();

            ApproximatePairLogit(samplesGrouping.GetPairs(),
                                 samplesGrouping.GetPairsWeights(),
                                 samplesGrouping.GetOffsetsBias(),
                                 point,
                                 indices,
                                 value,
                                 der,
                                 der2,
                                 stream);
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        static constexpr TStringBuf TargetName() {
            return "PairLogit";
        }

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::PairLogit);
            TVec weights = TVec::CopyMapping(TParent::GetTarget().GetTargets());
            FillBuffer(weights, 0.0f);
            MakePairWeights(TParent::GetSamplesGrouping().GetPairs(),
                            TParent::GetSamplesGrouping().GetPairsWeights(),
                            weights);
            TParent::Target.Weights = weights.ConstCopyView();
        }

    private:
        inline double GetPairsTotalWeight() const {
            if (PairsTotalWeight <= 0) {
                const auto& pairWeights = TParent::GetSamplesGrouping().GetPairsWeights();
                auto tmp = TVec::CopyMapping(pairWeights);
                FillBuffer(tmp, 1.0f);
                PairsTotalWeight = DotProduct(tmp, pairWeights);
                if (PairsTotalWeight <= 0) {
                    ythrow yexception() << "Observation weights should be greater or equal zero. Total weight should be greater, than zero";
                }
            }
            return PairsTotalWeight;
        }

    private:
        mutable double PairsTotalWeight = 0;
    };

}
