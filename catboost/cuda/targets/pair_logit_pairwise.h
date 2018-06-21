#pragma once

#include "target_func.h"
#include "kernel.h"
#include "non_diag_target_der.h"
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/metrics/pfound.h>
#include <catboost/cuda/gpu_data/dataset_base.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/methods/helpers.h>

namespace NCatboostCuda {
    template <class TSamplesMapping, class TDataSet>
    class TPairLogitPairwise;

    template <class TDataSet>
    class TPairLogitPairwise<NCudaLib::TStripeMapping, TDataSet>: public TNonDiagQuerywiseTarget<NCudaLib::TStripeMapping, TDataSet> {
    public:
        using TSamplesMapping = NCudaLib::TStripeMapping;
        using TParent = TNonDiagQuerywiseTarget<TSamplesMapping, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TSamplesMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TPairLogitPairwise(const TDataSet& dataSet,
                           TGpuAwareRandom& random,
                           const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            Init(targetOptions);
        }

        TPairLogitPairwise(const TPairLogitPairwise& target)
            : TParent(target)
        {
        }

        TPairLogitPairwise(TPairLogitPairwise&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString> params = TMap<TString, TString>()) const {
            CB_ENSURE(params.size() == 0);

            const auto& samplesGrouping = TParent::GetSamplesGrouping();
            TVector<float> result;
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

            ApproximatePairLogit(samplesGrouping.GetPairs(),
                                 samplesGrouping.GetPairsWeights(),
                                 samplesGrouping.GetOffsetsBias(),
                                 point,
                                 (const TBuffer<ui32>*)nullptr,
                                 &tmp,
                                 (TBuffer<float>*)nullptr,
                                 (TBuffer<float>*)nullptr);

            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            return MakeSimpleAdditiveStatistic(result[0], GetPairsTotalWeight());
        }

        static double Score(const TAdditiveStatistic& score) {
            return -score.Stats[0] / score.Stats[1];
        }

        double Score(const TConstVec& point) const {
            return Score(ComputeStats(point));
        }

        void StochasticGradient(const TConstVec& point,
                                const NCatboostOptions::TBootstrapConfig& config,
                                TNonDiagQuerywiseTargetDers* target) const {
            ApproximateStochastic(point, config, false, target);
        }

        void StochasticNewton(const TConstVec& point,
                              const NCatboostOptions::TBootstrapConfig& config,
                              TNonDiagQuerywiseTargetDers* target) const {
            ApproximateStochastic(point, config, true, target);
        }

        void ApproximateStochastic(const TConstVec& point,
                                   const NCatboostOptions::TBootstrapConfig& config,
                                   bool secondDer,
                                   TNonDiagQuerywiseTargetDers* target) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();

            auto& pairWeights = target->PairDer2OrWeights;
            auto& gradient = target->PointWeightedDer;
            auto& sampledDocs = target->Docs;
            target->PointDer2OrWeights.Clear();

            CB_ENSURE(samplesGrouping.GetPairs().GetObjectsSlice().Size());

            gradient.Reset(point.GetMapping());
            sampledDocs.Reset(point.GetMapping());
            MakeSequence(sampledDocs);

            auto& pairs = target->Pairs;

            {
                pairWeights.Reset(samplesGrouping.GetPairsWeights().GetMapping());
                pairWeights.Copy(samplesGrouping.GetPairsWeights());

                TBootstrap<NCudaLib::TStripeMapping> bootstrap(config);
                bootstrap.Bootstrap(TParent::GetRandom(),
                                    pairWeights);

                auto nzPairIndices = TCudaBuffer<ui32, TMapping>::CopyMapping(pairWeights);
                FilterZeroEntries(&pairWeights,
                                  &nzPairIndices);

                pairs.Reset(nzPairIndices.GetMapping());
                Gather(pairs, samplesGrouping.GetPairs(), nzPairIndices);
            }

            PairLogitPairwise(point,
                              pairs.ConstCopyView(),
                              pairWeights.ConstCopyView(),
                              &gradient,
                              secondDer ? &pairWeights : nullptr);
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        static constexpr TStringBuf ScoreMetricName() {
            return "PairLogitPairwise";
        }

        ELossFunction GetScoreMetricType() const {
            return ELossFunction::PairLogit;
        }

        static constexpr ENonDiagonalOracleType NonDiagonalOracleType() {
            return ENonDiagonalOracleType::Pairwise;
        }

        void FillPairsAndWeightsAtPoint(const TConstVec&,
                                        TStripeBuffer<uint2>* pairs,
                                        TStripeBuffer<float>* pairWeights) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();

            pairs->Reset(samplesGrouping.GetPairs().GetMapping());
            pairWeights->Reset(samplesGrouping.GetPairsWeights().GetMapping());

            pairs->Copy(samplesGrouping.GetPairs());
            pairWeights->Copy(samplesGrouping.GetPairsWeights());
        }

        void ApproximateAt(const TConstVec& point,
                           const TStripeBuffer<uint2>& pairs,
                           const TStripeBuffer<float>& pairWeights,
                           const TStripeBuffer<ui32>& scatterDerIndices,
                           TStripeBuffer<float>* value,
                           TStripeBuffer<float>* der,
                           TStripeBuffer<float>* pairDer2) const {
            PairLogitPairwise(point,
                              pairs,
                              pairWeights,
                              scatterDerIndices,
                              value,
                              der,
                              pairDer2);
        }

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::PairLogitPairwise);
        }

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
