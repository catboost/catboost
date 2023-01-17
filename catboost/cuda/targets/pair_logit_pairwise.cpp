#include "pair_logit_pairwise.h"

#include "kernel.h"

#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_reader.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/gpu_data/non_zero_filter.h>

namespace NCatboostCuda {
    TAdditiveStatistic TPairLogitPairwise<NCudaLib::TStripeMapping>::ComputeStats(
        const TPairLogitPairwise<NCudaLib::TStripeMapping>::TConstVec& point,
        const TMap<TString, TString> params) const {
        CB_ENSURE(params.size() == 0);

        const auto& samplesGrouping = TParent::GetSamplesGrouping();
        TVector<float> result;
        auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));
        FillBuffer(tmp, 0.0f);

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

    double TPairLogitPairwise<NCudaLib::TStripeMapping>::GetPairsTotalWeight() const {
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

    void TPairLogitPairwise<NCudaLib::TStripeMapping>::ApproximateAt(
        const TPairLogitPairwise<NCudaLib::TStripeMapping>::TConstVec& point,
        const TStripeBuffer<uint2>& pairs,
        const TStripeBuffer<float>& pairWeights, const TStripeBuffer<ui32>& scatterDerIndices,
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

    void TPairLogitPairwise<NCudaLib::TStripeMapping>::FillPairsAndWeightsAtPoint(
        const TPairLogitPairwise<NCudaLib::TStripeMapping>::TConstVec&, TStripeBuffer<uint2>* pairs,
        TStripeBuffer<float>* pairWeights) const {
        const auto& samplesGrouping = TParent::GetSamplesGrouping();

        pairs->Reset(samplesGrouping.GetPairs().GetMapping());
        pairWeights->Reset(samplesGrouping.GetPairsWeights().GetMapping());

        pairs->Copy(samplesGrouping.GetPairs());
        RemoveOffsetsBias(samplesGrouping.GetOffsetsBias(), pairs);
        pairWeights->Copy(samplesGrouping.GetPairsWeights());
    }

    void TPairLogitPairwise<NCudaLib::TStripeMapping>::ApproximateStochastic(
        const TPairLogitPairwise<NCudaLib::TStripeMapping>::TConstVec& point,
        const NCatboostOptions::TBootstrapConfig& config, bool secondDer,
        TNonDiagQuerywiseTargetDers* target) const {
        const auto& samplesGrouping = TParent::GetSamplesGrouping();

        auto& pairWeights = target->PairDer2OrWeights;
        auto& gradient = target->PointWeightedDer;
        auto& sampledDocs = target->Docs;
        target->PointDer2OrWeights.Clear();

        CB_ENSURE(samplesGrouping.GetPairs().GetObjectsSlice().Size(), "No pairs found");

        gradient.Reset(point.GetMapping());
        sampledDocs.Reset(point.GetMapping());
        MakeSequence(sampledDocs);

        auto& pairs = target->Pairs;

        {
            pairWeights.Reset(samplesGrouping.GetPairsWeights().GetMapping());
            pairWeights.Copy(samplesGrouping.GetPairsWeights());

            TBootstrap<NCudaLib::TStripeMapping> bootstrap(config);
            bootstrap.Bootstrap(TParent::GetRandom(), pairWeights);

            auto nzPairIndices = TCudaBuffer<ui64, TMapping>::CopyMapping(pairWeights);
            FilterZeroEntries(&pairWeights,
                              &nzPairIndices);

            pairs.Reset(nzPairIndices.GetMapping());
            Gather(pairs, samplesGrouping.GetPairs(), nzPairIndices);
        }

        RemoveOffsetsBias(samplesGrouping.GetOffsetsBias(), &pairs);
        PairLogitPairwise(point,
                          pairs.ConstCopyView(),
                          pairWeights.ConstCopyView(),
                          &gradient,
                          secondDer ? &pairWeights : nullptr);
    }
}
