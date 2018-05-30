#pragma once

#include "non_diagonal_oracle_base.h"
#include "leaves_estimation_helper.h"
#include <catboost/cuda/targets/non_diagonal_oralce_type.h>
#include <catboost/cuda/methods/pairwise_kernels.h>

namespace NCatboostCuda {

    template<class TGroupwiseTarget>
    class TNonDiagonalOracle<TGroupwiseTarget, ENonDiagonalOracleType::Groupwise>
            : public TNonDiagonalOracleBase<TNonDiagonalOracle<TGroupwiseTarget, ENonDiagonalOracleType::Groupwise>> {
    public:
        using TParent = TNonDiagonalOracleBase<TNonDiagonalOracle<TGroupwiseTarget, ENonDiagonalOracleType::Groupwise>>;

        constexpr static bool HasDiagonalPart() {
            return true;
        }

        void FillScoreAndDer(TStripeBuffer<float>* score,
                             TStripeBuffer<float>* derStats) {
            auto& cursor = TParent::Cursor;
            auto gatheredDer = TStripeBuffer<float>::CopyMapping(cursor);

            {
                auto der = TStripeBuffer<float>::CopyMapping(cursor);

                Target->ApproximateAt(cursor,
                                      score,
                                      &der,
                                      &DiagDer2,
                                      &ShiftedDer2,
                                      &GroupDer2);
                Gather(gatheredDer,
                       der,
                       BinOrder);
            }

            SegmentedReduceVector<float, NCudaLib::TStripeMapping, EPtrType::CudaDevice>(gatheredDer,
                                                                                         BinOffsets,
                                                                                         *derStats);
        }

        //we have guarantee, that FillScoreAndDer was called first
        void FillDer2(TStripeBuffer<float>* pointDer2Stats,
                      TStripeBuffer<float>* pairDer2Stats) {
            auto qids = Target->GetApproximateQids();

            {
                auto gatheredDer2 = TStripeBuffer<float>::CopyMapping(DiagDer2);

                Gather(gatheredDer2,
                       DiagDer2,
                       BinOrder);

                SegmentedReduceVector<float, NCudaLib::TStripeMapping, EPtrType::CudaDevice>(gatheredDer2,
                                                                                             BinOffsets,
                                                                                             *pointDer2Stats);
            }

            auto pairDer2 = TStripeBuffer<float>::CopyMapping(SupportPairs);

            FillGroupwisePairDer2(ShiftedDer2,
                                  GroupDer2,
                                  qids,
                                  SupportPairs,
                                  &pairDer2);

            SegmentedReduceVector<float, NCudaLib::TStripeMapping, EPtrType::CudaDevice>(pairDer2,
                                                                                         PairBinOffsets,
                                                                                         *pairDer2Stats);
        }

        static THolder<INonDiagonalOracle> Create(const TGroupwiseTarget& target,
                                                  TStripeBuffer<const float>&& baseline,
                                                  TStripeBuffer<const ui32>&& bins,
                                                  ui32 binCount,
                                                  const TLeavesEstimationConfig& estimationConfig) {
            //order and metadata used in Approximate
            auto docOrder = target.GetApproximateDocOrder();
            auto qids = target.GetApproximateQids();
            auto docOrderWeights = target.GetApproximateOrderWeights();

            auto orderBins = TStripeBuffer<ui32>::CopyMapping(bins);
            Gather(orderBins, bins, docOrder);

            auto orderedBaseline = TStripeBuffer<float>::CopyMapping(baseline);
            Gather(orderedBaseline, baseline, docOrder);

            //in docOrder
            TStripeBuffer<uint2> pairs;
            TStripeBuffer<ui32> pairLeafOffsets;
            TVector<float> pairLeafWeights;

            {
                target.CreateSecondDerMatrix(&pairs);

                TStripeBuffer<float> pairWeights;
                pairWeights = TStripeBuffer<float>::CopyMapping(pairs);
                FillBuffer(pairWeights, 1.0f);

                MakeSupportPairsMatrix(orderBins.ConstCopyView(),
                                       binCount,
                                       &pairs,
                                       &pairWeights,
                                       &pairLeafOffsets,
                                       &pairLeafWeights);
            }

            TVector<float> leafWeights;
            auto pointLeafIndices = TStripeBuffer<ui32>::CopyMapping(bins);
            TStripeBuffer<ui32> pointLeafOffsets;

            MakePointwiseComputeOrder(orderBins.ConstCopyView(),
                                      binCount,
                                      docOrderWeights,
                                      &pointLeafIndices,
                                      &pointLeafOffsets,
                                      &leafWeights);

            return new TNonDiagonalOracle(target,
                                          orderedBaseline,
                                          orderBins,
                                          leafWeights,
                                          pairLeafWeights,
                                          estimationConfig,
                                          std::move(pairs),
                                          std::move(pairLeafOffsets),
                                          std::move(pointLeafOffsets),
                                          std::move(pointLeafIndices));
        }

    private:
        TNonDiagonalOracle(const TGroupwiseTarget& target,
                /* ordered */
                           TStripeBuffer<const float>&& baseline,
                /* ordered */
                           TStripeBuffer<const ui32>&& bins,
                           const TVector<float>& leafWeights,
                           const TVector<float>& pairLeafWeights,
                           const TLeavesEstimationConfig& estimationConfig,
                           TStripeBuffer<uint2>&& pairs,
                           TStripeBuffer<ui32>&& pairLeafOffset,
                           TStripeBuffer<ui32>&& pointLeafOffsets,
                           TStripeBuffer<ui32>&& pointLeafIndices)
                : TParent(std::move(baseline),
                          std::move(bins),
                          leafWeights,
                          pairLeafWeights,
                          estimationConfig)
                  , Target(&target)
                  , SupportPairs(std::move(pairs))
                  , PairBinOffsets(std::move(pairLeafOffset))
                  , BinOffsets(std::move(pointLeafOffsets))
                  , BinOrder(std::move(pointLeafIndices)) {
            MATRIXNET_DEBUG_LOG << "Support pairs count " << SupportPairs.GetObjectsSlice().Size() << Endl;
            DiagDer2.Reset(TParent::Cursor.GetMapping());
            ShiftedDer2.Reset(TParent::Cursor.GetMapping());
            GroupDer2.Reset(Target->GetApproximateQidOffsets().GetMapping());
        }

    private:
        const TGroupwiseTarget* Target;

        TStripeBuffer<uint2> SupportPairs;
        TStripeBuffer<ui32> PairBinOffsets;

        TStripeBuffer<float> DiagDer2;
        TStripeBuffer<float> ShiftedDer2;
        TStripeBuffer<float> GroupDer2;

        TStripeBuffer<ui32> BinOffsets;
        TStripeBuffer<ui32> BinOrder;
    };
}
