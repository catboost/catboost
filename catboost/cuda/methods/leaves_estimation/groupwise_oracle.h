#pragma once

#include "matrix_per_tree_oracle_base.h"
#include "leaves_estimation_helper.h"
#include <catboost/cuda/targets/oracle_type.h>
#include <catboost/cuda/methods/pairwise_kernels.h>

namespace NCatboostCuda {
    using EPtrType = NCudaLib::EPtrType;

    template <class TGroupwiseTarget>
    class TOracle<TGroupwiseTarget, EOracleType::Groupwise>
       : public TPairBasedOracleBase<TOracle<TGroupwiseTarget, EOracleType::Groupwise>> {
    public:
        using TParent = TPairBasedOracleBase<TOracle<TGroupwiseTarget, EOracleType::Groupwise>>;

        constexpr static bool HasDiagonalPart() {
            return true;
        }

        void FillScoreAndDer(TStripeBuffer<float>* score,
                             TStripeBuffer<double>* derStats) {
            auto& cursor = TParent::Cursor;
            Y_ASSERT(cursor.GetColumnCount() == 1);
            auto gatheredDer = TStripeBuffer<float>::CopyMapping(cursor);

            {
                auto der = TStripeBuffer<float>::CopyMapping(cursor);

                Target->ApproximateAt(cursor.AsConstBuf(),
                                      score,
                                      &der,
                                      &DiagDer2,
                                      &ShiftedDer2,
                                      &GroupDer2);
                Gather(gatheredDer,
                       der,
                       BinOrder);
            }

            ComputePartitionStats(gatheredDer,
                                  BinOffsets,
                                  derStats);
        }

        //we have guarantee, that FillScoreAndDer was called first
        void FillDer2(TStripeBuffer<double>* pointDer2Stats,
                      TStripeBuffer<double>* pairDer2Stats) {
            auto qids = Target->GetApproximateQids();

            {
                auto gatheredDer2 = TStripeBuffer<float>::CopyMapping(DiagDer2);

                Gather(gatheredDer2,
                       DiagDer2,
                       BinOrder);

                ComputePartitionStats(gatheredDer2,
                                      BinOffsets,
                                      pointDer2Stats);
            }

            auto pairDer2 = TStripeBuffer<float>::CopyMapping(SupportPairs);

            FillGroupwisePairDer2(ShiftedDer2,
                                  GroupDer2,
                                  qids,
                                  SupportPairs,
                                  &pairDer2);

            ComputePartitionStats(pairDer2,
                                  PairBinOffsets,
                                  pairDer2Stats);
        }

        TVector<float> EstimateExact() {
            CB_ENSURE(false, "Exact leaves estimation method on GPU is not supported for groupwise oracle");
        }

        void AddLangevinNoiseToDerivatives(TVector<double>* derivatives,
                                           NPar::ILocalExecutor* localExecutor) {
            Y_UNUSED(derivatives);
            Y_UNUSED(localExecutor);
            CB_ENSURE(!this->LeavesEstimationConfig.Langevin, "Langevin on GPU is not supported for groupwise oracle");
        }

        static THolder<ILeavesEstimationOracle> Create(const TGroupwiseTarget& target,
                                                       TStripeBuffer<const float>&& baseline,
                                                       TStripeBuffer<ui32>&& binsBuf,
                                                       ui32 binCount,
                                                       const TLeavesEstimationConfig& estimationConfig,
                                                       TGpuAwareRandom& random) {
            auto bins = binsBuf.AsConstBuf();
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
            TVector<double> pairLeafWeights;

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

            TVector<double> leafWeights;
            auto pointLeafIndices = TStripeBuffer<ui32>::CopyMapping(bins);
            TStripeBuffer<ui32> pointLeafOffsets;

            MakePointwiseComputeOrder(orderBins.ConstCopyView(),
                                      binCount,
                                      docOrderWeights,
                                      &pointLeafIndices,
                                      &pointLeafOffsets,
                                      &leafWeights);

            return THolder<ILeavesEstimationOracle>(new TOracle(target,
                               orderedBaseline.AsConstBuf(),
                               orderBins.AsConstBuf(),
                               leafWeights,
                               pairLeafWeights,
                               estimationConfig,
                               std::move(pairs),
                               std::move(pairLeafOffsets),
                               std::move(pointLeafOffsets),
                               std::move(pointLeafIndices),
                               random));
        }

    private:
        TOracle(const TGroupwiseTarget& target,
                /* ordered */
                TStripeBuffer<const float>&& baseline,
                /* ordered */
                TStripeBuffer<const ui32>&& bins,
                const TVector<double>& leafWeights,
                const TVector<double>& pairLeafWeights,
                const TLeavesEstimationConfig& estimationConfig,
                TStripeBuffer<uint2>&& pairs,
                TStripeBuffer<ui32>&& pairLeafOffset,
                TStripeBuffer<ui32>&& pointLeafOffsets,
                TStripeBuffer<ui32>&& pointLeafIndices,
                TGpuAwareRandom& random)
            : TParent(std::move(baseline),
                      std::move(bins),
                      leafWeights,
                      pairLeafWeights,
                      estimationConfig,
                      random)
            , Target(&target)
            , SupportPairs(std::move(pairs))
            , PairBinOffsets(std::move(pairLeafOffset))
            , BinOffsets(std::move(pointLeafOffsets))
            , BinOrder(std::move(pointLeafIndices))
        {
            CATBOOST_DEBUG_LOG << "Support pairs count " << SupportPairs.GetObjectsSlice().Size() << Endl;
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
