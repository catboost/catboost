#pragma once

#include "descent_helpers.h"
#include "leaves_estimation_helper.h"
#include "leaves_estimation_config.h"
#include "oracle_interface.h"
#include "matrix_per_tree_oracle_base.h"

#include <catboost/cuda/methods/helpers.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>
#include <catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h>
#include <catboost/cuda/gpu_data/non_zero_filter.h>

namespace NCatboostCuda {
    using EPtrType = NCudaLib::EPtrType;

    template <class TPairwiseTarget>
    class TOracle<TPairwiseTarget, EOracleType::Pairwise>: public TPairBasedOracleBase<TOracle<TPairwiseTarget, EOracleType::Pairwise>> {
    public:
        using TParent = TPairBasedOracleBase<TOracle<TPairwiseTarget, EOracleType::Pairwise>>;

        constexpr static bool HasDiagonalPart() {
            return false;
        }

        void FillScoreAndDer(TStripeBuffer<float>* score,
                             TStripeBuffer<double>* derStats) {
            PairDer2.Reset(SupportPairs.GetMapping());

            auto& cursor = TParent::Cursor;
            auto der = TStripeBuffer<float>::CopyMapping(cursor);

            Target->ApproximateAt(cursor.AsConstBuf(),
                                  SupportPairs,
                                  PairWeights,
                                  ScatterDersOrder,
                                  score,
                                  &der,
                                  &PairDer2);

            ComputePartitionStats(der, PointBinOffsets, derStats);
        }

        void FillDer2(TStripeBuffer<double>*,
                      TStripeBuffer<double>* pairDer2Stats) {
            ComputePartitionStats(PairDer2, PairBinOffsets, pairDer2Stats);
        }

        static THolder<ILeavesEstimationOracle> Create(const TPairwiseTarget& target,
                                                       TStripeBuffer<const float>&& baseline,
                                                       TStripeBuffer<ui32>&& binsBuf,
                                                       ui32 binCount,
                                                       const TLeavesEstimationConfig& estimationConfig,
                                                       TGpuAwareRandom& random) {
            auto bins = binsBuf.AsConstBuf();
            TStripeBuffer<uint2> pairs;
            TStripeBuffer<float> pairWeights;

            target.FillPairsAndWeightsAtPoint(baseline,
                                              &pairs,
                                              &pairWeights);

            TVector<double> pairPartsLeafWeights;
            TStripeBuffer<ui32> pairLeafOffsets;

            MakeSupportPairsMatrix(bins,
                                   binCount,
                                   &pairs,
                                   &pairWeights,
                                   &pairLeafOffsets,
                                   &pairPartsLeafWeights);

            TVector<double> leafWeights;
            auto pointLeafIndices = TStripeBuffer<ui32>::CopyMapping(bins);
            TStripeBuffer<ui32> pointLeafOffsets;

            MakePointwiseComputeOrder(bins,
                                      binCount,
                                      target.GetTarget().GetWeights(),
                                      &pointLeafIndices,
                                      &pointLeafOffsets,
                                      &leafWeights);

            return THolder<ILeavesEstimationOracle>(new TOracle(target,
                               std::move(baseline),
                               bins.AsConstBuf(),
                               leafWeights,
                               pairPartsLeafWeights,
                               estimationConfig,
                               std::move(pairs),
                               std::move(pairWeights),
                               std::move(pairLeafOffsets),
                               std::move(pointLeafOffsets),
                               std::move(pointLeafIndices),
                               random));
        }

        TVector<float> EstimateExact() {
            CB_ENSURE(false, "Exact leaves estimation method on GPU is not supported for pairwise oracle");
        }

        void AddLangevinNoiseToDerivatives(TVector<double>* derivatives,
                                           NPar::ILocalExecutor* localExecutor) {
            Y_UNUSED(derivatives);
            Y_UNUSED(localExecutor);
            CB_ENSURE(!this->LeavesEstimationConfig.Langevin, "Langevin on GPU is not supported for pairwise oracle");
        }

    private:
        TOracle(const TPairwiseTarget& target,
                TStripeBuffer<const float>&& baseline,
                TStripeBuffer<const ui32>&& bins,
                const TVector<double>& leafWeights,
                const TVector<double>& pairLeafWeights,
                const TLeavesEstimationConfig& estimationConfig,
                TStripeBuffer<uint2>&& pairs,
                TStripeBuffer<float>&& pairWeights,
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
            , PairWeights(std::move(pairWeights))
            , PairBinOffsets(std::move(pairLeafOffset))
            , PointBinOffsets(std::move(pointLeafOffsets))
            , ScatterDersOrder(TStripeBuffer<ui32>::CopyMapping(pointLeafIndices))
        {
            InversePermutation(pointLeafIndices, ScatterDersOrder);
            CATBOOST_DEBUG_LOG << "Support pairs count " << SupportPairs.GetObjectsSlice().Size() << Endl;
        }

    private:
        const TPairwiseTarget* Target;
        TStripeBuffer<uint2> SupportPairs;
        TStripeBuffer<float> PairWeights;
        TStripeBuffer<ui32> PairBinOffsets;
        TStripeBuffer<float> PairDer2;

        TStripeBuffer<ui32> PointBinOffsets;
        TStripeBuffer<ui32> ScatterDersOrder;
    };

}
