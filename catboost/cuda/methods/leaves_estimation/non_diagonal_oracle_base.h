#pragma once

#include "leaves_estimation_config.h"
#include "non_diagonal_oracle_interface.h"

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
#include <catboost/cuda/targets/non_diagonal_oralce_type.h>

namespace NCatboostCuda {
    template <class TImpl>
    class TNonDiagonalOracleBase: public INonDiagonalOracle {
    public:
        TNonDiagonalOracleBase(TStripeBuffer<const float>&& baseline,
                               TStripeBuffer<const ui32>&& bins,
                               const TVector<float>& leafWeights,
                               const TVector<float>& pairLeafWeights,
                               const TLeavesEstimationConfig& estimationConfig)
            : LeavesEstimationConfig(estimationConfig)
            , Baseline(std::move(baseline))
            , Bins(std::move(bins))
            , BinWeightsSum(leafWeights)
            , PairBinWeightsSum(pairLeafWeights)
        {
            Cursor = TStripeBuffer<float>::CopyMapping(Baseline);

            {
                auto mapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount() + 1);
                ScoreAndFirstDerStats.Reset(mapping);
                FillBuffer(ScoreAndFirstDerStats, 0.0f);
            }

            if (estimationConfig.UseNewton) {
                auto pointMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount());
                PointwiseSecondDerStats.Reset(pointMapping);

                FillBuffer(PointwiseSecondDerStats, 0.f);

                auto pairMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount() * BinCount());
                PairwiseSecondDerOrWeightsStats.Reset(pairMapping);
                FillBuffer(PairwiseSecondDerOrWeightsStats, 0.0f);
            }
        }

        ui32 BinCount() const {
            return BinWeightsSum.size();
        }

        ui32 PointDim() const final {
            return TImpl::HasDiagonalPart() ? BinCount() : BinCount() - 1;
        }

        void MoveTo(TVector<float> point) final {
            const ui32 pointDim = PointDim();
            CB_ENSURE(pointDim == point.size(), pointDim << " neq " << point.size());
            point.resize(BinCount(), 0);
            auto guard = NCudaLib::GetProfiler().Profile("Move to point");

            LeafValues.Reset(NCudaLib::TMirrorMapping(point.size()));
            LeafValues.Write(point);

            Cursor.Copy(Baseline);

            AddBinModelValues(LeafValues,
                              Bins,
                              Cursor);

            CurrentPoint = point;

            HasSecondOrderStatsAtPoint = false;
            HasFirstOrderStatsAtPoint = false;
        }

        void Regularize(TVector<float>* point) final {
            for (ui32 i = 0; i < point->size(); ++i) {
                if (BinWeightsSum[i] < LeavesEstimationConfig.MinLeafWeight) {
                    (*point)[i] = 0;
                }
            }
        }

        void WriteValueAndFirstDerivatives(double* value,
                                           TVector<float>* gradient) override final {
            ComputeFirstOrderStats();

            auto& data = ScoreAndDer;
            (*value) = data[0];

            gradient->clear();
            gradient->resize(PointDim());

            Copy(data.begin() + 1,
                 data.begin() + 1 + PointDim(),
                 gradient->begin());

            const double lambda = LeavesEstimationConfig.Lambda;

            if (LeavesEstimationConfig.AddRidgeToTargetFunction) {
                double hingeLoss = 0;
                {
                    for (const auto& val : CurrentPoint) {
                        hingeLoss += val * val;
                    }
                    hingeLoss *= lambda / 2;
                }
                (*value) -= hingeLoss;

                for (size_t i = 0; i < gradient->size(); ++i) {
                    (*gradient)[i] -= lambda * CurrentPoint[i];
                }
            }
        }

        void WriteSecondDerivatives(TVector<float>* secondDer) override final {
            ComputeSecondOrderStats();

            auto& sigma = *secondDer;
            const ui32 pointDim = PointDim();
            const ui32 rowSize = BinCount();
            secondDer->clear();
            secondDer->resize(pointDim * pointDim);

            if (TImpl::HasDiagonalPart()) {
                if (LeavesEstimationConfig.UseNewton) {
                    for (ui32 i = 0; i < pointDim; ++i) {
                        sigma[i * PointDim() + i] = PointDer2[i];
                    }
                } else {
                    for (ui32 i = 0; i < PointDim(); ++i) {
                        sigma[i * PointDim() + i] = BinWeightsSum[i];
                    }
                }
            }

            double cellPrior = 1.0 / rowSize;
            const double lambda = LeavesEstimationConfig.Lambda;
            const double nonDiagLambda = LeavesEstimationConfig.NonDiagLambda;

            for (ui32 idx1 = 0; idx1 < pointDim; ++idx1) {
                for (ui32 idx2 = 0; idx2 < idx1; ++idx2) {
                    const double der2OrWeightDirect = LeavesEstimationConfig.UseNewton
                                                          ? PairDer2[idx1 * rowSize + idx2]
                                                          : PairBinWeightsSum[idx1 * rowSize + idx2];

                    const double der2OrWeightInverse = LeavesEstimationConfig.UseNewton
                                                           ? PairDer2[idx2 * rowSize + idx1]
                                                           : PairBinWeightsSum[idx2 * rowSize + idx1];

                    const double cellWeight = nonDiagLambda * cellPrior + der2OrWeightDirect + der2OrWeightInverse;
                    //                    const double cellWeight = der2OrWeightDirect + der2OrWeightInverse;
                    const ui32 lowerIdx = idx1 * pointDim + idx2;
                    const ui32 upperIdx = idx2 * pointDim + idx1;

                    sigma[lowerIdx] -= cellWeight;
                    sigma[upperIdx] -= cellWeight;

                    sigma[idx1 * (pointDim + 1)] += der2OrWeightDirect + der2OrWeightInverse;
                    sigma[idx2 * (pointDim + 1)] += der2OrWeightDirect + der2OrWeightInverse;
                }
                sigma[idx1 * pointDim + idx1] += nonDiagLambda * (1.0 - cellPrior) + lambda;
            }
        }

        void WriteWeights(TVector<float>* dst) final {
            *dst = BinWeightsSum;
        }

    private:
        void ComputeFirstOrderStats() {
            if (!HasFirstOrderStatsAtPoint) {
                auto scoreBuffer = NCudaLib::ParallelStripeView(ScoreAndFirstDerStats, TSlice(0, 1));
                auto derBuffer = NCudaLib::ParallelStripeView(ScoreAndFirstDerStats, TSlice(1, 1 + BinCount()));
                static_cast<TImpl*>(this)->FillScoreAndDer(&scoreBuffer,
                                                           &derBuffer);
                ScoreAndDer = ReadReduce(ScoreAndFirstDerStats);
                HasFirstOrderStatsAtPoint = true;
            }
        }

        void ComputeSecondOrderStats() {
            if (!HasSecondOrderStatsAtPoint && LeavesEstimationConfig.UseNewton) {
                ComputeFirstOrderStats();
                static_cast<TImpl*>(this)->FillDer2(&PointwiseSecondDerStats,
                                                    &PairwiseSecondDerOrWeightsStats);
                if (TImpl::HasDiagonalPart()) {
                    PointDer2 = ReadReduce(PointwiseSecondDerStats);
                }
                PairDer2 = ReadReduce(PairwiseSecondDerOrWeightsStats);

                HasSecondOrderStatsAtPoint = true;
            }
        }

    protected:
        TLeavesEstimationConfig LeavesEstimationConfig;
        TStripeBuffer<float> Cursor;

    private:
        TCudaBuffer<float, NCudaLib::TStripeMapping> ScoreAndFirstDerStats;
        TCudaBuffer<float, NCudaLib::TStripeMapping> PointwiseSecondDerStats;
        TCudaBuffer<float, NCudaLib::TStripeMapping> PairwiseSecondDerOrWeightsStats;

        TMirrorBuffer<float> LeafValues;

        TStripeBuffer<const float> Baseline;
        TStripeBuffer<const ui32> Bins;

        TVector<float> BinWeightsSum;
        TVector<float> CurrentPoint;

        bool HasFirstOrderStatsAtPoint = false;
        bool HasSecondOrderStatsAtPoint = false;

        TVector<float> ScoreAndDer;
        TVector<float> PointDer2;
        TVector<float> PairDer2;
        TVector<float> PairBinWeightsSum;
    };

    template <class TTarget,
              ENonDiagonalOracleType Type = TTarget::NonDiagonalOracleType()>
    class TNonDiagonalOracle;

    template <class TNonDiagonalTarget>
    class TNonDiagonalOracleFactory: public INonDiagonalOracleFactory {
    public:
        TNonDiagonalOracleFactory(const TNonDiagonalTarget& target)
            : Target(&target)
        {
        }

        THolder<INonDiagonalOracle> Create(const TLeavesEstimationConfig& config,
                                           TStripeBuffer<const float>&& baseline,
                                           TStripeBuffer<const ui32>&& bins,
                                           ui32 binCount) const final {
            using TOracle = TNonDiagonalOracle<TNonDiagonalTarget>;
            return TOracle::Create(*Target,
                                   std::move(baseline),
                                   std::move(bins),
                                   binCount,
                                   config);
        }

    private:
        const TNonDiagonalTarget* Target;
    };

}
