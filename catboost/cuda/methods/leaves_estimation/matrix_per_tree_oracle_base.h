#pragma once

#include "leaves_estimation_config.h"
#include "oracle_interface.h"

#include <catboost/cuda/methods/helpers.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>
#include <catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h>
#include <catboost/cuda/gpu_data/non_zero_filter.h>
#include <catboost/cuda/targets/oracle_type.h>

namespace NCatboostCuda {
    template <class TImpl>
    class TPairBasedOracleBase: public ILeavesEstimationOracle {
    public:
        TPairBasedOracleBase(TStripeBuffer<const float>&& baseline,
                             TStripeBuffer<const ui32>&& bins,
                             const TVector<double>& leafWeights,
                             const TVector<double>& pairLeafWeights,
                             const TLeavesEstimationConfig& estimationConfig,
                             TGpuAwareRandom& random)
            : LeavesEstimationConfig(estimationConfig)
            , Baseline(std::move(baseline))
            , Bins(std::move(bins))
            , BinWeightsSum(leafWeights)
            , PairBinWeightsSum(pairLeafWeights)
            , Random(random)
        {
            Cursor = TStripeBuffer<float>::CopyMapping(Baseline);

            {
                auto mapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount() + 1);
                ScoreAndFirstDerStats.Reset(mapping);
                FillBuffer(ScoreAndFirstDerStats, 0.0);

                PointStat.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(1));
                FillBuffer(PointStat, 0.0f);
            }

            if (estimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton) {
                auto pointMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount());
                PointwiseSecondDerStats.Reset(pointMapping);

                FillBuffer(PointwiseSecondDerStats, 0.0);

                auto pairMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount() * BinCount());
                PairwiseSecondDerOrWeightsStats.Reset(pairMapping);
                FillBuffer(PairwiseSecondDerOrWeightsStats, 0.0);
            }
        }

        ui32 HessianBlockSize() const final {
            return PointDim();
        }

        ui32 BinCount() const {
            return BinWeightsSum.size();
        }

        TVector<float> MakeEstimationResult(const TVector<float>& point) const override final {
            TVector<float> result = point;
            result.resize(BinCount());
            return result;
        }

        ui32 PointDim() const final {
            //otherwise for pure pairwise modes we remove last row to make system more stable (but l2 will be less fare)
            return TImpl::HasDiagonalPart() ? BinCount() : BinCount() - 1;
        }

        void MoveTo(const TVector<float>& dstPoint) final {
            auto point = dstPoint;
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
            RegularizeImpl(LeavesEstimationConfig, TConstArrayRef<double>(BinWeightsSum.begin(), BinWeightsSum.begin() + PointDim()), point);
        }

        void WriteValueAndFirstDerivatives(double* value,
                                           TVector<double>* gradient) override final {
            ComputeFirstOrderStats();

            auto& data = ScoreAndDer;
            (*value) = data[0];

            gradient->clear();
            gradient->resize(PointDim());

            Copy(data.begin() + 1,
                 data.begin() + 1 + PointDim(),
                 gradient->begin());

            AddRigdeRegulaizationIfNecessary(LeavesEstimationConfig, CurrentPoint, value, gradient);
        }

        void WriteSecondDerivatives(TVector<double>* secondDer) override final {
            ComputeSecondOrderStats();

            const ui32 pointDim = PointDim();
            const ui32 rowSize = BinCount();
            secondDer->clear();
            secondDer->resize(pointDim * pointDim);
            auto& sigma = *secondDer;

            if (TImpl::HasDiagonalPart()) {
                if (LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton) {
                    for (ui32 i = 0; i < pointDim; ++i) {
                        sigma[i * PointDim() + i] = PointDer2[i];
                    }
                }
            }

            const double cellPrior = 1.0 / rowSize;
            const double lambda = LeavesEstimationConfig.Lambda;
            const double bayesianLambda = LeavesEstimationConfig.NonDiagLambda;

            for (ui32 idx1 = 0; idx1 < rowSize; ++idx1) {
                for (ui32 idx2 = 0; idx2 < rowSize; ++idx2) {
                    if (idx1 == idx2) {
                        continue;
                    }
                    const double w = LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton
                                         ? PairDer2[idx1 * rowSize + idx2]
                                         : PairBinWeightsSum[idx1 * rowSize + idx2];

                    const bool needRemove = idx1 >= pointDim || idx2 >= pointDim;

                    const ui32 lowerIdx = idx2 * pointDim + idx1;
                    const ui32 upperIdx = idx1 * pointDim + idx2;

                    if (!needRemove) {
                        sigma[lowerIdx] -= w;
                        sigma[upperIdx] -= w;
                    }
                    if (idx1 < pointDim) {
                        sigma[idx1 * (pointDim + 1)] += w;
                    }
                    if (idx2 < pointDim) {
                        sigma[idx2 * (pointDim + 1)] += w;
                    }
                }
            }

            //regularize
            for (ui32 idx1 = 0; idx1 < pointDim; ++idx1) {
                for (ui32 idx2 = 0; idx2 < idx1; ++idx2) {
                    const ui32 lowerIdx = idx1 * pointDim + idx2;
                    const ui32 upperIdx = idx2 * pointDim + idx1;

                    sigma[lowerIdx] -= bayesianLambda * cellPrior;
                    sigma[upperIdx] -= bayesianLambda * cellPrior;
                }
                if (sigma[idx1 * pointDim + idx1] == 0) {
                    sigma[idx1 * pointDim + idx1] += 10.0;
                }
                sigma[idx1 * pointDim + idx1] += bayesianLambda * (1.0 - cellPrior) + lambda;
            }
        }

        void WriteWeights(TVector<double>* dst) final {
            *dst = BinWeightsSum;
        }

    private:
        void ComputeFirstOrderStats() {
            if (!HasFirstOrderStatsAtPoint) {
                auto scoreBuffer = NCudaLib::ParallelStripeView(ScoreAndFirstDerStats, TSlice(0, 1));
                auto derBuffer = NCudaLib::ParallelStripeView(ScoreAndFirstDerStats, TSlice(1, 1 + BinCount()));
                static_cast<TImpl*>(this)->FillScoreAndDer(&PointStat,
                                                           &derBuffer);
                CastCopy(PointStat, &scoreBuffer);
                ScoreAndDer = ReadReduce(ScoreAndFirstDerStats);
                HasFirstOrderStatsAtPoint = true;
            }
        }

        void ComputeSecondOrderStats() {
            if (!HasSecondOrderStatsAtPoint && LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton) {
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
        TCudaBuffer<float, NCudaLib::TStripeMapping> PointStat;
        TCudaBuffer<double, NCudaLib::TStripeMapping> ScoreAndFirstDerStats;
        TCudaBuffer<double, NCudaLib::TStripeMapping> PointwiseSecondDerStats;
        TCudaBuffer<double, NCudaLib::TStripeMapping> PairwiseSecondDerOrWeightsStats;

        TMirrorBuffer<float> LeafValues;

        TStripeBuffer<const float> Baseline;
        TStripeBuffer<const ui32> Bins;

        TVector<double> BinWeightsSum;
        TVector<float> CurrentPoint;

        bool HasFirstOrderStatsAtPoint = false;
        bool HasSecondOrderStatsAtPoint = false;

        TVector<double> ScoreAndDer;
        TVector<double> PointDer2;
        TVector<double> PairDer2;
        TVector<double> PairBinWeightsSum;

        TGpuAwareRandom& Random;
    };

}
