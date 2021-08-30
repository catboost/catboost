#pragma once

#include "leaves_estimation_config.h"
#include "oracle_interface.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>

#include <catboost/libs/helpers/math_utils.h>

namespace NCatboostCuda {
    class TBinOptimizedOracle: public ILeavesEstimationOracle {
    public:
        TBinOptimizedOracle(const TLeavesEstimationConfig& leavesEstimationConfig,
                            THolder<IPermutationDerCalcer>&& derCalcer,
                            TStripeBuffer<ui32>&& bins,
                            TStripeBuffer<ui32>&& partOffsets,
                            TStripeBuffer<float>&& cursor,
                            ui32 binCount,
                            TGpuAwareRandom& random);

        virtual ~TBinOptimizedOracle() {
        }

        TVector<float> MakeEstimationResult(const TVector<float>& point) const override final;

        ui32 PointDim() const final {
            return static_cast<ui32>(BinCount * SingleBinDim());
        }

        virtual ui32 HessianBlockSize() const final {
            if (LeavesEstimationConfig.LeavesEstimationMethod != ELeavesEstimation::Newton) {
                return 1;
            }

            if (DerCalcer->GetHessianType() == EHessianType::Diagonal) {
                return 1;
            } else {
                return static_cast<ui32>(SingleBinDim());
            }
        }

        void Regularize(TVector<float>* point) final;

        void MoveTo(const TVector<float>& point) final;

        void WriteValueAndFirstDerivatives(double* value, TVector<double>* gradient) final;

        void WriteSecondDerivatives(TVector<double>* secondDer) final;

        void WriteWeights(TVector<double>* dst) final;

        TVector<float> EstimateExact() final;

        void AddLangevinNoiseToDerivatives(TVector<double>* derivatives,
                                           NPar::ILocalExecutor* localExecutor) final;

    private:
        ui32 SingleBinDim() const {
            const ui32 cursorDim = Cursor.GetColumnCount();
            if (DerCalcer->GetType() == ELossFunction::MultiClass) {
                return cursorDim + 1;
            } else {
                return cursorDim;
            }
        }

    private:
        TLeavesEstimationConfig LeavesEstimationConfig;
        THolder<IPermutationDerCalcer> DerCalcer;
        TStripeBuffer<ui32> Bins;
        TStripeBuffer<ui32> Offsets;
        TStripeBuffer<float> Cursor;
        ui32 BinCount = 0;

        TVector<float> CurrentPoint;
        TVector<double> WeightsCpu;
        TMaybe<TVector<double>> DerAtPoint;
        TMaybe<TVector<double>> Der2AtPoint;

        TGpuAwareRandom& Random;
    };

    template <class TObjective>
    class TOracle<TObjective, EOracleType::Pointwise>: public TBinOptimizedOracle {
    public:
        static THolder<ILeavesEstimationOracle> Create(const TObjective& target,
                                                       TStripeBuffer<const float>&& baseline,
                                                       TStripeBuffer<ui32>&& bins,
                                                       ui32 binCount,
                                                       const TLeavesEstimationConfig& estimationConfig,
                                                       TGpuAwareRandom& random) {
            auto offsets = TStripeBuffer<ui32>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(binCount + 1));
            auto cursor = TStripeBuffer<float>::CopyMappingAndColumnCount(baseline);

            auto indices = TStripeBuffer<ui32>::CopyMapping(bins);
            MakeSequence(indices);
            ReorderBins(bins, indices, 0, binCount ? NCB::IntLog2(binCount) : 1);

            Gather(cursor, baseline, indices);
            UpdatePartitionOffsets(bins, offsets);

            auto derCalcer = CreatePermutationDerCalcer(TObjective(target), indices.AsConstBuf());

            return THolder<ILeavesEstimationOracle>(new TOracle(estimationConfig,
                               std::move(derCalcer),
                               std::move(bins),
                               std::move(offsets),
                               std::move(cursor),
                               binCount,
                               random));
        }

    private:
        TOracle(const TLeavesEstimationConfig& estimationConfig,
                THolder<IPermutationDerCalcer>&& derCalcer,
                TStripeBuffer<ui32>&& bins,
                TStripeBuffer<ui32>&& offsets,
                TStripeBuffer<float>&& cursor,
                ui32 binCount,
                TGpuAwareRandom& random)
            : TBinOptimizedOracle(estimationConfig,
                                  std::move(derCalcer),
                                  std::move(bins),
                                  std::move(offsets),
                                  std::move(cursor),
                                  binCount,
                                  random) {
        }
    };

}
