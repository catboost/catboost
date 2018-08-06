#pragma once

#include "leaves_estimation_config.h"
#include "oracle_interface.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>

namespace NCatboostCuda {

    class TBinOptimizedOracle  : public ILeavesEstimationOracle {
    public:
        TBinOptimizedOracle(const TLeavesEstimationConfig& leavesEstimationConfig,
                            bool multiclassLastDimHack,
                            THolder<IPermutationDerCalcer>&& derCalcer,
                            TStripeBuffer<ui32>&& bins,
                            TStripeBuffer<ui32>&& partOffsets,
                            TStripeBuffer<float>&& cursor,
                            ui32 binCount);

        virtual ~TBinOptimizedOracle() {
        }

        TVector<float> MakeEstimationResult(const TVector<float>& point) const override final;
        ui32 PointDim() const final {
            return static_cast<ui32>(BinCount * SingleBinDim());
        }
        virtual ui32 HessianBlockSize() const final {
            return LeavesEstimationConfig.UseNewton ? static_cast<ui32>(SingleBinDim()) : 1;
        }

        void Regularize(TVector<float>* point) final;

        void MoveTo(const TVector<float>& point) final;

        void WriteValueAndFirstDerivatives(double* value, TVector<double>* gradient) final;

        void WriteSecondDerivatives(TVector<double>* secondDer) final;

        void WriteWeights(TVector<double>* dst) final;

    private:
        ui32 SingleBinDim() const {
            const ui32 cursorDim = Cursor.GetColumnCount();
            if (MulticlassLastDimHack) {
                return cursorDim + 1;
            } else {
                return cursorDim;
            }
        }
    private:
        TLeavesEstimationConfig LeavesEstimationConfig;
        bool MulticlassLastDimHack;
        THolder<IPermutationDerCalcer> DerCalcer;
        TStripeBuffer<ui32> Bins;
        TStripeBuffer<ui32> Offsets;
        TStripeBuffer<float> Cursor;
        ui32 BinCount = 0;


        TVector<float> CurrentPoint;
        TVector<double> WeightsCpu;
        TMaybe<TVector<double>> DerAtPoint;
    };



    template <class TObjective>
    class TOracle<TObjective, EOracleType::Pointwise> : public TBinOptimizedOracle {
    public:

        static THolder<ILeavesEstimationOracle> Create(const TObjective& target,
                                                       TStripeBuffer<const float>&& baseline,
                                                       TStripeBuffer<ui32>&& bins,
                                                       ui32 binCount,
                                                       const TLeavesEstimationConfig& estimationConfig) {

            auto offsets = TStripeBuffer<ui32>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(binCount + 1));
            auto cursor = TStripeBuffer<float>::CopyMappingAndColumnCount(baseline);

            auto indices = TStripeBuffer<ui32>::CopyMapping(bins);
            MakeSequence(indices);
            ReorderBins(bins, indices, 0, binCount ? IntLog2(binCount) : 1);

            Gather(cursor, baseline, indices);
            UpdatePartitionOffsets(bins, offsets);

            auto derCalcer = CreatePermutationDerCalcer(TObjective(target), std::move(indices));

            return new TOracle(estimationConfig,
                               target.GetType() == ELossFunction::MultiClass,
                               std::move(derCalcer),
                               std::move(bins),
                               std::move(offsets),
                               std::move(cursor),
                               binCount);
        }
    private:

        TOracle(const TLeavesEstimationConfig& estimationConfig,
                bool multiclassLastDimHack,
                THolder<IPermutationDerCalcer>&& derCalcer,
                TStripeBuffer<ui32>&& bins,
                TStripeBuffer<ui32>&& offsets,
                TStripeBuffer<float>&& cursor,
                ui32 binCount) :
             TBinOptimizedOracle(estimationConfig,
                                 multiclassLastDimHack,
                                 std::move(derCalcer),
                                 std::move(bins),
                                 std::move(offsets),
                                 std::move(cursor),
                                 binCount) {

        }
    };



}
