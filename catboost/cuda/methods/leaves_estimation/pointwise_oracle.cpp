#include "leaves_estimation_helper.h"
#include "pointwise_oracle.h"
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>

namespace NCatboostCuda {
    void TBinOptimizedOracle::WriteWeights(TVector<double>* dst) {
        (*dst) = WeightsCpu;
    }

    void TBinOptimizedOracle::Regularize(TVector<float>* point) {
        const ui32 approxDim = SingleBinDim();
        RegularizeImpl(LeavesEstimationConfig, TConstArrayRef<double>(WeightsCpu.begin(), WeightsCpu.begin() + PointDim() / approxDim), point, approxDim);
    }

    TVector<float> TBinOptimizedOracle::MakeEstimationResult(const TVector<float>& point) const {
        const ui32 cursorDim = static_cast<const ui32>(Cursor.GetColumnCount());

        TVector<float> newPoint;
        if (DerCalcer->GetType() == ELossFunction::MultiClass) {
            newPoint.resize(cursorDim * BinCount);
            for (ui32 bin = 0; bin < BinCount; ++bin) {
                for (ui32 dim = 0; dim < cursorDim; ++dim) {
                    newPoint[bin * cursorDim + dim] = point[bin * SingleBinDim() + dim] - point[bin * SingleBinDim() + cursorDim];
                }
            }
        } else {
            newPoint = point;
        }
        return newPoint;
    }

    void TBinOptimizedOracle::MoveTo(const TVector<float>& point) {
        auto guard = NCudaLib::GetProfiler().Profile("Move to point");
        const ui32 cursorDim = static_cast<const ui32>(Cursor.GetColumnCount());
        auto toAppend = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(BinCount * cursorDim));
        CB_ENSURE(point.size() == BinCount * SingleBinDim());

        TVector<float> shift(cursorDim * BinCount);

        TVector<float> newPoint = MakeEstimationResult(point);

        for (size_t i = 0; i < shift.size(); ++i) {
            shift[i] = newPoint[i] - CurrentPoint[i];
        }

        toAppend.Write(shift);
        AddBinModelValues(toAppend,
                          Bins,
                          Cursor);
        DerAtPoint.Clear();
        Der2AtPoint.Clear();
        CurrentPoint = newPoint;
    }

    void TBinOptimizedOracle::WriteValueAndFirstDerivatives(double* value,
                                                            TVector<double>* gradient) {
        gradient->clear();
        gradient->resize(PointDim());
        (*value) = 0;

        auto valueGpu = TStripeBuffer<float>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(1));
        auto der = TStripeBuffer<float>::CopyMappingAndColumnCount(Cursor);
        const ui32 cursorDim = Cursor.GetColumnCount();
        const ui32 rowSize = SingleBinDim();

        TStripeBuffer<float> der2;
        if (rowSize == 1) {
            //compute ders in one kernel call, fast path for pools with small number of features and many docs
            der2 = TStripeBuffer<float>::CopyMappingAndColumnCount(Cursor);
            DerCalcer->ApproximateAt(Cursor, &valueGpu, &der, &der2);
        } else {
            DerCalcer->ComputeValueAndDerivative(Cursor, &valueGpu, &der);
        }

        auto reducedDer = TStripeBuffer<double>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount * cursorDim));

        ComputePartitionStats(der, Offsets, &reducedDer);
        DerAtPoint = ReadReduce(reducedDer);

        if (rowSize == 1) {
            ComputePartitionStats(der2, Offsets, &reducedDer);
            Der2AtPoint = ReadReduce(reducedDer);
            const double lambda = LeavesEstimationConfig.Lambda;
            for (ui32 i = 0; i < Der2AtPoint->size(); ++i) {
                (*Der2AtPoint)[i] += lambda;
            }
        }

        if (DerCalcer->GetType() == ELossFunction::MultiClass) {
            for (ui32 bin = 0; bin < BinCount; ++bin) {
                double total = 0;
                for (ui32 dim = 0; dim < cursorDim; ++dim) {
                    const double val = (*DerAtPoint)[bin * cursorDim + dim];
                    (*gradient)[bin * rowSize + dim] = val;
                    total += val;
                }
                (*gradient)[bin * rowSize + cursorDim] = -total; //sum of der is equal to zero
            }
        } else {
            (*gradient) = *DerAtPoint;
        }

        (*value) = static_cast<float>(ReadReduce(valueGpu)[0]);

        //TODO(noxoomo): support it in multiclass
        if (rowSize == 1) {
            AddRigdeRegulaizationIfNecessary(LeavesEstimationConfig, CurrentPoint, value, gradient);
        }
    }

    void TBinOptimizedOracle::WriteSecondDerivatives(TVector<double>* secondDer) {
        const ui32 rowSize = SingleBinDim();
        const double lambda = LeavesEstimationConfig.Lambda;

        if (DerCalcer->GetType() == ELossFunction::MultiClass) {
            CB_ENSURE(DerAtPoint.Defined(), "Error: write der first");
        }

        CB_ENSURE(LeavesEstimationConfig.LeavesEstimationMethod != ELeavesEstimation::Exact);
        if (LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton) {
            if (Der2AtPoint) {
                (*secondDer) = *Der2AtPoint;
            } else {
                const ui32 hessianBlockSize = HessianBlockSize();
                const ui32 matrixSize = hessianBlockSize * hessianBlockSize;
                CB_ENSURE(rowSize % hessianBlockSize == 0,
                          "Error: rowSize % hessianBlockSize ≠ 0, this is a bug, report to catboost team");
                const ui32 blockCount = rowSize / hessianBlockSize;

                const ui32 singleBinBlockedMatrixSize = matrixSize * blockCount;
                secondDer->resize(singleBinBlockedMatrixSize * BinCount);

                const ui32 lowTriangleMatrixSize = hessianBlockSize * (hessianBlockSize + 1) / 2;
                auto reducedHessianGpu = TStripeBuffer<double>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(
                    lowTriangleMatrixSize * blockCount * BinCount));

                ui32 offset = 0;
                for (ui32 hessianBlockRow = 0; hessianBlockRow < hessianBlockSize; ++hessianBlockRow) {
                    const ui32 columnCount = hessianBlockRow + 1;
                    TStripeBuffer<float> der2Row = TStripeBuffer<float>::CopyMapping(Cursor, columnCount * blockCount);

                    DerCalcer->ComputeSecondDerRowLowerTriangleForAllBlocks(Cursor, hessianBlockRow, &der2Row);

                    auto writeSlice =
                        NCudaLib::ParallelStripeView(reducedHessianGpu, TSlice(offset * blockCount * BinCount,
                                                                               (offset + columnCount) * blockCount * BinCount));
                    ComputePartitionStats(der2Row, Offsets, &writeSlice);
                    offset += columnCount * blockCount;
                }
                CB_ENSURE(
                    offset == lowTriangleMatrixSize * blockCount,
                    "Unexpected offset " << offset << ", should be " << lowTriangleMatrixSize * blockCount);
                auto hessianCpu = ReadReduce(reducedHessianGpu);

                secondDer->clear();
                secondDer->resize(BinCount * singleBinBlockedMatrixSize);

                for (ui32 bin = 0; bin < BinCount; ++bin) {
                    for (ui32 blockId = 0; blockId < blockCount; ++blockId) {
                        double* sigma = secondDer->data() + bin * singleBinBlockedMatrixSize + blockId * matrixSize;

                        ui32 rowOffset = 0;
                        for (ui32 row = 0; row < hessianBlockSize; ++row) {
                            const ui32 columnCount = row + 1;
                            for (ui32 col = 0; col < row; ++col) {
                                const ui32 lowerIdx = row * hessianBlockSize + col;
                                const ui32 upperIdx = col * hessianBlockSize + row;

                                const double val = hessianCpu[rowOffset * BinCount + bin * columnCount * blockCount + blockId * columnCount + col];
                                sigma[lowerIdx] = val;
                                sigma[upperIdx] = val;
                            }
                            {
                                sigma[row * hessianBlockSize + row] =
                                    hessianCpu[rowOffset * BinCount + bin * columnCount * blockCount + blockId * columnCount + row] + lambda;
                            }
                            rowOffset += columnCount * blockCount;
                        }
                    }
                }
            }
        } else {
            secondDer->resize(rowSize * BinCount);

            for (ui32 bin = 0; bin < BinCount; ++bin) {
                const double w = WeightsCpu[bin];
                for (ui32 approx = 0; approx < rowSize; ++approx) {
                    (*secondDer)[rowSize * bin + approx] = w + lambda;
                }
            }
        }
    }

    void TBinOptimizedOracle::AddLangevinNoiseToDerivatives(TVector<double>* derivatives,
                                                            NPar::ILocalExecutor* localExecutor) {
        if (LeavesEstimationConfig.Langevin) {
            AddLangevinNoise(LeavesEstimationConfig, derivatives, localExecutor, Random.NextUniformL());
        }
    }

    TVector<float> TBinOptimizedOracle::EstimateExact() {
        auto values = TStripeBuffer<float>::CopyMapping(Bins);
        auto weights = TStripeBuffer<float>::CopyMapping(Bins);
        DerCalcer->ComputeExactValue(Cursor.AsConstBuf(), &values, &weights);

        TVector<float> point(BinCount * SingleBinDim());
        ComputeExactApprox(Bins, values, weights, BinCount, point, LeavesEstimationConfig.LossDescription);

        MoveTo(point);
        return MakeEstimationResult(point);
    }

    TBinOptimizedOracle::TBinOptimizedOracle(const TLeavesEstimationConfig& leavesEstimationConfig,
                                             THolder<IPermutationDerCalcer>&& derCalcer,
                                             TStripeBuffer<ui32>&& bins,
                                             TStripeBuffer<ui32>&& partOffsets,
                                             TStripeBuffer<float>&& cursor,
                                             ui32 binCount,
                                             TGpuAwareRandom& random)
        : LeavesEstimationConfig(leavesEstimationConfig)
        , DerCalcer(std::move(derCalcer))
        , Bins(std::move(bins))
        , Offsets(std::move(partOffsets))
        , Cursor(std::move(cursor))
        , BinCount(binCount)
        , Random(random)
    {
        ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
        for (ui32 dev = 0; dev < devCount; ++dev) {
            CB_ENSURE(
                Offsets.GetMapping().DeviceSlice(dev).Size() == binCount + 1,
                "Unexpected size of slice " << Offsets.GetMapping().DeviceSlice(dev).Size()
                << " at device " << dev << ", should be " << binCount + 1);
        }

        CurrentPoint.resize(BinCount * Cursor.GetColumnCount());

        auto weights = DerCalcer->GetWeights(0);
        auto reducedWeights = TStripeBuffer<double>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount));
        ComputePartitionStats(weights, Offsets, &reducedWeights);
        WeightsCpu = ReadReduce(reducedWeights);
    }

}
