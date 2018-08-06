#include "pointwise_oracle.h"
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>


namespace NCatboostCuda {

    void TBinOptimizedOracle::WriteWeights(TVector<double>* dst) {
        (*dst) = WeightsCpu;
    }

    void TBinOptimizedOracle::Regularize(TVector<float>* point) {
        const ui32 approxDim = SingleBinDim();
        RegulalizeImpl(LeavesEstimationConfig, WeightsCpu, point, approxDim);

    }


    TVector<float> TBinOptimizedOracle::MakeEstimationResult(const TVector<float>& point) const {
        const ui32 cursorDim = static_cast<const ui32>(Cursor.GetColumnCount());

        TVector<float> newPoint;
        if (MulticlassLastDimHack) {
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
        CurrentPoint = newPoint;

    }

    void TBinOptimizedOracle::WriteValueAndFirstDerivatives(double* value,
                                                            TVector<double>* gradient) {

        gradient->clear();
        gradient->resize(PointDim());
        (*value) = 0;

        auto valueGpu = TStripeBuffer<float>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(1));
        auto der = TStripeBuffer<float>::CopyMappingAndColumnCount(Cursor);

        DerCalcer->ComputeValueAndDerivative(Cursor, &valueGpu, &der);

        const ui32 cursorDim = Cursor.GetColumnCount();
        auto reducedDer = TStripeBuffer<double>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount * cursorDim));

        ComputePartitionStats(der, Offsets, &reducedDer);
        DerAtPoint = ReadReduce(reducedDer);
        const ui32 rowSize = SingleBinDim();

        if (MulticlassLastDimHack) {
            for (ui32 bin = 0; bin < BinCount; ++bin) {
                double total = 0;
                for (ui32 dim = 0; dim < cursorDim; ++dim) {
                    const double val = (*DerAtPoint)[bin * cursorDim + dim];
                    (*gradient)[bin * rowSize + dim] = val;
                    total += val;
                }
                (*gradient)[bin * rowSize + cursorDim] = -total;//sum of der is equal to zero
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

        if (MulticlassLastDimHack) {
            CB_ENSURE(DerAtPoint.Defined(), "Error: write der first");
        }

        if (LeavesEstimationConfig.UseNewton) {
            const ui32 rowCount = HessianBlockSize();
            const ui32 matrixSize = rowCount * rowCount;
            secondDer->resize(matrixSize * BinCount);

            const ui32 lowTriangleMatrixSize = rowCount * (rowCount + 1) / 2;
            auto reducedHessianGpu = TStripeBuffer<double>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(lowTriangleMatrixSize * BinCount));


            ui32 offset = 0;
            for (ui32 hessianRow = 0; hessianRow < rowCount; ++hessianRow) {
                const ui32 columnCount = hessianRow + 1;
                TStripeBuffer<float> der2Row = TStripeBuffer<float>::CopyMapping(Cursor, columnCount);

                DerCalcer->ComputeSecondDerRowLowerTriangle(Cursor, hessianRow, &der2Row);
                auto writeSlice = NCudaLib::ParallelStripeView(reducedHessianGpu, TSlice(offset * BinCount,
                                                                                         (offset + columnCount) * BinCount));
                ComputePartitionStats(der2Row, Offsets, &writeSlice);
                offset += columnCount;
            }
            Y_VERIFY(offset == lowTriangleMatrixSize);
            auto hessianCpu = ReadReduce(reducedHessianGpu);

            secondDer->clear();
            secondDer->resize(BinCount * matrixSize);

            for (ui32 bin = 0; bin < BinCount; ++bin) {
                double* sigma = secondDer->data() + bin * matrixSize;
                ui32 rowOffset = 0;
                double der2Last = 0;
                for (ui32 row = 0; row < rowCount; ++row) {
                    const ui32 columnCount = row + 1;
                    for (ui32 col = 0; col < row; ++col) {
                        const ui32 lowerIdx = row * rowCount + col;
                        const ui32 upperIdx = col * rowCount + row;

                        const double val = hessianCpu[rowOffset * BinCount + bin * columnCount + col];
                        sigma[lowerIdx] = val;
                        sigma[upperIdx] = val;
                        der2Last += 2 * val;
                    }
                    {
                        sigma[row * rowCount + row] = hessianCpu[rowOffset * BinCount + bin * columnCount + row] + lambda;
                    }
                    rowOffset += columnCount;
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

    TBinOptimizedOracle::TBinOptimizedOracle(const TLeavesEstimationConfig& leavesEstimationConfig,
                                             bool multiclassLastDimHack,
                                             THolder<IPermutationDerCalcer>&& derCalcer,
                                             TStripeBuffer<ui32>&& bins,
                                             TStripeBuffer<ui32>&& partOffsets,
                                             TStripeBuffer<float>&& cursor,
                                             ui32 binCount)
            : LeavesEstimationConfig(leavesEstimationConfig)
            , MulticlassLastDimHack(multiclassLastDimHack)
              , DerCalcer(std::move(derCalcer))
              , Bins(std::move(bins))
              , Offsets(std::move(partOffsets))
              , Cursor(std::move(cursor))
              , BinCount(binCount) {

        ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
        for (ui32 dev = 0; dev < devCount; ++dev) {
            Y_VERIFY(Offsets.GetMapping().DeviceSlice(dev).Size() == binCount + 1);
        }

        CurrentPoint.resize(BinCount * Cursor.GetColumnCount());

        auto weights = DerCalcer->GetWeights(0);
        auto reducedWeights = TStripeBuffer<double>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(BinCount));
        ComputePartitionStats(weights, Offsets, &reducedWeights);
        WeightsCpu = ReadReduce(reducedWeights);
    }


}
