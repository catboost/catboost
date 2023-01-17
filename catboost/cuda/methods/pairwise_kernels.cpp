#include "pairwise_kernels.h"

#include <util/generic/cast.h>

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0x421201, TMakeLinearSystemKernel);
    REGISTER_KERNEL(0x421202, TUpdateBinsPairsKernel);
    REGISTER_KERNEL(0x421203, TExtractMatricesAndTargetsKernel);
    REGISTER_KERNEL(0x421204, TRegularizeKernel);
    REGISTER_KERNEL(0x421205, TCholeskySolverKernel);
    REGISTER_KERNEL(0x421206, TCalcScoresKernel);
    REGISTER_KERNEL(0x421207, TCopyReducedTempResultKernel);
    REGISTER_KERNEL(0x421208, TSelectBestSplitKernel);
    REGISTER_KERNEL(0x421209, TComputePairwiseHistogramKernel);
    REGISTER_KERNEL(0x421210, TZeroMeanKernel);
    REGISTER_KERNEL(0x421211, TFillPairDer2OnlyKernel);
    REGISTER_KERNEL(0x421212, TFillPairBinsKernel);
    REGISTER_KERNEL(0x421213, TZeroSameLeafBinWeightsKernel);

}

void TMakeLinearSystemKernel::Run(const TCudaStream& stream) const {
    const ui32 systemSize = LinearSystem.ObjectSize();
    const ui32 rowSize = GetRowSizeFromLinearSystemSize(systemSize);
    CB_ENSURE(rowSize > 1, systemSize);
    const ui32 leavesCount = rowSize / 2;

    CB_ENSURE(systemSize == (rowSize + rowSize * (rowSize + 1) / 2));
    CB_ENSURE(BlockFeaturesSlice.Size() <= HistogramLineSize);
    CB_ENSURE(BlockFeaturesSlice.Size() == LinearSystem.ObjectCount());
    const bool useWeights = PointwiseHistogram.ObjectSize() == 2;

    NKernel::MakePointwiseDerivatives(PointwiseHistogram.Get(), HistogramLineSize,
                                      PartStats.Get(),
                                      useWeights,
                                      rowSize,
                                      BlockFeaturesSlice.Left, BlockFeaturesSlice.Size(),
                                      LinearSystem.Get(), stream.GetStream());

    NKernel::MakePairwiseDerivatives(PairwiseHistogram.Get(),
                                     leavesCount,
                                     BlockFeaturesSlice.Left, BlockFeaturesSlice.Size(),
                                     HistogramLineSize,
                                     LinearSystem.Get(),
                                     stream.GetStream());
}

void TExtractMatricesAndTargetsKernel::Run(const TCudaStream& stream) const {
    const ui32 rowSize = GetRowSizeFromLinearSystemSize(LinearSystem.ObjectSize());
    CB_ENSURE(Solutions.ObjectSize() == rowSize);
    const ui32 matricesCount = LinearSystem.ObjectCount();
    CB_ENSURE(matricesCount == SolutionsSlice.Size());
    float* firstSolution = Solutions.GetForObject(SolutionsSlice.Left);
    float* firstMatrixDiag = MatrixDiag.GetForObject(SolutionsSlice.Left);
    NKernel::ExtractMatricesAndTargets(LinearSystem.Get(),
                                       matricesCount,
                                       rowSize,
                                       LowTriangleMatrices.Get(),
                                       firstSolution,
                                       firstMatrixDiag,
                                       stream.GetStream());
}

void TZeroMeanKernel::Run(const TCudaStream& stream) const {
    const ui32 rowSize = Solutions.ObjectSize();
    NKernel::ZeroMean(Solutions.GetForObject(SolutionsSlice.Left), rowSize, SolutionsSlice.Size(), stream.GetStream());
}

THolder<TCholeskySolverKernel::TKernelContext> TCholeskySolverKernel::PrepareContext(IMemoryManager& manager) const {
    const ui32 rowSize = Solutions.ObjectSize();
    if (!TKernelContext::UseCuSolver(rowSize, Matrices.ObjectCount())) {
        return MakeHolder<TKernelContext>();
    }

    auto context = MakeHolder<TKernelContext>(rowSize);
    context->AllocateBuffers(manager);

    return context;
}

void TCholeskySolverKernel::Run(const TCudaStream& stream, TCholeskySolverKernel::TKernelContext& context) const {
    const ui32 rowSize = Solutions.ObjectSize();
    CB_ENSURE(rowSize * (rowSize + 1) / 2 == Matrices.ObjectSize());
    CB_ENSURE(Matrices.ObjectCount() == SolutionsSlice.Size());

    NKernel::CholeskySolver(Matrices.Get(),
                            Solutions.GetForObject(SolutionsSlice.Left),
                            rowSize,
                            static_cast<int>(SolutionsSlice.Size()),
                            RemoveLast,
                            context,
                            stream.GetStream());

    if (RemoveLast) {
        NKernel::ZeroMean(Solutions.GetForObject(SolutionsSlice.Left),
                          rowSize, SolutionsSlice.Size(),
                          stream.GetStream());
    }
}

void TCalcScoresKernel::Run(const TCudaStream& stream) const {
    const ui32 matrixCount = LinearSystem.ObjectCount();
    const ui32 rowSize = Solutions.ObjectSize();
    CB_ENSURE(BinFeaturesSlice.Size() == SolutionsSlice.Size());
    CB_ENSURE(matrixCount == SolutionsSlice.Size());

    NCudaLib::CopyMemoryAsync(BinFeaturesSource.GetForObject(BinFeaturesSlice.Left),
                              BinFeaturesResult.GetForObject(SolutionsSlice.Left),
                              SolutionsSlice.Size(),
                              stream);

    NKernel::CalcScores(LinearSystem.Get(),
                        Solutions.GetForObject(SolutionsSlice.Left),
                        Scores.GetForObject(SolutionsSlice.Left),
                        rowSize,
                        matrixCount,
                        stream.GetStream());
}

void TComputePairwiseHistogramKernel::Run(const TCudaStream& stream) const {
    if (Depth == 0) {
        CB_ENSURE(FullPass, "Depth 0 requires full pass");
    }
    const auto leavesCount = static_cast<ui32>(1u << Depth);
    const ui32 partCount = leavesCount * leavesCount;

#define DISPATCH(KernelName, FromBit, ToBit)                           \
    NKernel::KernelName(Features.Get(), FeaturesCpu.Get(),             \
                        static_cast<int>(Features.Size()),             \
                        FoldsHist.FeatureCountForBits(FromBit, ToBit), \
                        CompressedIndex.Get(),                         \
                        Pairs.Get(), SafeIntegerCast<ui32>(Pairs.Size()),\
                        Weight.Get(),                                  \
                        Partition.Get(),                               \
                        partCount,                                     \
                        HistLineSize,                                  \
                        FullPass,                                      \
                        Histogram.Get(),                               \
                        ParallelStreamsCount,                          \
                        stream.GetStream());

    if (Pairs.Size()) {
        //binary and halfByte are grouped by host
        //5-8 bits are not splitted in separate gropups and groups are skiped during kernel runtime
        switch (Policy) {
            case NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures: {
                DISPATCH(ComputePairwiseHistogramBinary, 0, 1)
                break;
            }
            case NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures: {
                DISPATCH(ComputePairwiseHistogramHalfByte, 1, 4)
                break;
            }
            case NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures: {
                DISPATCH(ComputePairwiseHistogramOneByte5Bits, 4, 5)
                DISPATCH(ComputePairwiseHistogramOneByte6Bits, 6, 6)
                DISPATCH(ComputePairwiseHistogramOneByte7Bits, 7, 7)
                DISPATCH(ComputePairwiseHistogramOneByte8BitAtomics, 8, 8)
                break;
            }
            default: {
                CB_ENSURE(false);
            }
        }
#undef DISPATCH
    }
    if (Policy != NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
        NKernel::ScanPairwiseHistograms(Features.Get(),
                                        Features.Size(),
                                        partCount,
                                        HistLineSize,
                                        FullPass,
                                        Histogram.Get(),
                                        stream.GetStream());

        NKernel::BuildBinaryFeatureHistograms(Features.Get(),
                                              Features.Size(),
                                              Partition.Get(),
                                              PartitionStats.Get(),
                                              partCount,
                                              HistLineSize,
                                              FullPass,
                                              Histogram.Get(),
                                              stream.GetStream());
    }

    if (!FullPass) {
        NKernel::UpdatePairwiseHistograms(BinFeaturesSlice.Left, BinFeaturesSlice.Size(),
                                          Partition.Get(), partCount,
                                          HistLineSize,
                                          Histogram.Get(),
                                          stream.GetStream());
    }
}
