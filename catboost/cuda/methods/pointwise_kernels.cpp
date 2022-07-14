#include "pointwise_kernels.h"
#include <catboost/cuda/gpu_data/grid_policy.h>

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0x420000, TComputeHist2Kernel);
    REGISTER_KERNEL(0x420001, TComputeHist1Kernel);

    REGISTER_KERNEL(0x420003, TUpdateFoldBinsKernel);
    REGISTER_KERNEL(0x420004, TUpdatePartitionPropsKernel);
    REGISTER_KERNEL(0x420005, TGatherHistogramByLeavesKernel);
    REGISTER_KERNEL(0x420006, TFindOptimalSplitKernel);

}

void TComputeHist2Kernel::Run(const TCudaStream& stream) const {
#define DISPATCH(KernelName)                               \
    NKernel::KernelName(Features.Get(),                    \
                        static_cast<int>(Features.Size()), \
                        Cindex.Get(),                      \
                        Target.Get(),                      \
                        Weight.Get(),                      \
                        Indices.Get(), Indices.Size(),     \
                        Partition.Get(),                   \
                        PartCount, FoldCount,              \
                        FullPass,                          \
                        HistLineSize,                      \
                        BinSums.Get(),                     \
                        stream.GetStream());

#define DISPATCH_ONE_BYTE(KernelName, FromBit, ToBit)                         \
    NKernel::KernelName<ToBit>(Features.Get(),                                \
                               static_cast<int>(Features.Size()),             \
                               Cindex.Get(),                                  \
                               Target.Get(),                                  \
                               Weight.Get(),                                  \
                               Indices.Get(), Indices.Size(),                 \
                               Partition.Get(),                               \
                               PartCount, FoldCount,                          \
                               FullPass,                                      \
                               HistLineSize,                                  \
                               BinSums.Get(),                                 \
                               FoldsHist.FeatureCountForBits(FromBit, ToBit), \
                               stream.GetStream());

    switch (Policy) {
        case NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures: {
            DISPATCH(ComputeHist2Binary)
            break;
        }
        case NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures: {
            DISPATCH(ComputeHist2HalfByte)
            break;
        }
        case NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures: {
            DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 4, 5)
            DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 6, 6)
            DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 7, 7)
            DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 8, 8)
            break;
        }
        default: {
            CB_ENSURE(false, "Unexpected feature grouping policy");
        }
    }
#undef DISPATCH
#undef DISPATCH_ONE_BYTE

    if (Policy != NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
        NKernel::ScanPointwiseHistograms(Features.Get(),
                                         Features.Size(),
                                         PartCount,
                                         FoldCount,
                                         HistLineSize,
                                         FullPass,
                                         2,
                                         BinSums.Get(),
                                         stream.GetStream());
    }

    if (!FullPass) {
        NKernel::UpdatePointwiseHistograms(BinSums.Get(),
                                           BinFeaturesSlice.Left,
                                           BinFeaturesSlice.Size(),
                                           PartCount,
                                           FoldCount,
                                           2,
                                           HistLineSize,
                                           Partition.Get(),
                                           stream.GetStream());
    }
}

void TComputeHist1Kernel::Run(const TCudaStream& stream) const {
#define DISPATCH(KernelName)                               \
    NKernel::KernelName(Features.Get(),                    \
                        static_cast<int>(Features.Size()), \
                        Cindex.Get(),                      \
                        Target.Get(),                      \
                        Indices.Get(), Indices.Size(),     \
                        Partition.Get(),                   \
                        PartCount,                         \
                        FoldCount,                         \
                        FullPass,                          \
                        HistLineSize,                      \
                        BinSums.Get(),                     \
                        stream.GetStream());

    switch (Policy) {
        case NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures: {
            DISPATCH(ComputeHist1Binary)
            break;
        }
        case NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures: {
            DISPATCH(ComputeHist1HalfByte)
            break;
        }
        case NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures: {
            DISPATCH(ComputeHist1NonBinary)
            break;
        }
        default: {
            CB_ENSURE(false, "Unexpected feature grouping policy");
        }
    }
#undef DISPATCH

    if (Policy != NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
        NKernel::ScanPointwiseHistograms(Features.Get(),
                                         Features.Size(),
                                         PartCount,
                                         FoldCount,
                                         HistLineSize,
                                         FullPass,
                                         1,
                                         BinSums.Get(),
                                         stream.GetStream());
    }

    if (!FullPass) {
        NKernel::UpdatePointwiseHistograms(BinSums.Get(),
                                           BinFeaturesSlice.Left,
                                           BinFeaturesSlice.Size(),
                                           PartCount,
                                           FoldCount,
                                           1,
                                           HistLineSize,
                                           Partition.Get(),
                                           stream.GetStream());
    }
}
