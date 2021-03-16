#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/methods/kernel/exact_estimation.cuh>

namespace NKernelHost {

    class TComputeNeedWeightsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Targets;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const ui32> Offsets;
        TCudaBufferPtr<float> NeedWeights;
        float Alpha;

    public:
        TComputeNeedWeightsKernel() = default;

        TComputeNeedWeightsKernel(TCudaBufferPtr<const float> targets,
                                  TCudaBufferPtr<const float> weights,
                                  TCudaBufferPtr<const ui32> offsets,
                                  TCudaBufferPtr<float> needWeights,
                                  float alpha)
                : Targets(targets)
                , Weights(weights)
                , Offsets(offsets)
                , NeedWeights(needWeights)
                , Alpha(alpha)
        {
        }

        Y_SAVELOAD_DEFINE(Targets, Weights, Offsets, NeedWeights, Alpha);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeNeedWeights(Targets.Get(),
                                        Weights.Get(),
                                        Targets.Size(),
                                        NeedWeights.Size(),
                                        Offsets.Get(),
                                        Offsets.Get() + 1,
                                        NeedWeights.Get(),
                                        Alpha,
                                        stream.GetStream());
        }
    };

    class TComputeWeightsWithTargetsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Targets;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<float> WeightsWithTargets;

    public:
        TComputeWeightsWithTargetsKernel() = default;

        TComputeWeightsWithTargetsKernel(TCudaBufferPtr<const float> targets,
                                         TCudaBufferPtr<const float> weights,
                                         TCudaBufferPtr<float> weightsWithTargets)
                : Targets(targets)
                , Weights(weights)
                , WeightsWithTargets(weightsWithTargets)
        {
        }

        Y_SAVELOAD_DEFINE(Targets, Weights, WeightsWithTargets);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeWeightsWithTargets(Targets.Get(),
                                               Weights.Get(),
                                               WeightsWithTargets.Get(),
                                               Targets.Size(),
                                               stream.GetStream());
        }
    };

    class TComputeWeightedQuantileWithBinarySearchKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Targets;
        TCudaBufferPtr<const float> WeightsPrefixSum;
        TCudaBufferPtr<const float> NeedWeights;
        TCudaBufferPtr<const ui32> Offsets;
        ui32 BinCount;
        TCudaBufferPtr<float> Point;
        float Alpha;
        ui32 BinarySearchIterations;

    public:
        TComputeWeightedQuantileWithBinarySearchKernel() = default;

        TComputeWeightedQuantileWithBinarySearchKernel(TCudaBufferPtr<const float> targets,
                                                       TCudaBufferPtr<const float> weightsPrefixSum,
                                                       TCudaBufferPtr<const float> needWeights,
                                                       TCudaBufferPtr<const ui32> offsets,
                                                       ui32 binCount,
                                                       TCudaBufferPtr<float> point,
                                                       float alpha,
                                                       ui32 binarySearchIterations)
                : Targets(targets)
                , WeightsPrefixSum(weightsPrefixSum)
                , NeedWeights(needWeights)
                , Offsets(offsets)
                , BinCount(binCount)
                , Point(point)
                , Alpha(alpha)
                , BinarySearchIterations(binarySearchIterations)
        {
        }

        Y_SAVELOAD_DEFINE(Targets, WeightsPrefixSum, NeedWeights, Offsets, Point, BinCount, BinarySearchIterations, Alpha);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeWeightedQuantileWithBinarySearch(Targets.Get(),
                                                             WeightsPrefixSum.Get(),
                                                             Targets.Size(),
                                                             NeedWeights.Get(),
                                                             Offsets.Get(),
                                                             Offsets.Get() + 1,
                                                             BinCount,
                                                             Point.Get(),
                                                             Alpha,
                                                             BinarySearchIterations,
                                                             stream.GetStream());
        }
    };

    class TMakeEndOfBinsFlagsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Offsets;
        TCudaBufferPtr<ui32> Flags;
        ui32 BinCount;

    public:
        TMakeEndOfBinsFlagsKernel() = default;

        TMakeEndOfBinsFlagsKernel(TCudaBufferPtr<const ui32> offsets,
                                  TCudaBufferPtr<ui32> flags,
                                  ui32 binCount)
                : Offsets(offsets)
                , Flags(flags)
                , BinCount(binCount)
        {
        }

        Y_SAVELOAD_DEFINE(Offsets, Flags, BinCount);

        void Run(const TCudaStream& stream) const {
            NKernel::MakeEndOfBinsFlags(Offsets.Get(),
                                        Offsets.Get() + 1,
                                        BinCount,
                                        Flags.Get(),
                                        stream.GetStream());
        }
    };
}

template <class TMapping>
inline void ComputeNeedWeights(TCudaBuffer<float, TMapping>& targets,
                               TCudaBuffer<float, TMapping>& weights,
                               TCudaBuffer<ui32, TMapping>& offsets,
                               TCudaBuffer<float, TMapping>* needWeights,
                               float alpha,
                               ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeNeedWeightsKernel;
    LaunchKernels<TKernel>(targets.NonEmptyDevices(), stream, targets, weights, offsets, needWeights, alpha);
}

template <class TMapping>
inline void ComputeWeightsWithTargets(const TCudaBuffer<float, TMapping>& targets,
                                      const TCudaBuffer<float, TMapping>& weights,
                                      TCudaBuffer<float, TMapping>* weightsWithTargets,
                                      ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeWeightsWithTargetsKernel;
    LaunchKernels<TKernel>(targets.NonEmptyDevices(), stream, targets, weights, weightsWithTargets);
}

template <class TDataMapping, class TPointMapping>
inline void CalculateQuantileWithBinarySearch(const TCudaBuffer<float, TDataMapping>& targets,
                                              const TCudaBuffer<float, TDataMapping>& weightsPrefixSum,
                                              const TCudaBuffer<float, TPointMapping>& needWeights,
                                              const TCudaBuffer<ui32, TDataMapping>& offsets,
                                              ui32 binCount,
                                              TCudaBuffer<float, TPointMapping>* point,
                                              float alpha,
                                              ui32 binarySearchIterations,
                                              ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeWeightedQuantileWithBinarySearchKernel;
    LaunchKernels<TKernel>(targets.NonEmptyDevices(), stream, targets, weightsPrefixSum, needWeights, offsets, binCount, point, alpha, binarySearchIterations);
}

template <class TMapping>
inline void MakeEndOfBinsFlags(const TCudaBuffer<ui32, TMapping>& offsets,
                               TCudaBuffer<ui32, TMapping>* flags,
                               ui32 binCount,
                               ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeEndOfBinsFlagsKernel;
    LaunchKernels<TKernel>(offsets.NonEmptyDevices(), stream, offsets, flags, binCount);
}
