#pragma once

#include "ctr.h"

#include <catboost/cuda/ctrs/kernel/ctr_calcers.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/partitions.h>

namespace NKernelHost {
    class TUpdateBordersMaskKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<const ui32> PrevBins;
        TCudaBufferPtr<ui32> Indices;

    public:
        TUpdateBordersMaskKernel() = default;

        TUpdateBordersMaskKernel(TCudaBufferPtr<const ui32> bins,
                                 TCudaBufferPtr<const ui32> prevBins,
                                 TCudaBufferPtr<ui32> indices)
            : Bins(bins)
            , PrevBins(prevBins)
            , Indices(indices)
        {
        }

        Y_SAVELOAD_DEFINE(Bins, PrevBins, Indices);

        void Run(const TCudaStream& stream) const {
            NKernel::UpdateBordersMask(Bins.Get(), PrevBins.Get(), Indices.Get(),
                                       Indices.Size(), stream.GetStream());
        }
    };


    class TFixGroupwiseCtrKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> FixIndices;
        TCudaBufferPtr<float> Ctr;

    public:
        TFixGroupwiseCtrKernel() = default;

        TFixGroupwiseCtrKernel(TCudaBufferPtr<const ui32> fixIndices,
                             TCudaBufferPtr<float> ctr)
            : FixIndices(fixIndices)
            , Ctr(ctr)
        {
        }

        Y_SAVELOAD_DEFINE(FixIndices, Ctr);

        void Run(const TCudaStream& stream) const {
            NKernel::ApplyGroupwiseCtrFix(FixIndices.Size(),
                FixIndices.Get(),
                Ctr.Get(),
                stream.GetStream());
        }
    };

    class TMakeGroupStartFlagsKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const ui32> GroupIds;
        TCudaBufferPtr<ui32> Flags;
        ui32 Mask;

    public:
        TMakeGroupStartFlagsKernel() = default;

        TMakeGroupStartFlagsKernel(TCudaBufferPtr<const ui32> indices,
                               TCudaBufferPtr<const ui32> groupIds,
                               TCudaBufferPtr<ui32> flags,
                               ui32 mask)
            : Indices(indices)
            , GroupIds(groupIds)
            , Flags(flags)
            , Mask(mask)
        {
        }

        Y_SAVELOAD_DEFINE(Indices, GroupIds, Flags, Mask);

        void Run(const TCudaStream& stream) const {
            NKernel::MakeGroupStarts(Mask,
                Indices.Get(),
                GroupIds.Get(),
                Indices.Size(),
                Flags.Get(),
                stream.GetStream());
        }
    };



    class TFillBinIndicesKernel : public TStatelessKernel {
    private:
        ui32 Mask;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<ui32> BinIndices;

    public:
        TFillBinIndicesKernel() = default;

        TFillBinIndicesKernel(ui32 mask,
                              TCudaBufferPtr<const ui32> indices,
                              TCudaBufferPtr<const ui32> bins,
                              TCudaBufferPtr<ui32> binIndices)
            : Mask(mask)
            , Indices(indices)
            , Bins(bins)
            , BinIndices(binIndices)
        {
        }

        Y_SAVELOAD_DEFINE(Mask, Indices,  Bins, BinIndices);

        void Run(const TCudaStream& stream) const {
            NKernel::FillBinIndices(Mask,
                                    Indices.Get(),
                                    Bins.Get(),
                                    Indices.Size(),
                                    BinIndices.Get(),
                                    stream.GetStream());
        }
    };


    class TCreateFixedIndicesKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<const ui32> BinIndices;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<ui32> FixedIndices;
        ui32 Mask;

    public:
        TCreateFixedIndicesKernel () = default;

        TCreateFixedIndicesKernel (TCudaBufferPtr<const ui32> bins,
                                   TCudaBufferPtr<const ui32> binIndices,
                                   TCudaBufferPtr<const ui32> indices,
                                   TCudaBufferPtr<ui32> fixedIndices,
                                   ui32 mask)
            : Bins(bins)
            , BinIndices(binIndices)
            , Indices(indices)
            , FixedIndices(fixedIndices)
            , Mask(mask)
        {
        }

        Y_SAVELOAD_DEFINE(Bins, BinIndices, Indices, FixedIndices, Mask);

        void Run(const TCudaStream& stream) const {
            NKernel::CreateFixedIndices(Bins.Get(),
                                        BinIndices.Get(),
                                        Mask,
                                        Indices.Get(),
                                        Indices.Size(),
                                        FixedIndices.Get(),
                                        stream.GetStream());
        }
    };

    class TMergeBitsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui32> Bins;
        TCudaBufferPtr<const ui32> CurrentBins;
        ui32 Shift;

    public:
        TMergeBitsKernel() = default;

        TMergeBitsKernel(TCudaBufferPtr<ui32> bins,
                         TCudaBufferPtr<const ui32> current,
                         ui32 shift)
            : Bins(bins)
            , CurrentBins(current)
            , Shift(shift)
        {
        }

        Y_SAVELOAD_DEFINE(Bins, CurrentBins, Shift);

        void Run(const TCudaStream& stream) const {
            NKernel::MergeBinsKernel(Bins.Get(), CurrentBins.Get(), Shift, Bins.Size(), stream.GetStream());
        }
    };

    class TExtractBorderMasksKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<ui32> Dst;
        bool StartSegment;

    public:
        TExtractBorderMasksKernel() = default;

        TExtractBorderMasksKernel(TCudaBufferPtr<const ui32> indices,
                                  TCudaBufferPtr<ui32> dst, bool startSegment)
            : Indices(indices)
            , Dst(dst)
            , StartSegment(startSegment)
        {
        }

        Y_SAVELOAD_DEFINE(Indices, Dst, StartSegment);

        void Run(const TCudaStream& stream) const {
            NKernel::ExtractBorderMasks(Indices.Get(), Dst.Get(), Dst.Size(), StartSegment, stream.GetStream());
        }
    };

    class TFillBinarizedTargetsStatsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui8> Sample;
        TCudaBufferPtr<const float> SampleWeights;
        TCudaBufferPtr<float> Sums;
        ui32 BinIndex;
        bool Borders;

    public:
        TFillBinarizedTargetsStatsKernel() = default;

        TFillBinarizedTargetsStatsKernel(TCudaBufferPtr<const ui8> sample,
                                         TCudaBufferPtr<const float> sampleWeights,
                                         TCudaBufferPtr<float> sums, ui32 binIndex,
                                         bool borders)
            : Sample(sample)
            , SampleWeights(sampleWeights)
            , Sums(sums)
            , BinIndex(binIndex)
            , Borders(borders)
        {
        }

        Y_SAVELOAD_DEFINE(Sample, SampleWeights, Sums, BinIndex, Borders);

        void Run(const TCudaStream& stream) const {
            NKernel::FillBinarizedTargetsStats(Sample.Get(), SampleWeights.Get(), (ui32)SampleWeights.Size(),
                                               Sums.Get(), BinIndex, Borders, stream.GetStream());
        }
    };

    class TMakeMeanKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<float> Sums;
        TCudaBufferPtr<const float> Weights;
        float SumPrior;
        float WeightPrior;

    public:
        TMakeMeanKernel() = default;

        TMakeMeanKernel(TCudaBufferPtr<float> dst,
                        TCudaBufferPtr<const float> weights,
                        float sumPrior,
                        float weightPrior)
            : Sums(dst)
            , Weights(weights)
            , SumPrior(sumPrior)
            , WeightPrior(weightPrior)
        {
        }

        Y_SAVELOAD_DEFINE(Sums, Weights, SumPrior, WeightPrior);

        void Run(const TCudaStream& stream) const {
            NKernel::MakeMeans(Sums.Get(), Weights.Get(), (ui32)Sums.Size(), SumPrior, WeightPrior, stream.GetStream());
        }
    };

    class TMakeMeanAndScatterKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Sums;
        TCudaBufferPtr<const float> Weights;
        float SumPrior;
        float WeightPrior;
        TCudaBufferPtr<const ui32> Map;
        ui32 Mask;
        TCudaBufferPtr<float> Dst;

    public:
        TMakeMeanAndScatterKernel() = default;

        TMakeMeanAndScatterKernel(TCudaBufferPtr<const float> sums,
                                  TCudaBufferPtr<const float> weights,
                                  float sumPrior,
                                  float weightPrior,
                                  TCudaBufferPtr<const ui32> map,
                                  ui32 mask,
                                  TCudaBufferPtr<float> dst)
            : Sums(sums)
            , Weights(weights)
            , SumPrior(sumPrior)
            , WeightPrior(weightPrior)
            , Map(map)
            , Mask(mask)
            , Dst(dst)
        {
        }

        TMakeMeanAndScatterKernel(TCudaBufferPtr<const float> sums,
                                  TCudaBufferPtr<const float> weights,
                                  float sumPrior,
                                  float weightPrior,
                                  TCudaBufferPtr<float> dst)
            : Sums(sums)
            , Weights(weights)
            , SumPrior(sumPrior)
            , WeightPrior(weightPrior)
            , Mask(0)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(Sums, Weights, SumPrior, WeightPrior, Map, Mask, Dst);

        void Run(const TCudaStream& stream) const {
            NKernel::MakeMeansAndScatter(Sums.Get(), Weights.Get(), (ui32)Sums.Size(), SumPrior, WeightPrior,
                                         Map.Get(), Mask, Dst.Get(), stream.GetStream());
        }
    };

    class TComputeWeightedBinFreqCtrKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<const float> BinSums;
        float TotalWeight;
        float Prior;
        float PriorObservations;
        TCudaBufferPtr<float> Dst;

    public:
        TComputeWeightedBinFreqCtrKernel() = default;

        TComputeWeightedBinFreqCtrKernel(TCudaBufferPtr<const ui32> indices,
                                         TCudaBufferPtr<const ui32> bins,
                                         TCudaBufferPtr<const float> binSums,
                                         float totalWeight,
                                         float prior,
                                         float priorObservations,
                                         TCudaBufferPtr<float> dst)
            : Indices(indices)
            , Bins(bins)
            , BinSums(binSums)
            , TotalWeight(totalWeight)
            , Prior(prior)
            , PriorObservations(priorObservations)
            , Dst(dst)
        {
        }

        TComputeWeightedBinFreqCtrKernel(TCudaBufferPtr<const ui32> bins,
                                         TCudaBufferPtr<const float> binSums,
                                         float totalWeight,
                                         float prior,
                                         float priorObservations,
                                         TCudaBufferPtr<float> dst)
            : Bins(bins)
            , BinSums(binSums)
            , TotalWeight(totalWeight)
            , Prior(prior)
            , PriorObservations(priorObservations)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(Indices, Bins, BinSums, TotalWeight, Prior, PriorObservations, Dst);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeWeightedBinFreqCtr(Indices.Get(), Bins.Get(), BinSums.Get(),
                                               TotalWeight, Prior, PriorObservations, Dst.Get(),
                                               (ui32)Dst.Size(), stream.GetStream());
        }
    };

    class TComputeNonWeightedBinFreqCtrKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<const ui32> BinOffsets;
        float Prior;
        float PriorObservations;
        TCudaBufferPtr<float> Dst;

    public:
        TComputeNonWeightedBinFreqCtrKernel() = default;

        TComputeNonWeightedBinFreqCtrKernel(TCudaBufferPtr<const ui32> indices,
                                            TCudaBufferPtr<const ui32> bins,
                                            TCudaBufferPtr<const ui32> binOffsets,
                                            float prior,
                                            float priorObservations,
                                            TCudaBufferPtr<float> dst)
            : Indices(indices)
            , Bins(bins)
            , BinOffsets(binOffsets)
            , Prior(prior)
            , PriorObservations(priorObservations)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(Indices, Bins, BinOffsets, Prior, PriorObservations, Dst);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeNonWeightedBinFreqCtr(Indices.Get(), Bins.Get(), BinOffsets.Get(),
                                                  static_cast<ui32>(Bins.Size()),
                                                  Prior, PriorObservations, Dst.Get(),
                                                  stream.GetStream());
        }
    };

    class TGatherTrivialWeightsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<float> Dst;
        ui32 FirstZeroIndex;
        bool WriteSegmentStartFloatMask;

    public:
        TGatherTrivialWeightsKernel() = default;

        TGatherTrivialWeightsKernel(TCudaBufferPtr<const ui32> indices,
                                    TCudaBufferPtr<float> dst,
                                    ui32 firstZeroIndex,
                                    bool writeSegmentStartFloatMask)
            : Indices(indices)
            , Dst(dst)
            , FirstZeroIndex(firstZeroIndex)
            , WriteSegmentStartFloatMask(writeSegmentStartFloatMask)
        {
        }

        Y_SAVELOAD_DEFINE(Indices, FirstZeroIndex, Dst, WriteSegmentStartFloatMask);

        void Run(const TCudaStream& stream) const {
            NKernel::GatherTrivialWeights(Indices.Get(), Indices.Size(), FirstZeroIndex, WriteSegmentStartFloatMask, Dst.Get(), stream.GetStream());
        }
    };

    class TWriteMaskKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<float> Dst;

    public:
        TWriteMaskKernel() = default;

        TWriteMaskKernel(TCudaBufferPtr<const ui32> indices,
                         TCudaBufferPtr<float> dst)
            : Indices(indices)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(Indices, Dst);

        void Run(const TCudaStream& stream) const {
            NKernel::WriteMask(Indices.Get(), Indices.Size(), Dst.Get(), stream.GetStream());
        }
    };
}

template <class TMapping, class TUint32>
inline void GatherTrivialWeights(TCudaBuffer<float, TMapping>& dst,
                                 const TCudaBuffer<TUint32, TMapping>& indices,
                                 ui32 firstZeroIndex,
                                 bool writeSegmentStartFloatMask,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TGatherTrivialWeightsKernel;
    LaunchKernels<TKernel>(indices.NonEmptyDevices(), stream, indices, dst, firstZeroIndex,
                           writeSegmentStartFloatMask);
}

template <class TMapping, class TUint32>
inline void WriteFloatMask(const TCudaBuffer<TUint32, TMapping>& indices,
                           TCudaBuffer<float, TMapping>& dst,
                           ui32 stream = 0) {
    using TKernel = NKernelHost::TWriteMaskKernel;
    LaunchKernels<TKernel>(indices.NonEmptyDevices(), stream, indices, dst);
}

template <class TMapping, class TUint32>
inline void ExtractMask(const TCudaBuffer<TUint32, TMapping>& indices,
                        TCudaBuffer<ui32, TMapping>& dst,
                        bool startSegment = true, //mark segment start as 1; otherwise segment end
                        ui32 stream = 0) {
    using TKernel = NKernelHost::TExtractBorderMasksKernel;
    LaunchKernels<TKernel>(indices.NonEmptyDevices(), stream, indices, dst, startSegment);
}

template <class TMapping>
inline void UpdateBordersMask(const TCudaBuffer<ui32, TMapping>& bins, TCudaBuffer<ui32, TMapping>& indices, ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdateBordersMaskKernel;
    LaunchKernels<TKernel>(indices.NonEmptyDevices(), stream, bins, indices);
}

template <class TMapping>
inline void UpdateBordersMask(const TCudaBuffer<ui32, TMapping>& bins,
                              const TCudaBuffer<ui32, TMapping>& prevBins,
                              TCudaBuffer<ui32, TMapping>& indices, ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdateBordersMaskKernel;
    LaunchKernels<TKernel>(indices.NonEmptyDevices(), stream, bins, prevBins, indices);
}

template <class TMapping>
inline void UpdateCtrBins(TCudaBuffer<ui32, TMapping>& bins,
                          const TCudaBuffer<ui32, TMapping>& prevBins,
                          ui32 bits,
                          ui32 stream = 0) {
    using TKernel = NKernelHost::TMergeBitsKernel;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, prevBins, bits);
}

template <class TMapping, class TUi8>
inline void FillBinarizedTargetsStats(const TCudaBuffer<TUi8, TMapping>& target,
                                      const TCudaBuffer<float, TMapping>& weights,
                                      TCudaBuffer<float, TMapping>& dst,
                                      ui32 binIndex, ECtrType type, ui32 stream = 0) {
    CB_ENSURE(NCatboostCuda::IsBinarizedTargetCtr(type));
    using TKernel = NKernelHost::TFillBinarizedTargetsStatsKernel;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, target, weights, dst, binIndex, NCatboostCuda::IsBordersBasedCtr(type));
}

template <class TMapping>
inline void DivideWithPriors(TCudaBuffer<float, TMapping>& sums,
                             const TCudaBuffer<float, TMapping>& weights,
                             float sumPrior, float weightPrior, ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeMeanKernel;
    LaunchKernels<TKernel>(sums.NonEmptyDevices(), stream, sums, weights, sumPrior, weightPrior);
}

template <class TMapping>
inline void DivideWithPriorsAndScatter(const TCudaBuffer<float, TMapping>& sums,
                                       const TCudaBuffer<float, TMapping>& weights,
                                       float sumPrior, float weightPrior,
                                       const TCudaBuffer<const ui32, TMapping>& indices, ui32 mask,
                                       TCudaBuffer<float, TMapping>& dst,
                                       ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeMeanAndScatterKernel;
    LaunchKernels<TKernel>(sums.NonEmptyDevices(), stream, sums, weights, sumPrior, weightPrior, indices, mask,
                           dst);
}

template <class TMapping>
inline void DivideWithPriors(const TCudaBuffer<float, TMapping>& sums,
                             const TCudaBuffer<float, TMapping>& weights,
                             float sumPrior, float weightPrior,
                             TCudaBuffer<float, TMapping>& dst,
                             ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeMeanAndScatterKernel;
    LaunchKernels<TKernel>(sums.NonEmptyDevices(), stream, sums, weights, sumPrior, weightPrior, dst);
}

template <class TMapping, class TUi32>
inline void ComputeWeightedBinFreqCtr(const TCudaBuffer<TUi32, TMapping>& indices,
                                      const TCudaBuffer<ui32, TMapping>& bins,
                                      const TCudaBuffer<float, TMapping>& binWeights,
                                      float totalWeight,
                                      float prior,
                                      float priorObservations,
                                      TCudaBuffer<float, TMapping>& dst,
                                      ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeWeightedBinFreqCtrKernel;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, indices, bins, binWeights, totalWeight, prior,
                           priorObservations, dst);
}

template <class TMapping, class TUi32>
inline void ComputeNonWeightedBinFreqCtr(const TCudaBuffer<TUi32, TMapping>& indices,
                                         const TCudaBuffer<ui32, TMapping>& bins,
                                         const TCudaBuffer<ui32, TMapping>& offsets,
                                         float prior,
                                         float priorObservations,
                                         TCudaBuffer<float, TMapping>& dst,
                                         ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeNonWeightedBinFreqCtrKernel;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, indices, bins, offsets, prior, priorObservations, dst);
}

template <class TMapping, class TUi32>
inline void ComputeBinFreqCtr(const TCudaBuffer<ui32, TMapping>& bins,
                              const TCudaBuffer<float, TMapping>& binWeights,
                              float totalWeight,
                              float prior,
                              float priorObservations,
                              TCudaBuffer<float, TMapping>& dst,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeWeightedBinFreqCtrKernel;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, bins, binWeights, totalWeight, prior, priorObservations,
                           dst);
}


template <class TMapping>
inline void ApplyFixForGroupwiseCtr(const TCudaBuffer<ui32, TMapping>& fixIndices,
                                    TCudaBuffer<float, TMapping>& ctr,
                                    ui32 stream = 0) {
    using TKernel = NKernelHost::TFixGroupwiseCtrKernel;
    LaunchKernels<TKernel>(fixIndices.NonEmptyDevices(), stream, fixIndices, ctr);
}


template <class TMapping>
inline void MakeGroupStartFlags(const TCudaBuffer<const ui32, TMapping>& indices,
                                const TCudaBuffer<const ui32, TMapping>& groupIds,
                                TCudaBuffer<ui32, TMapping>* flags,
                                ui32 mask,
                                ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeGroupStartFlagsKernel;
    LaunchKernels<TKernel>(flags->NonEmptyDevices(), stream, indices, groupIds, *flags, mask);
}



template <class TMapping>
inline void FillBinIndices(ui32 mask, const TCudaBuffer<const ui32, TMapping>& indices,
                           const TCudaBuffer<ui32, TMapping>& bins,
                           TCudaBuffer<ui32, TMapping>* binIndices,
                           ui32 stream = 0) {
    using TKernel = NKernelHost::TFillBinIndicesKernel;
    LaunchKernels<TKernel>(binIndices->NonEmptyDevices(), stream, mask, indices, bins, *binIndices);
}


template <class TMapping>
inline void CreateFixedIndices(const TCudaBuffer<ui32, TMapping>& bins,
                               const TCudaBuffer<ui32, TMapping>& binIndices,
                               const TCudaBuffer<const ui32, TMapping>& indices,
                               ui32 mask,
                               TCudaBuffer<ui32, TMapping>* fixedIndices,
                               ui32 stream = 0) {
    using TKernel = NKernelHost::TCreateFixedIndicesKernel;
    LaunchKernels<TKernel>(fixedIndices->NonEmptyDevices(), stream, bins, binIndices, indices, *fixedIndices, mask);
}
