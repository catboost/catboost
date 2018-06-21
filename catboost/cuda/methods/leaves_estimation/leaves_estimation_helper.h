#pragma once

#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/reduce.h>

namespace NCatboostCuda {

    struct TEstimationTaskHelper {
        THolder<IPermutationDerCalcer> DerCalcer;

        TStripeBuffer<ui32> Bins;
        TStripeBuffer<ui32> Offsets;

        TStripeBuffer<float> Baseline;
        TStripeBuffer<float> Cursor;

        TStripeBuffer<float> TmpDer;
        TStripeBuffer<float> TmpValue;
        TStripeBuffer<float> TmpDer2;

        TEstimationTaskHelper() = default;

        void MoveToPoint(const TMirrorBuffer<float>& point,
                         ui32 stream = 0);

        template <NCudaLib::EPtrType Type>
        void ProjectWeights(TCudaBuffer<float, NCudaLib::TStripeMapping, Type>& weightsDst,
                            ui32 streamId = 0) {
            SegmentedReduceVector(DerCalcer->GetWeights(streamId),
                                  Offsets,
                                  weightsDst,
                                  EOperatorType::Sum,
                                  streamId);
        }

        template <NCudaLib::EPtrType PtrType>
        void Project(TCudaBuffer<float, NCudaLib::TStripeMapping, PtrType>* value,
                     TCudaBuffer<float, NCudaLib::TStripeMapping, PtrType>* der,
                     TCudaBuffer<float, NCudaLib::TStripeMapping, PtrType>* der2,
                     ui32 stream = 0) {
            if (value) {
                TmpValue.Reset(Cursor.GetMapping().Transform([&](const TSlice&) -> ui64 {
                    return 1;
                }));
            }
            if (der) {
                TmpDer.Reset(Cursor.GetMapping());
            }
            if (der2) {
                TmpDer2.Reset(Cursor.GetMapping());
            }

            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            DerCalcer->ApproximateAt(Cursor,
                                     value ? &TmpValue : nullptr,
                                     der ? &TmpDer : nullptr,
                                     der2 ? &TmpDer2 : nullptr,
                                     stream);
            if (value) {
                value->Copy(TmpValue, stream);
            }
            {
                auto guard = profiler.Profile("Segmented reduce derivatives");
                if (der) {
                    SegmentedReduceVector(TmpDer, Offsets, *der, EOperatorType::Sum, stream);
                }
                if (der2) {
                    SegmentedReduceVector(TmpDer2, Offsets, *der2, EOperatorType::Sum, stream);
                }
            }
        }
    };

    void ReorderPairs(TStripeBuffer<ui32>* pairBins,
                      ui32 binCount,
                      TStripeBuffer<uint2>* pairs,
                      TStripeBuffer<float>* pairWeights);

    TStripeBuffer<ui32> ComputeBinOffsets(const TStripeBuffer<ui32>& sortedBins,
                                          ui32 binCount);

    void FilterZeroLeafBins(const TStripeBuffer<const ui32>& bins,
                            TStripeBuffer<uint2>* pairs,
                            TStripeBuffer<float>* pairWeights);

    TVector<float> ComputeBinStatisticsForParts(const TStripeBuffer<float>& stat,
                                                const TStripeBuffer<ui32>& partOffsets,
                                                ui32 partCount);

    void ComputeByLeafOrder(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                            TStripeBuffer<ui32>* binOffsets,
                            TStripeBuffer<ui32>* indices);

    void MakeSupportPairsMatrix(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                                TStripeBuffer<uint2>* pairs,
                                TStripeBuffer<float>* pairWeights,
                                TStripeBuffer<ui32>* pairPartOffsets,
                                TVector<float>* partLeafWeights);

    void MakePointwiseComputeOrder(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                                   const TStripeBuffer<const float>& weights,
                                   TStripeBuffer<ui32>* orderByPart,
                                   TStripeBuffer<ui32>* partOffsets,
                                   TVector<float>* pointLeafWeights);

}
