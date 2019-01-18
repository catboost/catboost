#pragma once

#include "pairwise_optimization_subsets.h"
#include "blocked_histogram_helper.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/reduce_scatter.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/cuda/methods/pointwise_kernels.h>
#include <catboost/cuda/methods/histograms_helper.h>

#include <util/stream/labeled.h>

namespace NCatboostCuda {
    struct TBinaryFeatureSplitResults {
        TStripeBuffer<TCBinFeature> BinFeatures;
        TStripeBuffer<float> Scores;
        TStripeBuffer<float> Solutions;
        TStripeBuffer<float> MatrixDiagonal;

        //optional, will be used for tests only
        THolder<TStripeBuffer<float>> LinearSystems;
        THolder<TStripeBuffer<float>> SqrtMatrices;

        void ReadBestSolution(ui32 idx,
                              TVector<float>* resultPtr,
                              TVector<float>* matrixDiagonal) {
            const ui32 rowSize = Solutions.GetMapping().SingleObjectSize();
            auto& result = *resultPtr;

            Solutions
                .CreateReader()
                .SetReadSlice((TSlice(idx, (idx + 1))))
                .Read(result);

            MatrixDiagonal
                .CreateReader()
                .SetReadSlice((TSlice(idx, (idx + 1))))
                .Read(*matrixDiagonal);

            CB_ENSURE(result.size() == rowSize, LabeledOutput(result.size(), rowSize));
        }
    };

    class TComputePairwiseScoresHelper: public TMoveOnly, public TGuidHolder {
    private:
        struct TTempData {
            TVector<TStripeBuffer<float>> LinearSystems;
            TVector<TStripeBuffer<float>> SqrtMatrices;
        };

        using TGpuDataSet = typename TSharedCompressedIndex<TDocParallelLayout>::TCompressedDataSet;
        using TFeaturesMapping = typename TDocParallelLayout::TFeaturesMapping;
        using TSamplesMapping = typename TDocParallelLayout::TSamplesMapping;

    public:
        TComputePairwiseScoresHelper(EFeaturesGroupingPolicy policy,
                                     const TGpuDataSet& dataSet,
                                     const TPairwiseOptimizationSubsets& subsets,
                                     TRandom& random,
                                     ui32 maxDepth,
                                     double l2Reg,
                                     double nonDiagReg,
                                     double rsm);

        TComputePairwiseScoresHelper& Compute(TScopedCacheHolder& scoresCacheHolder,
                                              TBinaryFeatureSplitResults* result);

        void EnsureCompute() {
            if (!Synchronized) {
                for (const auto& stream : Streams) {
                    stream.Synchronize();
                }
                Synchronized = true;
            }
        };

    private:
        void SampleFeatures(TRandom& random, double rsm);
        void ResetHistograms();
        TMirrorBuffer<const TCBinFeature>& GetBinaryFeatures() const;
        const TStripeBuffer<TCFeature>& GetGpuFeaturesBuffer() const;
        const TCpuGrid& GetCpuGrid() const;
        TCudaBuffer<const TCFeature, TFeaturesMapping, NCudaLib::EPtrType::CudaHost>& GetCpuFeatureBuffer() const;
        void ValidateSampledGrid() const;

    private:
        EFeaturesGroupingPolicy Policy;
        const TGpuDataSet& DataSet;
        const TPairwiseOptimizationSubsets& Subsets;
        ui32 MaxDepth;
        int CurrentBit = -1;
        bool BuildFromScratch = true;
        //if we need to add pointwise der2 (or weights): need for llmax-like targets
        bool NeedPointwiseWeights = false;
        double LambdaDiag = 0.0;
        double LambdaNonDiag = 0.1;

        TCudaBuffer<float, TFeaturesMapping> PairwiseHistograms;
        TCudaBuffer<float, TFeaturesMapping> PointwiseHistograms;

        //memory used by one part. parts are accessed via partId * partSize + binFeatureId * histCount
        ui64 HistogramLineSize = 0;

        TVector<TComputationStream> Streams;
        bool Synchronized = true;

        //if rsm < 1, will be generated for current pass
        bool IsSampledGrid = false;
        TMaybe<TCpuGrid> CpuGrid;
        TMaybe<TStripeBuffer<TCFeature>> GpuGrid;
        TMaybe<TVector<TCBinFeature>> BinFeaturesCpu;

        const ui32 MaxStreamCount = 8;

        mutable TScopedCacheHolder CacheHolder;
    };

}
