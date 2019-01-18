#pragma once

#include "add_doc_parallel.h"
#include "region_model.h"
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>

namespace NCatboostCuda {
    template <>
    class TAddModelDocParallel<TRegionModel> {
    public:
        using TVec = TStripeBuffer<float>;
        using TDataSet = TDocParallelDataSet;
        using TCompressedIndex = typename TDataSet::TCompressedIndex;

        TAddModelDocParallel(bool useStreams = false) {
            if (useStreams) {
                const ui32 streamCount = 2;
                for (ui32 i = 0; i < streamCount; ++i) {
                    Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                }
            }
        }

        TAddModelDocParallel& AddTask(const TRegionModel& model,
                                      const TDataSet& dataSet,
                                      TStripeBuffer<float>& cursor);

        void Proceed();

    private:
        void Append(ui32 taskId,
                    const TStripeBuffer<TCFeature>& features,
                    const TMirrorBuffer<TRegionDirection>& directions,
                    const TMirrorBuffer<float>& values,
                    ui32 stream);

    private:
        TVector<TComputationStream> Streams;
        const TCompressedIndex* CompressedIndex = nullptr;

        TVector<TVec*> Cursors;

        TVector<TSlice> LeavesSlices;
        TVector<TSlice> FeaturesSlices;
        TVector<float> CpuLeaves;
        TVector<TRegionDirection> TreeNodes;

        NCudaLib::TParallelStripeVectorBuilder<TCFeature> FeaturesBuilder;
    };

    void ComputeBinsForModel(const TRegionStructure& structure,
                             const TDocParallelDataSet& dataSet,
                             TStripeBuffer<ui32>* bins);
}
