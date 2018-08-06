#pragma once

#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>

namespace NCatboostCuda {

    class TAddDocParallelObliviousTree {
    public:
        using TVec = TStripeBuffer<float>;
        using TDataSet = TDocParallelDataSet;
        using TCompressedIndex = typename TDataSet::TCompressedIndex;

        TAddDocParallelObliviousTree(bool useStreams = false) {
            if (useStreams) {
                const ui32 streamCount = 2;
                for (ui32 i = 0; i < streamCount; ++i) {
                    Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                }
            }
        }

        TAddDocParallelObliviousTree& AddTask(const TObliviousTreeModel& model,
                                              const TDataSet& dataSet,
                                              TStripeBuffer<float>& cursor);

        void Proceed();

    private:
        void Append(ui32 taskId,
                    const TStripeBuffer<TCFeature>& features,
                    const TMirrorBuffer<ui8>& bins,
                    const TMirrorBuffer<float>& values,
                    ui32 stream);

    private:
        TVector<TComputationStream> Streams;
        const TCompressedIndex* CompressedIndex = nullptr;

        TVector<TVec*> Cursors;

        TVector<TSlice> LeavesSlices;
        TVector<TSlice> FeaturesSlices;
        TVector<float> CpuLeaves;
        TVector<ui8> FeatureBins;
        NCudaLib::TStripeVectorBuilder<TCFeature> FeaturesBuilder;
    };



    void ComputeBinsForModel(const TObliviousTreeStructure& structure,
                             const TDocParallelDataSet& dataSet,
                             TStripeBuffer<ui32>* bins);
}
