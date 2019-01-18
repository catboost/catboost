#pragma once

#include "add_doc_parallel.h"
#include "non_symmetric_tree.h"
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>

namespace NCatboostCuda {
    template <>
    class TAddModelDocParallel<TNonSymmetricTree> {
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

        TAddModelDocParallel& AddTask(const TNonSymmetricTree& model,
                                      const TDataSet& dataSet,
                                      TStripeBuffer<float>& cursor);

        void Proceed();

    private:
        TVector<TComputationStream> Streams;
        const TCompressedIndex* CompressedIndex = nullptr;

        TVector<TNonSymmetricTreeStructure> Structures;
        TVector<TVec*> Cursors;
        TVector<TSlice> LeavesSlices;
        TVector<float> CpuLeaves;
        TVector<TStripeBuffer<ui32>> TempBins;
        TVector<const TDataSet*> DataSets;
    };

    void ComputeBinsForModel(const TNonSymmetricTreeStructure& structure,
                             const TDocParallelDataSet& dataSet,
                             TStripeBuffer<ui32>* bins);
}
