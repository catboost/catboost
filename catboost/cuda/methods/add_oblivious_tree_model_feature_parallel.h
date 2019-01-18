#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>

namespace NCatboostCuda {
    class TAddObliviousTreeFeatureParallel {
    public:
        using TDataSet = TFeatureParallelDataSet;

        TAddObliviousTreeFeatureParallel(TScopedCacheHolder& cacheHolder,
                                         const TBinarizedFeaturesManager& featuresManager,
                                         const TObliviousTreeStructure& modelStructure)
            : CacheHolder(cacheHolder)
            , FeaturesManager(featuresManager)
            , ModelStructure(modelStructure)
        {
            const ui32 streamCount = 2;
            for (ui32 i = 0; i < streamCount; ++i) {
                Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
            }
        }

        TAddObliviousTreeFeatureParallel& Append(const TObliviousTreeModel& model,
                                                 const TDataSet& dataSet,
                                                 const TMirrorBuffer<const ui32>& indices,
                                                 TMirrorBuffer<float>& cursor);

        TAddObliviousTreeFeatureParallel& AddTask(const TObliviousTreeModel& model,
                                                  const TDataSet& dataSet,
                                                  TMirrorBuffer<const ui32>&& indices,
                                                  TMirrorBuffer<float>& cursor);

        void Proceed();

    private:
        struct TAddModelTask {
            TMirrorBuffer<const ui32> Indices;
            TMirrorBuffer<float>* Cursor;
            const TDataSet* DataSet;
        };

        void Append(ui32 taskId,
                    const TMirrorBuffer<float>& values,
                    ui32 stream);

    private:
        TVector<TComputationStream> Streams;
        TVector<TAddModelTask> Tasks;
        TScopedCacheHolder& CacheHolder;
        const TBinarizedFeaturesManager& FeaturesManager;
        TObliviousTreeStructure ModelStructure;
        TVector<float> CpuLeaves;
    };

}
