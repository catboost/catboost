#pragma once

#include <catboost/cuda/models/add_model.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/methods/helpers.h>

namespace NCatboostCuda {
    template <NCudaLib::EPtrType CatFeatureStorageType>
    class TAddModelValue<TObliviousTreeModel, TFeatureParallelDataSet<CatFeatureStorageType>> {
    public:
        using TDataSet = TFeatureParallelDataSet<CatFeatureStorageType>;

        TAddModelValue(TScopedCacheHolder& cacheHolder,
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

        TAddModelValue& Append(const TObliviousTreeModel& model,
                               const TDataSet& dataSet,
                               const TMirrorBuffer<const ui32>& indices,
                               TMirrorBuffer<float>& cursor) {
            Y_ASSERT(model.GetStructure() == ModelStructure);

            const auto& bins = GetBinsForModel(CacheHolder,
                                               FeaturesManager,
                                               dataSet,
                                               model.GetStructure());

            auto gpuValues = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(model.GetValues().size()));
            gpuValues.Write(model.GetValues());

            AddBinModelValues<NCudaLib::TMirrorMapping>(gpuValues,
                                                        bins,
                                                        indices /*read indices*/,
                                                        cursor);

            return *this;
        }

        TAddModelValue& AddTask(const TObliviousTreeModel& model,
                                const TDataSet& dataSet,
                                TMirrorBuffer<const ui32>&& indices,
                                TMirrorBuffer<float>& cursor) {
            Y_ASSERT(model.GetStructure() == ModelStructure);
            Tasks.push_back({std::move(indices), &cursor, &dataSet});
            const TVector<float>& modelValues = model.GetValues();
            CB_ENSURE(modelValues.size() == 1 << ModelStructure.GetDepth());

            for (ui32 i = 0; i < modelValues.size(); ++i) {
                CpuLeaves.push_back(modelValues[i]);
            }

            return *this;
        }

        void Proceed() {
            TMirrorBuffer<float> leaves = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(CpuLeaves.size()));
            leaves.Write(CpuLeaves);
            ui32 leavesCount = ModelStructure.LeavesCount();

            NCudaLib::GetCudaManager().WaitComplete();

            for (ui32 taskId = 0; taskId < Tasks.size(); ++taskId) {
                auto taskValues = leaves.SliceView(TSlice(taskId * leavesCount, (taskId + 1) * leavesCount));
                Append(taskId, taskValues, Streams[taskId % Streams.size()].GetId());
            }
            NCudaLib::GetCudaManager().WaitComplete();
        }

    private:
        struct TAddModelTask {
            TMirrorBuffer<const ui32> Indices;
            TMirrorBuffer<float>* Cursor;
            const TDataSet* DataSet;
        };

        void Append(ui32 taskId,
                    const TMirrorBuffer<float>& values,
                    ui32 stream) {
            auto& task = Tasks.at(taskId);

            auto& bins = GetBinsForModel(CacheHolder,
                                         FeaturesManager,
                                         *task.DataSet,
                                         ModelStructure);

            AddBinModelValues<NCudaLib::TMirrorMapping>(values,
                                                        bins,
                                                        task.Indices /*read indices*/,
                                                        *task.Cursor,
                                                        stream

            );
        }

    private:
        TVector<TComputationStream> Streams;
        TVector<TAddModelTask> Tasks;
        TScopedCacheHolder& CacheHolder;
        const TBinarizedFeaturesManager& FeaturesManager;
        TObliviousTreeStructure ModelStructure;
        TVector<float> CpuLeaves;
    };
}
