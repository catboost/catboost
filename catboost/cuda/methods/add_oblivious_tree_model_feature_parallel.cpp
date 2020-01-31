#include "add_oblivious_tree_model_feature_parallel.h"
#include "helpers.h"
#include <catboost/cuda/models/add_bin_values.h>

namespace NCatboostCuda {
    TAddObliviousTreeFeatureParallel& TAddObliviousTreeFeatureParallel::Append(const TObliviousTreeModel& model,
                                                                               const TAddObliviousTreeFeatureParallel::TDataSet& dataSet,
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

    void TAddObliviousTreeFeatureParallel::Append(ui32 taskId, const TMirrorBuffer<float>& values, ui32 stream) {
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

    TAddObliviousTreeFeatureParallel& TAddObliviousTreeFeatureParallel::AddTask(const TObliviousTreeModel& model,
                                                                                const TAddObliviousTreeFeatureParallel::TDataSet& dataSet,
                                                                                TMirrorBuffer<const ui32>&& indices,
                                                                                TMirrorBuffer<float>& cursor) {
        Y_ASSERT(model.GetStructure() == ModelStructure);
        Tasks.push_back({std::move(indices), &cursor, &dataSet});
        const TVector<float>& modelValues = model.GetValues();
        CB_ENSURE(modelValues.size() == 1ull << ModelStructure.GetDepth());

        for (ui32 i = 0; i < modelValues.size(); ++i) {
            CpuLeaves.push_back(modelValues[i]);
        }

        return *this;
    }

    void TAddObliviousTreeFeatureParallel::Proceed() {
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

}
