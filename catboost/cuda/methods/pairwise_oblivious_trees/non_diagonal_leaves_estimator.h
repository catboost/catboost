#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h>
#include <catboost/cuda/methods/leaves_estimation/leaves_estimation_config.h>
#include <catboost/cuda/methods/leaves_estimation/non_diagonal_oracle_interface.h>
#include <catboost/cuda/methods/leaves_estimation/non_diagonal_oracle_base.h>
#include <catboost/cuda/methods/leaves_estimation/groupwise_oracle.h>
#include <catboost/cuda/methods/leaves_estimation/pairwise_oracle.h>
#include <catboost/cuda/methods/leaves_estimation/descent_helpers.h>

namespace NCatboostCuda {

    /*
     * Pairwise leaves estimator
     */
    class TPairwiseObliviousTreeLeavesEstimator {
    public:
        TPairwiseObliviousTreeLeavesEstimator(const TLeavesEstimationConfig& leavesEstimationConfig)
        : LeavesEstimationConfig(leavesEstimationConfig) {
        }

        template <class TTarget>
        TPairwiseObliviousTreeLeavesEstimator& AddEstimationTask(const TTarget& target,
                                                                 TStripeBuffer<const float>&& cursor,
                                                                 TObliviousTreeModel* model) {

            TTask task;
            task.Model = model;
            task.Cursor = std::move(cursor);
            task.DataSet = &target.GetDataSet();
            task.DerCalcerFactory = new TNonDiagonalOracleFactory<TTarget>(target);
            Tasks.push_back(std::move(task));
            return *this;
        }

        void Estimate() {
            for (ui32 taskId = 0; taskId < Tasks.size(); ++taskId) {
                Estimate(taskId);
            }
        }

    private:

        struct TTask {
            TObliviousTreeModel* Model = nullptr;
            TStripeBuffer<const float> Cursor;
            const TDocParallelDataSet* DataSet = nullptr;
            THolder<INonDiagonalOracleFactory> DerCalcerFactory;
        };

    private:
        THolder<INonDiagonalOracle> CreateDerCalcer(const TTask& task);

        void Estimate(ui32 taskId);
    private:
        TLeavesEstimationConfig LeavesEstimationConfig;
        TVector<TTask> Tasks;
    };
}
