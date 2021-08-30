#pragma once

#include "matrix_per_tree_oracle_base.h"
#include "leaves_estimation_factory.h"
#include "pairwise_oracle.h"
#include "groupwise_oracle.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/methods/leaves_estimation/leaves_estimation_config.h>
#include <catboost/cuda/methods/leaves_estimation/oracle_interface.h>
#include <catboost/cuda/methods/leaves_estimation/descent_helpers.h>
#include <catboost/cuda/models/bin_optimized_model.h>

namespace NCatboostCuda {
    /*
     * Pairwise leaves estimator
     */
    class TDocParallelLeavesEstimator {
    public:
        TDocParallelLeavesEstimator(const TLeavesEstimationConfig& leavesEstimationConfig, TGpuAwareRandom& random)
            : LeavesEstimationConfig(leavesEstimationConfig), Random(random)
        {
        }

        template <class TTarget>
        TDocParallelLeavesEstimator& AddEstimationTask(const TTarget& target,
                                                       const TDocParallelDataSet& dataSet,
                                                       TStripeBuffer<const float>&& cursor,
                                                       IBinOptimizedModel* model) {
            TTask task;
            task.Model = model;
            task.Cursor = std::move(cursor);
            task.DataSet = &dataSet;
            task.DerCalcerFactory = MakeHolder<TOracleFactory<TTarget>>(target);
            Tasks.push_back(std::move(task));
            return *this;
        }

        void Estimate(NPar::ILocalExecutor* localExecutor) {
            for (ui32 taskId = 0; taskId < Tasks.size(); ++taskId) {
                Estimate(taskId, localExecutor);
            }
        }

    private:
        struct TTask {
            IBinOptimizedModel* Model = nullptr;
            TStripeBuffer<const float> Cursor;
            const TDocParallelDataSet* DataSet = nullptr;
            THolder<ILeavesEstimationOracleFactory> DerCalcerFactory;
        };

    private:
        THolder<ILeavesEstimationOracle> CreateDerCalcer(const TTask& task);

        void Estimate(ui32 taskId, NPar::ILocalExecutor* localExecutor);

    private:
        TLeavesEstimationConfig LeavesEstimationConfig;
        TVector<TTask> Tasks;
        TGpuAwareRandom& Random;
    };
}
