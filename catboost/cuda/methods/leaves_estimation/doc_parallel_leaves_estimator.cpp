#include "doc_parallel_leaves_estimator.h"

void NCatboostCuda::TDocParallelLeavesEstimator::Estimate(ui32 taskId, NPar::TLocalExecutor* localExecutor) {
    auto& task = Tasks.at(taskId);
    auto derCalcer = CreateDerCalcer(task);

    TNewtonLikeWalker newtonLikeWalker(*derCalcer,
                                       LeavesEstimationConfig.Iterations,
                                       LeavesEstimationConfig.BacktrackingType);

    TVector<float> point;
    TVector<double> weights;

    point.resize(task.Model->BinCount() * task.Model->OutputDim());
    point = newtonLikeWalker.Estimate(point, localExecutor);
    derCalcer->WriteWeights(&weights);
    Y_VERIFY(task.Model->BinCount() == weights.size());

    if (LeavesEstimationConfig.MakeZeroAverage) {
        double sum = 0;
        double weight = 0;
        for (size_t i = 0; i < point.size(); ++i) {
            sum += point[i];
            weight += 1;
        }
        const double bias = weight > 0 ? -sum / weight : 0;

        for (size_t i = 0; i < point.size(); ++i) {
            point[i] += bias;
        }
    }

    task.Model->UpdateLeaves(std::move(point));
    task.Model->UpdateWeights(std::move(weights));
}

THolder<NCatboostCuda::ILeavesEstimationOracle> NCatboostCuda::TDocParallelLeavesEstimator::CreateDerCalcer(const NCatboostCuda::TDocParallelLeavesEstimator::TTask& task) {
    const ui32 binCount = static_cast<ui32>(task.Model->BinCount());
    auto bins = TStripeBuffer<ui32>::CopyMapping(task.Cursor);
    {
        auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Compute bins doc-parallel");
        task.Model->ComputeBins(*task.DataSet,
                                &bins);
    }

    return task.DerCalcerFactory->Create(LeavesEstimationConfig,
                                         task.Cursor.ConstCopyView(),
                                         std::move(bins),
                                         binCount);
}
