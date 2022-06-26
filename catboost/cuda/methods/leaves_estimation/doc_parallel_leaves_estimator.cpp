#include "doc_parallel_leaves_estimator.h"

void NCatboostCuda::TDocParallelLeavesEstimator::Estimate(ui32 taskId, NPar::ILocalExecutor* localExecutor) {
    auto& task = Tasks.at(taskId);
    auto derCalcer = CreateDerCalcer(task);

    TVector<float> point;
    TVector<double> weights;

    point.resize(task.Model->BinCount() * task.Model->OutputDim());
    if (this->LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Exact) {
        point = derCalcer->EstimateExact();
    } else {
        TNewtonLikeWalker newtonLikeWalker(*derCalcer,
                                           LeavesEstimationConfig.Iterations,
                                           LeavesEstimationConfig.BacktrackingType);
        point = newtonLikeWalker.Estimate(point, localExecutor);
    }

    derCalcer->WriteWeights(&weights);
    CB_ENSURE(
        task.Model->BinCount() == weights.size(),
        "Unexpected number of weights " << weights.size() << ", should be " << task.Model->BinCount());

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

    auto calcer = task.DerCalcerFactory->Create(LeavesEstimationConfig,
                                         task.Cursor.ConstCopyView(),
                                         std::move(bins),
                                         binCount,
                                         Random);
    return std::move(calcer);
}
