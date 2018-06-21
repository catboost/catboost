#include "oblivious_tree_leaves_estimator.h"

namespace NCatboostCuda {
    void TObliviousTreeLeavesEstimator::WriteValueAndFirstDerivatives(double* value,
                                                                      TVector<float>* gradient) {
        auto& data = GetCurrentPointInfo();
        CB_ENSURE(TaskSlices.size());
        CB_ENSURE(value);
        CB_ENSURE(gradient);

        const bool normalize = LeavesEstimationConfig.IsNormalize;

        const ui32 taskCount = TaskHelpers.size();
        (*value) = 0;
        for (ui32 i = 0; i < taskCount; ++i) {
            (*value) += normalize ? data[i] / TaskTotalWeights[i] : data[i];
        }

        const ui32 totalLeavesCount = TaskSlices.back().Right;
        gradient->clear();
        gradient->resize(totalLeavesCount);

        Copy(data.begin() + taskCount,
             data.begin() + taskCount + totalLeavesCount,
             gradient->begin());

        if (normalize) {
            NormalizeDerivatives(*gradient);
        }

        const double lambda = LeavesEstimationConfig.Lambda;

        if (LeavesEstimationConfig.AddRidgeToTargetFunction) {
            double hingeLoss = 0;
            {
                for (const auto& val : CurrentPoint) {
                    hingeLoss += val * val;
                }
                hingeLoss *= lambda / 2;
            }
            (*value) -= hingeLoss;
            for (size_t i = 0; i < gradient->size(); ++i) {
                (*gradient)[i] -= lambda * CurrentPoint[i];
            }
        }
    }

    void NCatboostCuda::TObliviousTreeLeavesEstimator::ComputePartWeights() {
        const ui32 totalLeavesCount = TaskSlices.back().Right;
        const ui32 taskCount = TaskHelpers.size();

        CB_ENSURE(PartStats.GetMapping().DeviceSlice(0).Size() >= totalLeavesCount);

        RunInStreams(taskCount, Min<ui32>(taskCount, 8), [&](ui32 taskId, ui32 streamId) {
            TEstimationTaskHelper& taskHelper = TaskHelpers[taskId];
            auto weightBuffer = NCudaLib::ParallelStripeView(PartStats,
                                                             TaskSlices[taskId]);

            taskHelper.ProjectWeights(weightBuffer, streamId);
        });

        auto weightsBufferSlice = NCudaLib::ParallelStripeView(PartStats,
                                                               TSlice(0, totalLeavesCount));
        TSlice reduceSlice = weightsBufferSlice.GetMapping().DeviceSlice(0);

        weightsBufferSlice
            .CreateReader()
            .SetReadSlice(reduceSlice)
            .SetFactorSlice(reduceSlice)
            .ReadReduce(LeafWeights);

        TaskTotalWeights.resize(taskCount);

        for (ui32 i = 0; i < TaskHelpers.size(); ++i) {
            auto slice = TaskSlices[i];
            for (ui32 leaf = slice.Left; leaf < slice.Right; ++leaf) {
                TaskTotalWeights[i] += LeafWeights[leaf];
            }
        }
    }

    const TVector<float>& NCatboostCuda::TObliviousTreeLeavesEstimator::GetCurrentPointInfo() {
        if (CurrentPointInfo == nullptr) {
            CB_ENSURE(CurrentPoint.size(), "Error: set point first");
            CurrentPointInfo = new TVector<float>;
            auto& profiler = NCudaLib::GetProfiler();
            auto projectDerGuard = profiler.Profile("Compute values and derivatives");

            const ui32 taskCount = static_cast<const ui32>(TaskHelpers.size());
            //            const ui32 leavesCount = Structure.LeavesCount();
            const ui32 streamCount = Min<ui32>(taskCount, 8);
            RunInStreams(taskCount, streamCount, [&](ui32 taskId, ui32 streamId) {
                TEstimationTaskHelper& taskHelper = TaskHelpers[taskId];
                auto scoreView = NCudaLib::ParallelStripeView(PartStats, TSlice(taskId, taskId + 1));
                TSlice taskSlice = TaskSlices[taskId];
                const ui32 derOffset = taskCount + taskSlice.Left;

                auto derView = NCudaLib::ParallelStripeView(PartStats,
                                                            TSlice(derOffset, derOffset + taskSlice.Size()));

                TCudaBuffer<float, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost> der2View;
                if (LeavesEstimationConfig.UseNewton) {
                    const ui32 der2Offset = taskCount + TaskSlices.back().Right + taskSlice.Left;
                    der2View = NCudaLib::ParallelStripeView(PartStats,
                                                            TSlice(der2Offset, der2Offset + taskSlice.Size()));
                }

                taskHelper.Project(&scoreView,
                                   &derView,
                                   LeavesEstimationConfig.UseNewton ? &der2View : nullptr,
                                   streamId);
            });

            //TODO(noxoomo): check change to reduceToAll and migrate all gradient descent to device side
            //for 64 leaves cpu side code is fast enough (
            PartStats.CreateReader()
                .SetReadSlice(PartStats.GetMapping().DeviceSlice(0))
                .SetFactorSlice(PartStats.GetMapping().DeviceSlice(0))
                .ReadReduce(*CurrentPointInfo);
        }
        return *CurrentPointInfo;
    }

    void NCatboostCuda::TObliviousTreeLeavesEstimator::WriteSecondDerivatives(TVector<float>* secondDer) {
        auto& data = GetCurrentPointInfo();
        CB_ENSURE(TaskSlices.size());

        const bool normalize = LeavesEstimationConfig.IsNormalize;

        const ui32 taskCount = TaskHelpers.size();
        const ui32 totalLeavesCount = TaskSlices.back().Right;

        secondDer->clear();
        secondDer->resize(totalLeavesCount);

        if (LeavesEstimationConfig.UseNewton) {
            Copy(data.begin() + taskCount + totalLeavesCount,
                 data.begin() + taskCount + 2 * totalLeavesCount,
                 secondDer->begin());
        } else {
            Copy(LeafWeights.begin(), LeafWeights.end(), secondDer->begin());
        }
        const double lambda = LeavesEstimationConfig.Lambda;
        if (normalize) {
            NormalizeDerivatives(*secondDer);
        }
        for (ui32 i = 0; i < secondDer->size(); ++i) {
            (*secondDer)[i] += lambda;
        }
    }

    void TObliviousTreeLeavesEstimator::Estimate() {
        CreatePartStats();
        ComputePartWeights();

        const ui32 totalLeavesCount = TaskSlices.back().Right;
        LeafValues = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(totalLeavesCount));
        FillBuffer(LeafValues, 0.0f);

        TNewtonLikeWalker<TObliviousTreeLeavesEstimator> newtonLikeWalker(*this,
                                                                          LeavesEstimationConfig.Iterations,
                                                                          LeavesEstimationConfig.BacktrackingType);

        TVector<float> point;
        point.resize(totalLeavesCount);
        point = newtonLikeWalker.Estimate(point);

        for (ui32 taskId = 0; taskId < TaskHelpers.size(); ++taskId) {
            float* values = ~point + TaskSlices[taskId].Left;
            float* weights = ~LeafWeights + TaskSlices[taskId].Left;

            TObliviousTreeModel& dst = *WriteDst[taskId];
            const auto taskLeavesCount = TaskSlices[taskId].Size();
            TVector<float> leaves(taskLeavesCount);
            TVector<float> leavesWeights(taskLeavesCount);

            double bias = 0;
            if (LeavesEstimationConfig.MakeZeroAverage) {
                double sum = 0;
                double weight = 0;
                for (ui32 i = 0; i < taskLeavesCount; ++i) {
                    sum += weights[i] * leaves[i];
                    weight += weights[i];
                }
                bias = weight > 0 ? -sum / weight : 0;
            }

            for (ui32 i = 0; i < taskLeavesCount; ++i) {
                leaves[i] = values[i] + bias;
                leavesWeights[i] = weights[i];
            }
            dst.UpdateLeaves(std::move(leaves));
            dst.UpdateLeavesWeights(std::move(leavesWeights));
        }
    }

    TEstimationTaskHelper& TObliviousTreeLeavesEstimator::NextTask(TObliviousTreeModel& model) {
        CB_ENSURE(TaskHelpers.size() == WriteDst.size());
        CB_ENSURE(TaskHelpers.size() == TaskSlices.size());

        TSlice taskSlice;
        if (TaskHelpers.size()) {
            taskSlice.Left = TaskSlices.back().Right;
        }
        taskSlice.Right = taskSlice.Left + model.GetValues().size();

        TaskHelpers.push_back(TEstimationTaskHelper());
        WriteDst.push_back(&model);
        TaskSlices.push_back(taskSlice);
        return TaskHelpers.back();
    }

    void TObliviousTreeLeavesEstimator::MoveTo(const TVector<float>& point) {
        auto guard = NCudaLib::GetProfiler().Profile("Move to point");
        CB_ENSURE(LeafValues.GetObjectsSlice().Size() == point.size());
        LeafValues.Write(point);

        const ui32 streamCount = Min<ui32>(TaskHelpers.size(), 8);
        RunInStreams(TaskHelpers.size(), streamCount, [&](ui32 taskId, ui32 streamId) {
            TaskHelpers[taskId].MoveToPoint(LeavesView(LeafValues, taskId), streamId);
        });

        CurrentPoint = point;
        CurrentPointInfo.Reset();
    }

    void TObliviousTreeLeavesEstimator::Regularize(TVector<float>* point) {
        for (ui32 i = 0; i < point->size(); ++i) {
            if (LeafWeights[i] < LeavesEstimationConfig.MinLeafWeight) {
                (*point)[i] = 0;
            }
        }
    }

    void TObliviousTreeLeavesEstimator::NormalizeDerivatives(TVector<float>& derOrDer2) {
        for (ui32 i = 0; i < TaskHelpers.size(); ++i) {
            double taskWeight = TaskTotalWeights[i];
            TSlice taskSlice = TaskSlices[i];
            for (ui32 leaf = taskSlice.Left; leaf < taskSlice.Right; ++leaf) {
                derOrDer2[leaf] /= taskWeight;
            }
        }
    }

    void TObliviousTreeLeavesEstimator::CreatePartStats() {
        const ui32 taskCount = TaskHelpers.size();
        const ui32 totalLeavesCount = TaskSlices.back().Right;
        ui32 sumCount = LeavesEstimationConfig.UseNewton ? 2 : 1;
        auto mapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(totalLeavesCount * sumCount + taskCount);
        PartStats.Reset(mapping);
    }
}
