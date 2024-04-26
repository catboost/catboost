#include "oblivious_tree_leaves_estimator.h"
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>

namespace NCatboostCuda {
    void TObliviousTreeLeavesEstimator::WriteValueAndFirstDerivatives(double* value,
                                                                      TVector<double>* gradient) {
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

        AddRigdeRegulaizationIfNecessary(LeavesEstimationConfig, CurrentPoint, value, gradient);
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

    TVector<float> NCatboostCuda::TObliviousTreeLeavesEstimator::EstimateExact() {
        const ui32 taskCount = static_cast<const ui32>(TaskHelpers.size());
        const ui32 streamCount = Min<ui32>(taskCount, 8);

        TVector<float> result(TaskSlices.back().Right);
        RunInStreams(taskCount, streamCount, [&](ui32 taskId, ui32 streamId) {
            TEstimationTaskHelper& taskHelper = TaskHelpers[taskId];
            TSlice taskSlice = TaskSlices[taskId];
            ui32 taskSliceSize = taskSlice.Size();

            TVector<float> point(taskSliceSize);
            taskHelper.ComputeExact(point, this->LeavesEstimationConfig.LossDescription, streamId);
            CB_ENSURE(point.size() == taskSliceSize);

            for (ui32 index = taskSlice.Left; index < taskSlice.Right; ++index) {
                result[index] = point[index - taskSlice.Left];
            }
        });

        MoveTo(result);
        return MakeEstimationResult(result);
    }

    const TVector<double>& NCatboostCuda::TObliviousTreeLeavesEstimator::GetCurrentPointInfo() {
        if (CurrentPointInfo == nullptr) {
            CB_ENSURE(CurrentPoint.size(), "Error: set point first");
            CurrentPointInfo = MakeHolder<TVector<double>>();
            auto& profiler = NCudaLib::GetProfiler();
            auto projectDerGuard = profiler.Profile("Compute values and derivatives");

            const ui32 taskCount = static_cast<const ui32>(TaskHelpers.size());
            //            const ui32 leavesCount = Structure.LeavesCount();
            const ui32 streamCount = Min<ui32>(taskCount, 8);
            FillBuffer(PartStats, 0.0);
            RunInStreams(taskCount, streamCount, [&](ui32 taskId, ui32 streamId) {
                TEstimationTaskHelper& taskHelper = TaskHelpers[taskId];
                auto scoreView = NCudaLib::ParallelStripeView(PartStats, TSlice(taskId, taskId + 1));
                TSlice taskSlice = TaskSlices[taskId];
                const ui32 derOffset = taskCount + taskSlice.Left;

                auto derView = NCudaLib::ParallelStripeView(PartStats,
                                                            TSlice(derOffset, derOffset + taskSlice.Size()));

                TCudaBuffer<double, NCudaLib::TStripeMapping> der2View;
                if (LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton) {
                    const ui32 der2Offset = taskCount + TaskSlices.back().Right + taskSlice.Left;
                    der2View = NCudaLib::ParallelStripeView(PartStats,
                                                            TSlice(der2Offset, der2Offset + taskSlice.Size()));
                }

                taskHelper.Project(&scoreView,
                                   &derView,
                                   LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton ? &der2View : nullptr,
                                   streamId);
            });

            //TODO(noxoomo): check change to reduceToAll and migrate all gradient descent to device side
            //for 64 leaves CPU side code is fast enough (
            PartStats
                .CreateReader()
                .SetReadSlice(PartStats.GetMapping().DeviceSlice(0))
                .SetFactorSlice(PartStats.GetMapping().DeviceSlice(0))
                .ReadReduce(*CurrentPointInfo);
        }
        return *CurrentPointInfo;
    }

    void NCatboostCuda::TObliviousTreeLeavesEstimator::WriteSecondDerivatives(TVector<double>* secondDer) {
        auto& data = GetCurrentPointInfo();
        CB_ENSURE(TaskSlices.size());

        const bool normalize = LeavesEstimationConfig.IsNormalize;

        const ui32 taskCount = TaskHelpers.size();
        const ui32 totalLeavesCount = TaskSlices.back().Right;

        secondDer->clear();
        secondDer->resize(totalLeavesCount);

        if (LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton) {
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

    void TObliviousTreeLeavesEstimator::AddLangevinNoiseToDerivatives(TVector<double>* derivatives,
                                                                      NPar::ILocalExecutor* localExecutor) {
        if (LeavesEstimationConfig.Langevin) {
            AddLangevinNoise(LeavesEstimationConfig, derivatives, localExecutor, Random.NextUniformL());
        }
    }

    void TObliviousTreeLeavesEstimator::Estimate(NPar::ILocalExecutor* localExecutor) {
        CreatePartStats();
        ComputePartWeights();

        const ui32 totalLeavesCount = TaskSlices.back().Right;
        LeafValues = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(totalLeavesCount));
        FillBuffer(LeafValues, 0.0f);

        TVector<float> point;
        if (this->LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Exact) {
            point = EstimateExact();
        } else {
            TNewtonLikeWalker newtonLikeWalker(*this,
                                               LeavesEstimationConfig.Iterations,
                                               LeavesEstimationConfig.BacktrackingType);
            point = newtonLikeWalker.Estimate(point, localExecutor);
        }

        for (ui32 taskId = 0; taskId < TaskHelpers.size(); ++taskId) {
            float* values = point.data() + TaskSlices[taskId].Left;
            double* weights = LeafWeights.data() + TaskSlices[taskId].Left;

            TObliviousTreeModel& dst = *WriteDst[taskId];
            const auto taskLeavesCount = TaskSlices[taskId].Size();
            TVector<float> leaves(taskLeavesCount);
            TVector<double> leavesWeights(taskLeavesCount);

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
            dst.UpdateLeaves(leaves);
            dst.UpdateWeights(leavesWeights);
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
        RegularizeImpl(LeavesEstimationConfig, TConstArrayRef<double>(LeafWeights.begin(), LeafWeights.begin() + PointDim()), point);
    }

    void TObliviousTreeLeavesEstimator::NormalizeDerivatives(TVector<double>& derOrDer2) {
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
        ui32 sumCount = LeavesEstimationConfig.LeavesEstimationMethod == ELeavesEstimation::Newton ? 2 : 1;
        auto mapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(totalLeavesCount * sumCount + taskCount);
        PartStats.Reset(mapping);
    }

    void TEstimationTaskHelper::MoveToPoint(const TMirrorBuffer<float>& point, ui32 stream) {
        Cursor.Copy(Baseline, stream);

        AddBinModelValues(point,
                          Bins,
                          Cursor,
                          stream);
    }

    void
    TEstimationTaskHelper::ProjectWeights(TCudaBuffer<double, NCudaLib::TStripeMapping>& weightsDst, ui32 streamId) {
        ComputePartitionStats(DerCalcer->GetWeights(streamId),
                              Offsets,
                              &weightsDst,
                              streamId);
    }

    void TEstimationTaskHelper::Project(TCudaBuffer<double, NCudaLib::TStripeMapping>* value,
                                        TCudaBuffer<double, NCudaLib::TStripeMapping>* der,
                                        TCudaBuffer<double, NCudaLib::TStripeMapping>* der2, ui32 stream) {
        if (value) {
            TmpValue.Reset(Cursor.GetMapping().Transform([&](const TSlice&) -> ui64 {
                return 1;
            }));
        }
        if (der) {
            TmpDer.Reset(Cursor.GetMapping());
        }
        if (der2) {
            TmpDer2.Reset(Cursor.GetMapping());
        }

        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
        DerCalcer->ApproximateAt(Cursor,
                                 value ? &TmpValue : nullptr,
                                 der ? &TmpDer : nullptr,
                                 der2 ? &TmpDer2 : nullptr,
                                 stream);
        if (value) {
            CastCopy(TmpValue, value, stream);
            //                value->Copy(TmpValue, stream);
        }
        {
            auto guard = profiler.Profile("Segmented reduce derivatives");
            if (der) {
                ComputePartitionStats(TmpDer, Offsets, der, stream);
            }
            if (der2) {
                ComputePartitionStats(TmpDer2, Offsets, der2, stream);
            }
        }
    }
}
