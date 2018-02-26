#pragma once

#include "helpers.h"
#include "descent_helpers.h"
#include "leaves_estimation_helper.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>
#include <catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h>

namespace NCatboostCuda {
    struct TLeavesEstimationConfig {
        bool UseNewton = true;
        double Lambda = 1.0; //l2 reg
        ui32 Iterations = 10;
        double MinLeafWeight = 1e-20;
        bool IsNormalize = false;
        bool AddRidgeToTargetFunction = false;
        bool MakeZeroAverage = false;

        TLeavesEstimationConfig(bool useNewton,
                                double lambda,
                                ui32 iterations,
                                double minLeafWeight,
                                bool normalize,
                                bool addRidgeToTargetFunction,
                                bool zeroAverage)
            : UseNewton(useNewton)
            , Lambda(lambda)
            , Iterations(iterations)
            , MinLeafWeight(minLeafWeight)
            , IsNormalize(normalize)
            , AddRidgeToTargetFunction(addRidgeToTargetFunction)
            , MakeZeroAverage(zeroAverage)
        {
        }
    };

    /*
     * Oblivious tree batch estimator
     */
    class TObliviousTreeLeavesEstimator {
    public:
        using TDescentPoint = TPointwiseDescentPoint;

    private:
        TVector<TEstimationTaskHelper> TaskHelpers;
        TVector<TSlice> TaskSlices;

        TVector<double> TaskTotalWeights;
        TVector<float> LeafWeights;
        TMirrorBuffer<float> LeafValues;
        TCudaBuffer<float, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost> PartStats;

        const TBinarizedFeaturesManager& FeaturesManager;
        TLeavesEstimationConfig LeavesEstimationConfig;
        TVector<TObliviousTreeModel*> WriteDst;

    private:
        static TDescentPoint Create(ui32 dim) {
            return TPointwiseDescentPoint(dim);
        }

        ui32 GetDim() const {
            CB_ENSURE(TaskHelpers.size());
            return static_cast<ui32>(TaskSlices.back().Right);
        }

        void MoveTo(const TVector<float>& point) {
            auto guard = NCudaLib::GetProfiler().Profile("Move to point");
            CB_ENSURE(LeafValues.GetObjectsSlice().Size() == point.size());
            LeafValues.Write(point);

            const ui32 streamCount = Min<ui32>(TaskHelpers.size(), 8);
            RunInStreams(TaskHelpers.size(), streamCount, [&](ui32 taskId, ui32 streamId) {
                TaskHelpers[taskId].MoveToPoint(LeavesView(LeafValues, taskId), streamId);
            });
        }

        void Regularize(TVector<float>& point) {
            for (ui32 i = 0; i < point.size(); ++i) {
                if (LeafWeights[i] < LeavesEstimationConfig.MinLeafWeight) {
                    point[i] = 0;
                }
            }
        }

        void ComputeValueAndDerivatives(TPointwiseDescentPoint& descentPoint) {
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

            TVector<float> data;
            //TODO(noxoomo): check change to reduceToAll and migrate all gradient descent to device side
            //for 64 leaves cpu side code is fast enough (
            PartStats.CreateReader()
                .SetReadSlice(PartStats.GetMapping().DeviceSlice(0))
                .SetFactorSlice(PartStats.GetMapping().DeviceSlice(0))
                .ReadReduce(data);

            WriteValueAndDerivatives(data, descentPoint);
        }

        TMirrorBuffer<float> LeavesView(TMirrorBuffer<float>& leaves,
                                        ui32 taskId) {
            return leaves.SliceView(TaskSlices[taskId]);
        }

        void NormalizeDerivatives(TVector<float>& derOrDer2) {
            for (ui32 i = 0; i < TaskHelpers.size(); ++i) {
                double taskWeight = TaskTotalWeights[i];
                TSlice taskSlice = TaskSlices[i];
                for (ui32 leaf = taskSlice.Left; leaf < taskSlice.Right; ++leaf) {
                    derOrDer2[leaf] /= taskWeight;
                }
            }
        }

        void WriteValueAndDerivatives(const TVector<float>& data,
                                      TPointwiseDescentPoint& point) {
            CB_ENSURE(TaskSlices.size());

            const bool normalize = LeavesEstimationConfig.IsNormalize;

            const ui32 taskCount = TaskHelpers.size();
            point.Value = 0;
            for (ui32 i = 0; i < taskCount; ++i) {
                point.Value += normalize ? data[i] / TaskTotalWeights[i] : data[i];
            }

            const ui32 totalLeavesCount = TaskSlices.back().Right;
            //            const ui32 leavesCount = Structure.LeavesCount();
            point.Gradient.resize(TaskSlices.back().Right);
            point.Hessian.resize(TaskSlices.back().Right);

            Copy(data.begin() + taskCount,
                 data.begin() + taskCount + totalLeavesCount,
                 point.Gradient.begin());

            if (LeavesEstimationConfig.UseNewton) {
                Copy(data.begin() + taskCount + totalLeavesCount,
                     data.begin() + taskCount + 2 * totalLeavesCount,
                     point.Hessian.begin());
            } else {
                Copy(LeafWeights.begin(), LeafWeights.end(), point.Hessian.begin());
            }

            if (normalize) {
                NormalizeDerivatives(point.Gradient);
                NormalizeDerivatives(point.Hessian);
            }

            AddRidgeRegularization(LeavesEstimationConfig.Lambda, point);
        }

        void WriteWeights(TVector<float>& dst) {
            dst.resize(LeafWeights.size());
            Copy(LeafWeights.begin(), LeafWeights.end(), dst.begin());
        }

        void CreatePartStats() {
            const ui32 taskCount = TaskHelpers.size();
            const ui32 totalLeavesCount = TaskSlices.back().Right;
            ui32 sumCount = LeavesEstimationConfig.UseNewton ? 2 : 1;
            auto mapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(totalLeavesCount * sumCount + taskCount);
            PartStats.Reset(mapping);
        }

        void ComputePartWeights() {
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

        template <class TOracle,
                  class TBacktrackingStepEstimator>
        friend class TNewtonLikeWalker;

        inline TEstimationTaskHelper& NextTask(TObliviousTreeModel& model) {
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

    public:
        TObliviousTreeLeavesEstimator(const TBinarizedFeaturesManager& featuresManager,
                                      const TLeavesEstimationConfig& config)
            : FeaturesManager(featuresManager)
            , LeavesEstimationConfig(config)
        {
        }

        template <class TTarget>
        TObliviousTreeLeavesEstimator& AddEstimationTask(TScopedCacheHolder& scopedCache,
                                                         TTarget&& target,
                                                         TMirrorBuffer<const float>&& current,
                                                         TObliviousTreeModel* dst) {
            const ui32 binCount = static_cast<ui32>(1 << dst->GetStructure().GetDepth());

            const auto& docBins = GetBinsForModel(scopedCache,
                                                  FeaturesManager,
                                                  target.GetDataSet(),
                                                  dst->GetStructure());
            TEstimationTaskHelper& task = NextTask(*dst);

            auto strippedTarget = MakeStripeTargetFunc(target);
            task.Bins = strippedTarget.template CreateGpuBuffer<ui32>();
            Gather(task.Bins, docBins, strippedTarget.GetTarget().GetIndices());

            task.Baseline = TStripeBuffer<float>::CopyMapping(task.Bins);
            task.Cursor = TStripeBuffer<float>::CopyMapping(task.Bins);

            auto indices = strippedTarget.template CreateGpuBuffer<ui32>();
            MakeSequence(indices);
            ReorderBins(task.Bins, indices, 0, dst->GetStructure().GetDepth());

            Gather(task.Baseline,
                   NCudaLib::StripeView(current, indices.GetMapping()),
                   indices);

            auto offsetsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(binCount + 1);
            task.Offsets = TCudaBuffer<ui32, NCudaLib::TStripeMapping>::Create(offsetsMapping);
            UpdatePartitionOffsets(task.Bins, task.Offsets);

            task.DerCalcer = CreatePermutationDerCalcer(std::move(strippedTarget), std::move(indices));

            return *this;
        }

        template <class TTarget>
        TObliviousTreeLeavesEstimator& AddEstimationTask(TTarget&& target,
                                                         TStripeBuffer<const float>&& current,
                                                         TObliviousTreeModel* dst) {
            const ui32 binCount = static_cast<ui32>(1 << dst->GetStructure().GetDepth());

            TEstimationTaskHelper& task = NextTask(*dst);

            task.Bins = target.template CreateGpuBuffer<ui32>();
            {
                auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Compute bins doc-parallel");
                ComputeBinsForModel(dst->GetStructure(),
                                    target.GetDataSet(),
                                    &task.Bins);
            }

            task.Baseline = target.template CreateGpuBuffer<float>();
            task.Cursor = target.template CreateGpuBuffer<float>();
            auto indices = target.template CreateGpuBuffer<ui32>();
            auto offsetsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(binCount + 1);
            task.Offsets = TCudaBuffer<ui32, NCudaLib::TStripeMapping>::Create(offsetsMapping);

            MakeSequence(indices);
            ReorderBins(task.Bins, indices, 0, dst->GetStructure().GetDepth());
            Gather(task.Baseline, current, indices);

            UpdatePartitionOffsets(task.Bins,
                                   task.Offsets);

            task.DerCalcer = CreatePermutationDerCalcer(std::move(target),
                                                        std::move(indices));

            return *this;
        }

        void Estimate() {
            CreatePartStats();
            ComputePartWeights();

            const ui32 totalLeavesCount = TaskSlices.back().Right;
            LeafValues = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(totalLeavesCount));
            FillBuffer(LeafValues, 0.0f);

            TNewtonLikeWalker<TObliviousTreeLeavesEstimator, TSimpleStepEstimator> newtonLikeWalker(*this,
                                                                                                    LeavesEstimationConfig.Iterations);

            TVector<float> point;
            point.resize(totalLeavesCount);
            point = newtonLikeWalker.Estimate(point);

            for (ui32 taskId = 0; taskId < TaskHelpers.size(); ++taskId) {
                float* values = ~point + TaskSlices[taskId].Left;
                float* weights = ~LeafWeights + TaskSlices[taskId].Left;

                TObliviousTreeModel& dst = *WriteDst[taskId];
                const auto taskLeavesCount = TaskSlices[taskId].Size();
                TVector<float> leaves(taskLeavesCount);

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
                }
                dst.UpdateLeaves(std::move(leaves));
            }
        }
    };
}
