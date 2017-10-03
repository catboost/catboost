#pragma once

#include "helpers.h"
#include "descent_helpers.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_base.h>
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>

/*
 * Oblivious tree batch estimator
 */
template <template <class TMapping, class> class TTarget, class TDataSet>
class TObliviousTreeLeavesEstimator {
public:
    using TMirrorTarget = TTarget<NCudaLib::TMirrorMapping, TDataSet>;
    using TStripeTarget = TTarget<NCudaLib::TStripeMapping, TDataSet>;
    using TVec = typename TMirrorTarget::TVec;
    using TConstVec = typename TMirrorTarget::TConstVec;
    using TDescentPoint = TPointwiseDescentPoint;

private:
    struct TEstimationTaskHelper {
        TPermutationDerCalcer<TStripeTarget> DerCalcer;

        TStripeBuffer<ui32> Bins;
        TStripeBuffer<ui32> Offsets;
        TStripeBuffer<float> Baseline;
        TStripeBuffer<float> Cursor;

        TStripeBuffer<float> TmpDer;
        TStripeBuffer<float> TmpValue;
        TStripeBuffer<float> TmpDer2;

        TEstimationTaskHelper() = default;

        void MoveToPoint(const TMirrorBuffer<float>& point, ui32 stream = 0) {
            Cursor.Copy(Baseline, stream);

            AddBinModelValues(Cursor,
                              point,
                              Bins,
                              stream
            );
        }

        template <NCudaLib::EPtrType Type>
        void ProjectWeights(TCudaBuffer<float, NCudaLib::TStripeMapping, Type>& weightsDst, ui32 streamId = 0) {
            SegmentedReduceVector(DerCalcer.GetWeights(), Offsets, weightsDst, EOperatorType::Sum, streamId);
        }

        template <NCudaLib::EPtrType PtrType>
        void Project(TCudaBuffer<float, NCudaLib::TStripeMapping, PtrType>* value,
                     TCudaBuffer<float, NCudaLib::TStripeMapping, PtrType>* der,
                     TCudaBuffer<float, NCudaLib::TStripeMapping, PtrType>* der2,
                     ui32 stream = 0) {
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
            DerCalcer.ApproximateAt(Cursor,
                                    value ? &TmpValue : nullptr,
                                    der ? &TmpDer : nullptr,
                                    der2 ? &TmpDer2 : nullptr,
                                    stream);

            value->Copy(TmpValue, stream);
            {
                auto guard = profiler.Profile("Segmented reduce derivatives");
                if (der) {
                    SegmentedReduceVector(TmpDer, Offsets, *der, EOperatorType::Sum, stream);
                }
                if (der2) {
                    SegmentedReduceVector(TmpDer2, Offsets, *der2, EOperatorType::Sum, stream);
                }
            }
        }
    };

    yvector<TEstimationTaskHelper> TaskHelpers;

    yvector<float> LeafWeights;
    yvector<double> TaskTotalWeights;
    TMirrorBuffer<float> LeafValues;
    TCudaBuffer<float, NCudaLib::TStripeMapping, NCudaLib::CudaHost> PartStats;

    TObliviousTreeStructure Structure;
    const TBinarizedFeaturesManager& FeaturesManager;
    TScopedCacheHolder& ScopedCache;
    yvector<TObliviousTreeModel*> WriteDst;

    bool UseNewton = true;
    double Lambda = 1.0; //l2 reg
    ui32 Iterations = 10;
    ui32 MinLeafWeight = 1;
    bool IsNormalize;
    bool AddRidgeToTargetFunction = false;

private:
    static TDescentPoint Create(ui32 dim) {
        return TPointwiseDescentPoint(dim);
    }

    ui32 GetDim() const {
        return static_cast<ui32>(TaskHelpers.size() * (1 << Structure.GetDepth()));
    }

    void MoveTo(const yvector<float>& point) {
        auto guard = NCudaLib::GetProfiler().Profile("Move to point");
        CB_ENSURE(LeafValues.GetObjectsSlice().Size() == point.size());

        LeafValues.Write(point);
        const ui32 streamCount = Min<ui32>(TaskHelpers.size(), 8);
        RunInStreams(TaskHelpers.size(), streamCount, [&](ui32 taskId, ui32 streamId) {
            TaskHelpers[taskId].MoveToPoint(LeavesView(LeafValues, taskId), streamId);
        });
    }

    void Regularize(yvector<float>& point) {
        for (ui32 i = 0; i < point.size(); ++i) {
            if (LeafWeights[i] < MinLeafWeight) {
                point[i] = 0;
            }
        }
    }

    void ComputeValueAndDerivatives(TPointwiseDescentPoint& descentPoint) {
        auto& profiler = NCudaLib::GetProfiler();
        auto projectDerGuard = profiler.Profile("Compute values and derivatives");

        const ui32 taskCount = static_cast<const ui32>(TaskHelpers.size());
        const ui32 leavesCount = Structure.LeavesCount();
        const ui32 streamCount = Min<ui32>(taskCount, 8);
        RunInStreams(taskCount, streamCount, [&](ui32 taskId, ui32 streamId) {
            TEstimationTaskHelper& taskHelper = TaskHelpers[taskId];
            auto scoreView = NCudaLib::ParallelStripeView(PartStats, TSlice(taskId, taskId + 1));
            const ui32 derOffset = taskCount + taskId * leavesCount;
            auto derView = NCudaLib::ParallelStripeView(PartStats, TSlice(derOffset, derOffset + leavesCount));

            TCudaBuffer<float, NCudaLib::TStripeMapping, NCudaLib::CudaHost> der2View;
            if (UseNewton) {
                const ui32 der2Offset = taskCount * (leavesCount + 1) + taskId * leavesCount;
                der2View = NCudaLib::ParallelStripeView(PartStats, TSlice(der2Offset, der2Offset + leavesCount));
            }

            taskHelper.Project(&scoreView,
                               &derView,
                               UseNewton ? &der2View : nullptr,
                               streamId
            );
        });

        yvector<float> data;
        //TODO(noxoomo): change to reduceToAll and migrate all gradient descent to device side

        PartStats.CreateReader()
            .SetReadSlice(PartStats.GetMapping().DeviceSlice(0))
            .SetFactorSlice(PartStats.GetMapping().DeviceSlice(0))
            .ReadReduce(data);
        WriteValueAndDerivatives(data, descentPoint);
    }

    TMirrorBuffer<float> LeavesView(TMirrorBuffer<float>& leaves,
                                    ui32 taskId) {
        return leaves.SliceView(TSlice(taskId * Structure.LeavesCount(), (taskId + 1) * Structure.LeavesCount()));
    }

    void NormalizeDerivatives(yvector<float>& derOrDer2) {
        ui32 cursor = 0;

        for (ui32 i = 0; i < TaskHelpers.size(); ++i) {
            double taskWeight = TaskTotalWeights[i];

            for (ui32 leaf = 0; leaf < Structure.LeavesCount(); ++leaf) {
                derOrDer2[cursor++] /= taskWeight;
            }
        }
    }

    void WriteValueAndDerivatives(const yvector<float>& data,
                                  TPointwiseDescentPoint& point) {
        const bool normalize = IsNormalize;

        const ui32 taskCount = TaskHelpers.size();
        point.Value = 0;
        for (ui32 i = 0; i < taskCount; ++i) {
            point.Value += TaskTotalWeights && normalize ? data[i] / TaskTotalWeights[i] : data[i];
        }

        const ui32 leavesCount = Structure.LeavesCount();
        point.Gradient.resize(taskCount * leavesCount);
        point.Hessian.resize(taskCount * leavesCount);

        Copy(data.begin() + taskCount,
             data.begin() + taskCount + taskCount * leavesCount,
             point.Gradient.begin());

        if (UseNewton) {
            Copy(data.begin() + taskCount + taskCount * leavesCount,
                 data.begin() + taskCount + 2 * taskCount * leavesCount,
                 point.Hessian.begin());
        } else {
            Copy(LeafWeights.begin(), LeafWeights.end(), point.Hessian.begin());
        }

        if (normalize) {
            NormalizeDerivatives(point.Gradient);
            NormalizeDerivatives(point.Hessian);
        }

        AddRidgeRegularizer(point, Lambda);
    }

    void WriteWeights(yvector<float>& dst) {
        dst.resize(LeafWeights.size());
        Copy(LeafWeights.begin(), LeafWeights.end(), dst.begin());
    }

    void AddRidgeRegularizer(TPointwiseDescentPoint& pointInfo,
                             double lambda) {
        if (AddRidgeToTargetFunction) {
            double hingeLoss = 0;
            {
                for (auto val : pointInfo.Point) {
                    hingeLoss += val * val;
                }
                hingeLoss *= lambda / 2;
            }
            pointInfo.Value -= hingeLoss;
        }

        for (uint i = 0; i < pointInfo.Gradient.size(); ++i) {
            pointInfo.AddToHessianDiag(i, Lambda);
            if (AddRidgeToTargetFunction) {
                pointInfo.Gradient[i] -= Lambda * pointInfo.Point[i];
            }
        }
    }

    double GradientNorm(const TPointwiseDescentPoint& pointInfo) {
        const auto& gradient = pointInfo.Gradient;
        double gradNorm = 0;

        for (uint leaf = 0; leaf < gradient.size(); ++leaf) {
            const double grad = gradient[leaf];
            gradNorm += grad * grad;
        }

        return sqrt(gradNorm);
    }

    void CreatePartStats() {
        const ui32 leafCount = Structure.LeavesCount();
        const ui32 taskCount = TaskHelpers.size();
        ui32 sumCount = UseNewton ? 2 : 1;
        auto mapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(leafCount * taskCount * sumCount + taskCount);
        PartStats.Reset(mapping);
    }

    void ComputePartWeights() {
        const ui32 leavesCount = Structure.LeavesCount();
        const ui32 taskCount = TaskHelpers.size();

        CB_ENSURE(PartStats.GetMapping().DeviceSlice(0).Size() >= taskCount * leavesCount);

        RunInStreams(taskCount, Min<ui32>(taskCount, 8), [&](ui32 taskId, ui32 streamId) {
            TEstimationTaskHelper& taskHelper = TaskHelpers[taskId];
            auto weightBuffer = NCudaLib::ParallelStripeView(PartStats,
                                                             TSlice(taskId * leavesCount, (taskId + 1) * leavesCount));

            taskHelper.ProjectWeights(weightBuffer, streamId);
        });

        auto weightsBufferSlice = NCudaLib::ParallelStripeView(PartStats,
                                                               TSlice(0, taskCount * leavesCount));
        TSlice reduceSlice = weightsBufferSlice.GetMapping().DeviceSlice(0);

        weightsBufferSlice
            .CreateReader()
            .SetReadSlice(reduceSlice)
            .SetFactorSlice(reduceSlice)
            .ReadReduce(LeafWeights);

        TaskTotalWeights.resize(taskCount);

        ui32 cursor = 0;
        for (ui32 i = 0; i < TaskHelpers.size(); ++i) {
            for (ui32 leaf = 0; leaf < Structure.LeavesCount(); ++leaf) {
                TaskTotalWeights[i] += LeafWeights[cursor++];
            }
        }
    }

    template <class TOracle,
              class TBacktrackingStepEstimator>
    friend class TNewtonLikeWalker;

public:
    TObliviousTreeLeavesEstimator(const TObliviousTreeStructure& structure,
                                  const TBinarizedFeaturesManager& featuresManager,
                                  TScopedCacheHolder& scopedCache,
                                  bool useNewton,
                                  double lambda,
                                  ui32 iterations,
                                  bool normalize,
                                  bool addRidgeToTargetFunction)
        : Structure(structure)
        , FeaturesManager(featuresManager)
        , ScopedCache(scopedCache)
        , UseNewton(useNewton)
        , Lambda(lambda)
        , Iterations(iterations)
        , IsNormalize(normalize)
        , AddRidgeToTargetFunction(addRidgeToTargetFunction)
    {
    }

    TObliviousTreeLeavesEstimator& AddEstimationTask(TMirrorTarget&& target,
                                                     TConstVec&& current,
                                                     TObliviousTreeModel* dst) {
        const ui32 binCount = static_cast<ui32>(1 << Structure.GetDepth());

        const auto& docBins = GetBinsForModel(ScopedCache,
                                              FeaturesManager,
                                              target.GetDataSet(),
                                              Structure);
        TEstimationTaskHelper task;

        auto strippedTarget = MakeStripeTarget(target);
        task.Bins = strippedTarget.template CreateGpuBuffer<ui32>();
        Gather(task.Bins, docBins, strippedTarget.GetIndices());

        task.Baseline = TStripeBuffer<float>::CopyMapping(task.Bins);
        task.Cursor = TStripeBuffer<float>::CopyMapping(task.Bins);

        auto indices = strippedTarget.template CreateGpuBuffer<ui32>();
        MakeSequence(indices);
        ReorderBins(task.Bins, indices, 0, Structure.GetDepth());

        Gather(task.Baseline,
               NCudaLib::StripeView(current, indices.GetMapping()),
               indices);

        auto offsetsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(binCount + 1);
        task.Offsets = TCudaBuffer<ui32, NCudaLib::TStripeMapping>::Create(offsetsMapping);
        UpdatePartitionOffsets(task.Bins, task.Offsets);

        task.DerCalcer = CreateDerCalcer(std::move(strippedTarget), std::move(indices));

        TaskHelpers.push_back(std::move(task));
        WriteDst.push_back(dst);

        return *this;
    }

    void Estimate() {
        CreatePartStats();
        ComputePartWeights();
        const ui32 leavesCount = Structure.LeavesCount();

        LeafValues = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(leavesCount * TaskHelpers.size()));
        FillBuffer(LeafValues, 0.0f);

        TNewtonLikeWalker<TObliviousTreeLeavesEstimator, TSimpleStepEstimator> newtonLikeWalker(*this, Iterations);

        yvector<float> point;
        point.resize(leavesCount * TaskHelpers.size());
        point = newtonLikeWalker.Estimate(point);

        for (ui32 taskId = 0; taskId < TaskHelpers.size(); ++taskId) {
            float* values = ~point + taskId * leavesCount;
            TObliviousTreeModel& dst = *WriteDst[taskId];
            yvector<float> leaves(leavesCount);
            for (ui32 i = 0; i < leavesCount; ++i) {
                leaves[i] = values[i];
            }
            dst.UpdateLeaves(std::move(leaves));
        }
    }
};
