#pragma once

#include "descent_helpers.h"
#include "leaves_estimation_helper.h"
#include "leaves_estimation_config.h"

#include <catboost/cuda/methods/helpers.h>
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
    /*
     * Oblivious tree batch estimator
     */
    class TObliviousTreeLeavesEstimator {
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

        TVector<float> CurrentPoint;
        THolder<TVector<float>> CurrentPointInfo;

    private:
        template <class TOracle>
        friend class TNewtonLikeWalker;

        ui32 PointDim() const {
            CB_ENSURE(TaskHelpers.size());
            return static_cast<ui32>(TaskSlices.back().Right);
        }

        void MoveTo(const TVector<float>& point);

        void Regularize(TVector<float>* point);

        void WriteValueAndFirstDerivatives(double* value,
                                           TVector<float>* gradient);

        void WriteSecondDerivatives(TVector<float>* secondDer);

        void WriteWeights(TVector<float>* dst) {
            dst->resize(LeafWeights.size());
            Copy(LeafWeights.begin(), LeafWeights.end(), dst->begin());
        }

        const TVector<float>& GetCurrentPointInfo();

        TMirrorBuffer<float> LeavesView(TMirrorBuffer<float>& leaves,
                                        ui32 taskId) {
            return leaves.SliceView(TaskSlices[taskId]);
        }

        void NormalizeDerivatives(TVector<float>& derOrDer2);

        void CreatePartStats();

        void ComputePartWeights();

        TEstimationTaskHelper& NextTask(TObliviousTreeModel& model);

    public:
        TObliviousTreeLeavesEstimator(const TBinarizedFeaturesManager& featuresManager,
                                      const TLeavesEstimationConfig& config)
            : FeaturesManager(featuresManager)
            , LeavesEstimationConfig(config)
        {
        }

        template <class TTarget, class TDataSet>
        TObliviousTreeLeavesEstimator& AddEstimationTask(TScopedCacheHolder& scopedCache,
                                                         TTarget&& target,
                                                         const TDataSet& dataSet,
                                                         TMirrorBuffer<const float>&& current,
                                                         TObliviousTreeModel* dst) {
            const ui32 binCount = static_cast<ui32>(1u << dst->GetStructure().GetDepth());

            const auto& docBins = GetBinsForModel(scopedCache,
                                                  FeaturesManager,
                                                  dataSet,
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

        template <class TTarget, class TDataSet>
        TObliviousTreeLeavesEstimator& AddEstimationTask(const TTarget& target,
                                                         const TDataSet& dataSet,
                                                         TStripeBuffer<const float>&& current,
                                                         TObliviousTreeModel* dst) {
            const ui32 binCount = static_cast<ui32>(1 << dst->GetStructure().GetDepth());

            TEstimationTaskHelper& task = NextTask(*dst);

            task.Bins = target.template CreateGpuBuffer<ui32>();
            {
                auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Compute bins doc-parallel");
                ComputeBinsForModel(dst->GetStructure(),
                                    dataSet,
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

            task.DerCalcer = CreatePermutationDerCalcer(TTarget(target),
                                                        std::move(indices));

            return *this;
        }

        void Estimate();
    };
}
