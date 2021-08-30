#pragma once

#include "descent_helpers.h"
#include "leaves_estimation_helper.h"
#include "leaves_estimation_config.h"

#include <catboost/cuda/methods/helpers.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>
#include <catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h>

#include <library/cpp/threading/local_executor/local_executor.h>

namespace NCatboostCuda {
    struct TEstimationTaskHelper {
        THolder<IPermutationDerCalcer> DerCalcer;

        TStripeBuffer<ui32> Bins;
        TStripeBuffer<ui32> Offsets;

        TStripeBuffer<float> Baseline;
        TStripeBuffer<float> Cursor;

        TStripeBuffer<float> TmpDer;
        TStripeBuffer<float> TmpValue;
        TStripeBuffer<float> TmpDer2;

        ui32 BinCount;

        TEstimationTaskHelper() = default;

        void MoveToPoint(const TMirrorBuffer<float>& point, ui32 stream = 0);

        void ProjectWeights(TCudaBuffer<double, NCudaLib::TStripeMapping>& weightsDst,
                            ui32 streamId = 0);

        void Project(TCudaBuffer<double, NCudaLib::TStripeMapping>* value,
                     TCudaBuffer<double, NCudaLib::TStripeMapping>* der,
                     TCudaBuffer<double, NCudaLib::TStripeMapping>* der2,
                     ui32 stream = 0);

        void ComputeExact(TVector<float>& point,
                          const NCatboostOptions::TLossDescription& lossDescription,
                          ui32 stream = 0) {
            auto values = TStripeBuffer<float>::CopyMapping(Bins);
            auto weights = TStripeBuffer<float>::CopyMapping(Bins);

            DerCalcer->ComputeExactValue(Baseline.AsConstBuf(), &values, &weights, stream);
            ComputeExactApprox(Bins, values, weights, BinCount, point, lossDescription);
        }
    };

    /*
     * Oblivious tree batch estimator
     */
    class TObliviousTreeLeavesEstimator: public ILeavesEstimationOracle {
    private:
        TVector<TEstimationTaskHelper> TaskHelpers;
        TVector<TSlice> TaskSlices;

        TVector<double> TaskTotalWeights;
        TVector<double> LeafWeights;
        TMirrorBuffer<float> LeafValues;
        TCudaBuffer<double, NCudaLib::TStripeMapping> PartStats;

        const TBinarizedFeaturesManager& FeaturesManager;
        TLeavesEstimationConfig LeavesEstimationConfig;
        TVector<TObliviousTreeModel*> WriteDst;

        TVector<float> CurrentPoint;
        THolder<TVector<double>> CurrentPointInfo;
        TGpuAwareRandom& Random;

    private:
        const TVector<double>& GetCurrentPointInfo();

        TMirrorBuffer<float> LeavesView(TMirrorBuffer<float>& leaves,
                                        ui32 taskId) {
            return leaves.SliceView(TaskSlices[taskId]);
        }

        void NormalizeDerivatives(TVector<double>& derOrDer2);

        void CreatePartStats();

        void ComputePartWeights();

        TEstimationTaskHelper& NextTask(TObliviousTreeModel& model);

        TVector<float> EstimateExact() final;

    public:
        TObliviousTreeLeavesEstimator(const TBinarizedFeaturesManager& featuresManager,
                                      const TLeavesEstimationConfig& config,
                                      TGpuAwareRandom& random)
            : FeaturesManager(featuresManager)
            , LeavesEstimationConfig(config)
            , Random(random)
        {
        }

        ui32 PointDim() const final {
            CB_ENSURE(TaskHelpers.size());
            return static_cast<ui32>(TaskSlices.back().Right);
        }

        ui32 HessianBlockSize() const final {
            return 1;
        }

        TVector<float> MakeEstimationResult(const TVector<float>& point) const override final {
            return point;
        }

        void MoveTo(const TVector<float>& point) final;

        void Regularize(TVector<float>* point) final;

        void WriteValueAndFirstDerivatives(double* value,
                                           TVector<double>* gradient) final;

        void WriteSecondDerivatives(TVector<double>* secondDer) final;

        void WriteWeights(TVector<double>* dst) final {
            dst->resize(LeafWeights.size());
            Copy(LeafWeights.begin(), LeafWeights.end(), dst->begin());
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

            task.DerCalcer = CreatePermutationDerCalcer(std::move(strippedTarget), indices.AsConstBuf());
            task.BinCount = binCount;

            return *this;
        }

        template <class TTarget>
        TObliviousTreeLeavesEstimator& AddEstimationTask(const TTarget& target,
                                                         const TDocParallelDataSet& dataSet,
                                                         TStripeBuffer<const float>&& current,
                                                         TObliviousTreeModel* dst) {
            const ui32 binCount = static_cast<ui32>(1 << dst->GetStructure().GetDepth());

            TEstimationTaskHelper& task = NextTask(*dst);

            task.Bins = target.template CreateGpuBuffer<ui32>();
            {
                auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Compute bins doc-parallel");
                dst->ComputeBins(dataSet, &task.Bins);
            }

            task.Baseline = TStripeBuffer<float>::CopyMappingAndColumnCount(current);
            task.Cursor = TStripeBuffer<float>::CopyMappingAndColumnCount(current);
            auto indices = TStripeBuffer<ui32>::CopyMapping(current);

            auto offsetsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(binCount + 1);
            task.Offsets = TCudaBuffer<ui32, NCudaLib::TStripeMapping>::Create(offsetsMapping);

            MakeSequence(indices);
            ReorderBins(task.Bins, indices, 0, dst->GetStructure().GetDepth());
            Gather(task.Baseline, current, indices);

            UpdatePartitionOffsets(task.Bins,
                                   task.Offsets);

            task.DerCalcer = CreatePermutationDerCalcer(TTarget(target),
                                                        indices.AsConstBuf());
            task.BinCount = binCount;

            return *this;
        }

        void Estimate(NPar::ILocalExecutor* localExecutor);

        void AddLangevinNoiseToDerivatives(TVector<double>* derivatives,
                                           NPar::ILocalExecutor* localExecutor) override;
    };
}
