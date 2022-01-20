#pragma once

#include "weak_target_helpers.h"
#include "pointwise_optimization_subsets.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/private/libs/options/oblivious_tree_options.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/private/libs/options/boosting_options.h>

namespace NCatboostCuda {
    class IMirrorTargetWrapper: public TNonCopyable {
    public:
        virtual ~IMirrorTargetWrapper() {
        }

        virtual void GradientAtZero(TMirrorBuffer<float>& weightedDer,
                                    TMirrorBuffer<float>& weights,
                                    ui32 stream = 0) const = 0;

        virtual void NewtonAtZero(TMirrorBuffer<float>& weightedDer,
                                  TMirrorBuffer<float>& weightedDer2,
                                  ui32 stream = 0) const = 0;

        virtual const TTarget<NCudaLib::TMirrorMapping>& GetTarget() const = 0;
        virtual TGpuAwareRandom& GetRandom() const = 0;
    };

    template <class TTargetFunc>
    class TMirrorTargetWrapper: public IMirrorTargetWrapper {
    public:
        TMirrorTargetWrapper(TTargetFunc&& target)
            : Target(std::move(target))
        {
        }

        const TTarget<NCudaLib::TMirrorMapping>& GetTarget() const final {
            return Target.GetTarget();
        }

        TGpuAwareRandom& GetRandom() const final {
            return Target.GetRandom();
        }

        void GradientAtZero(TMirrorBuffer<float>& weightedDer,
                            TMirrorBuffer<float>& weights,
                            ui32 stream = 0) const final {
            Target.GradientAtZero(weightedDer, weights, stream);
        }

        void NewtonAtZero(TMirrorBuffer<float>& weightedDer,
                          TMirrorBuffer<float>& weightedDer2,
                          ui32 stream = 0) const final {
            Target.NewtonAtZero(weightedDer, weightedDer2, stream);
        };

    private:
        TTargetFunc Target;
    };

    class TFeatureParallelObliviousTreeSearcher {
    public:
        using TVec = TCudaBuffer<float, NCudaLib::TMirrorMapping>;
        using TDataSet = TFeatureParallelDataSet;

        TFeatureParallelObliviousTreeSearcher(TScopedCacheHolder& cache,
                                              TBinarizedFeaturesManager& featuresManager,
                                              const NCatboostOptions::TBoostingOptions& boostingOptions,
                                              const TFeatureParallelDataSet& dataSet,
                                              TBootstrap<NCudaLib::TMirrorMapping>& bootstrap,
                                              const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions,
                                              TGpuAwareRandom& random)
            : ScopedCache(cache)
            , FeaturesManager(featuresManager)
            , BoostingOptions(boostingOptions)
            , DataSet(dataSet)
            , CtrTargets(dataSet.GetCtrTargets())
            , Bootstrap(bootstrap)
            , TreeConfig(learnerOptions)
            , Random(random)
        {
        }

        template <class TTarget>
        TFeatureParallelObliviousTreeSearcher& AddTask(TTarget&& learnTarget,
                                                       TTarget&& testTarget) {
            CB_ENSURE(SingleTaskTarget == nullptr, "We can't mix learn/test splits and full estimation");
            FoldBasedTasks.push_back(TOptimizationTask(std::move(learnTarget),
                                                       std::move(testTarget)));
            return *this;
        }

        template <class TTarget>
        TFeatureParallelObliviousTreeSearcher& SetTarget(TTarget&& target) {
            CB_ENSURE(SingleTaskTarget == nullptr, "Target already was set");
            CB_ENSURE(FoldBasedTasks.size() == 0, "Can't mix foldBased and singleTask targets");
            SingleTaskTarget = MakeHolder<TMirrorTargetWrapper<TTarget>>((std::move(target)));
            return *this;
        }

        TFeatureParallelObliviousTreeSearcher& SetRandomStrength(double strength) {
            ModelLengthMultiplier = strength;
            return *this;
        }

        TObliviousTreeStructure Fit();

    private:
        struct TOptimizationTask: public TMoveOnly {
            THolder<IMirrorTargetWrapper> LearnTarget;
            THolder<IMirrorTargetWrapper> TestTarget;

            template <class TTargetFunc>
            TOptimizationTask(TTargetFunc&& learn,
                              TTargetFunc&& test)
                : LearnTarget(new TMirrorTargetWrapper<TTargetFunc>(std::move(learn)))
                , TestTarget(new TMirrorTargetWrapper<TTargetFunc>(std::move(test)))
            {
            }
        };

        //with first zero bit is estimation part, with first 1 bit is evaluation part
        //we store task in first bits of bin
        TVector<TSlice> MakeTaskSlices();

        ui64 GetTotalIndicesSize() const;

        TVector<TDataPartition> WriteFoldBasedInitialBins(TMirrorBuffer<ui32>& bins);

        TVector<TDataPartition> WriteSingleTaskInitialBins(TMirrorBuffer<ui32>& bins);

        void MakeDocIndicesForSingleTask(TMirrorBuffer<ui32>& indices,
                                         ui32 stream = 0);

        //if features should be accessed by order[i]
        void MakeDocIndices(TMirrorBuffer<ui32>& indices);
        //if features should be accessed by i
        void MakeDirectDocIndicesIndices(TMirrorBuffer<ui32>& indices) {
            MakeIndicesFromInversePermutation(DataSet.GetInverseIndices(), indices);
        }

        void MakeIndicesFromInversePermutationSingleTask(const TMirrorBuffer<ui32>& inversePermutation,
                                                         TMirrorBuffer<ui32>& indices) {
            CB_ENSURE(SingleTaskTarget != nullptr);
            const auto& targetIndices = SingleTaskTarget->GetTarget().GetIndices();
            indices.Reset(NCudaLib::TMirrorMapping(targetIndices.GetMapping()));
            Gather(indices,
                   inversePermutation,
                   targetIndices);
        }

        void MakeIndicesFromInversePermutation(const TMirrorBuffer<ui32>& inversePermutation,
                                               TMirrorBuffer<ui32>& indices);

        TL2Target<NCudaLib::TMirrorMapping> ComputeWeakTarget();

        TGpuAwareRandom& GetRandom() {
            return SingleTaskTarget == nullptr ? FoldBasedTasks[0].LearnTarget->GetRandom()
                                               : SingleTaskTarget->GetRandom();
        }

        TOptimizationSubsets<NCudaLib::TMirrorMapping> CreateSubsets(ui32 maxDepth, TL2Target<NCudaLib::TMirrorMapping>& src);

    private:
        TScopedCacheHolder& ScopedCache;
        //our learn algorithm could generate new features, so no const
        TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TBoostingOptions& BoostingOptions;
        const TDataSet& DataSet;
        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;

        TBootstrap<NCudaLib::TMirrorMapping>& Bootstrap;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        double ModelLengthMultiplier = 0.0;
        double ScoreStdDev = 0.0;

        //should one or another, no mixing
        TVector<TOptimizationTask> FoldBasedTasks;
        THolder<IMirrorTargetWrapper> SingleTaskTarget;
        TGpuAwareRandom& Random;
    };
}
