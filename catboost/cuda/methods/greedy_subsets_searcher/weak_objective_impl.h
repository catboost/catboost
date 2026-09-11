#pragma once

#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/targets/weak_objective.h>

#include <util/generic/xrange.h>

namespace NCatboostCuda {
    template <class TTargetFunc>
    class TWeakObjective: public IWeakObjective, public TMoveOnly {
    public:
        using TMapping = typename TTargetFunc::TMapping;
        template <class T>
        using TBuffer = TCudaBuffer<T, TMapping>;
        using TVec = TBuffer<float>;
        using TConstVec = TBuffer<const float>;

        TWeakObjective(const TTargetFunc& target)
            : Target(target)
        {
        }

        void StochasticDer(const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                           TMaybe<float> mvsLambda,
                           bool secondDerAsWeights,
                           TOptimizationTarget* targetDer) const final {
            if (bootstrapConfig.GetBootstrapType() == EBootstrapType::MVS) {
                MvsStochasticDer(bootstrapConfig, mvsLambda, secondDerAsWeights, targetDer);
                return;
            }

            TGpuAwareRandom& random = Target.GetRandom();

            auto samplesMapping = Target.GetTarget().GetSamplesMapping();
            TStripeBuffer<float> sampledWeights;
            TStripeBuffer<ui32> sampledIndices;

            const bool isContinuousIndices = TBootstrap<NCudaLib::TStripeMapping>::BootstrapAndFilter(
                bootstrapConfig,
                random,
                samplesMapping,
                &sampledWeights,
                &sampledIndices);

            CATBOOST_DEBUG_LOG << "Sampled docs count " << sampledIndices.GetObjectsSlice().Size() << Endl;

            Target.StochasticDer(std::move(sampledWeights),
                                 std::move(sampledIndices),
                                 secondDerAsWeights,
                                 targetDer);

            targetDer->IsContinuousIndices = isContinuousIndices;
        }

        ui32 GetDim() const final {
            return Target.GetDim();
        }

        TGpuAwareRandom& GetRandom() const {
            return Target.GetRandom();
        }

    private:
        /* MVS sampling probabilities depend on the derivatives, so derivatives are computed for all samples first,
         * then the bootstrap weights are applied and samples with zero weights are filtered out
         * (the same as TBootstrap::BootstrapAndFilter does for oblivious trees).
         */
        void MvsStochasticDer(const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                              TMaybe<float> mvsLambda,
                              bool secondDerAsWeights,
                              TOptimizationTarget* targetDer) const {
            TGpuAwareRandom& random = Target.GetRandom();
            auto samplesMapping = Target.GetTarget().GetSamplesMapping();

            TStripeBuffer<float> unitWeights;
            unitWeights.Reset(samplesMapping);
            FillBuffer(unitWeights, 1.0f);

            TStripeBuffer<ui32> allIndices;
            allIndices.Reset(samplesMapping);
            MakeSequence(allIndices);

            Target.StochasticDer(std::move(unitWeights),
                                 std::move(allIndices),
                                 secondDerAsWeights,
                                 targetDer);

            auto& stats = targetDer->StatsToAggregate;
            const ui32 statsCount = static_cast<ui32>(stats.GetColumnCount());
            // column 0 is weights (or second derivatives), other columns are derivatives
            CB_ENSURE(statsCount == 2, "MVS bootstrap is not supported for multidimensional targets on GPU");

            TStripeBuffer<float> ders;
            ders.Reset(stats.GetMapping());
            {
                auto dersView = stats.ColumnView(1);
                ders.Copy(dersView);
            }

            TMaybe<float> lambda = bootstrapConfig.GetMvsReg();
            if (!lambda.Defined()) {
                lambda = mvsLambda;
            }

            TStripeBuffer<float> bootstrappedWeights;
            bootstrappedWeights.Reset(stats.GetMapping());
            TBootstrap<NCudaLib::TStripeMapping>::Bootstrap(bootstrapConfig,
                                                            random,
                                                            bootstrappedWeights,
                                                            lambda,
                                                            &ders);

            for (auto column : xrange(statsCount)) {
                auto columnView = stats.ColumnView(column);
                MultiplyVector(columnView, bootstrappedWeights);
            }

            TStripeBuffer<ui32> nzIndices;
            FilterZeroEntries(&bootstrappedWeights, &nzIndices);

            CATBOOST_DEBUG_LOG << "Sampled docs count " << nzIndices.GetObjectsSlice().Size() << Endl;

            TStripeBuffer<float> sampledStats;
            sampledStats.Reset(nzIndices.GetMapping(), statsCount);
            for (auto column : xrange(statsCount)) {
                auto srcView = stats.ColumnView(column);
                auto dstView = sampledStats.ColumnView(column);
                Gather(dstView, srcView, nzIndices);
            }

            TStripeBuffer<ui32> sampledIndices;
            sampledIndices.Reset(nzIndices.GetMapping());
            Gather(sampledIndices, targetDer->Indices, nzIndices);

            targetDer->StatsToAggregate = std::move(sampledStats);
            targetDer->Indices = std::move(sampledIndices);
            targetDer->IsContinuousIndices = false;
        }

    private:
        const TTargetFunc& Target;
    };

}
