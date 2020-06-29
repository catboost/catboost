#pragma once

#include "non_zero_filter.h"
#include <catboost/private/libs/options/bootstrap_options.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/filter.h>
#include <catboost/cuda/cuda_util/helpers.h>

#include <util/generic/ymath.h>

namespace NCatboostCuda {
    template <class TMapping>
    class TBootstrap {
    public:
        TBootstrap(const NCatboostOptions::TBootstrapConfig& config, TMaybe<float> leavesL1Sum = Nothing())
            : Config(config)
            , Lambda(config.GetMvsReg())
        {
            if (!Lambda.Defined()) {
                Lambda = leavesL1Sum;
            }
        }

        void Reset(TMaybe<float> leavesL1Sum) {
            if (!(Config.GetMvsReg().Defined())) {
                Lambda = leavesL1Sum;
            }
        }

        static void Bootstrap(
            const NCatboostOptions::TBootstrapConfig& config,
            TGpuAwareRandom& random,
            TCudaBuffer<float, TMapping>& weights,
            TMaybe<float> mvsLambda = Nothing(),
            const TCudaBuffer<float, TMapping>* derPtr = nullptr)
        {
            auto& seeds = random.GetGpuSeeds<TMapping>();

            switch (config.GetBootstrapType()) {
                case EBootstrapType::Poisson: {
                    PoissonBootstrap(seeds, weights, config.GetPoissonLambda());
                    break;
                }
                case EBootstrapType::Bayesian: {
                    BayesianBootstrap(seeds, weights, config.GetBaggingTemperature());
                    break;
                }
                case EBootstrapType::Bernoulli: {
                    UniformBootstrap(seeds, weights, config.GetTakenFraction());
                    break;
                }
                case EBootstrapType::MVS: {
                    const ui32 size = derPtr->GetObjectsSlice().Size();

                    if (!mvsLambda.Defined()) {
                        mvsLambda = Sqr(ReduceToHost(*derPtr, EOperatorType::L1Sum) / size);
                    }
                    MvsBootstrapRadixSort(seeds, weights, *derPtr, config.GetTakenFraction(), mvsLambda.GetRef());
                    break;
                }
                case EBootstrapType::No: {
                    break;
                }
                default: {
                    ythrow TCatBoostException() << "Unknown bootstrap type " << config.GetBootstrapType();
                }
            }
        }

        void Bootstrap(
            TGpuAwareRandom& random,
            TCudaBuffer<float, TMapping>& weights)
        {
            Bootstrap(Config, random, weights, Nothing(), nullptr);
        }

        TCudaBuffer<float, TMapping> BootstrappedWeights(TGpuAwareRandom& random,
                                                         TCudaBuffer<float, TMapping>* derPtr=nullptr) {
            TCudaBuffer<float, TMapping> weights = TCudaBuffer<float, TMapping>::Create(derPtr->GetMapping());
            if (Config.GetBootstrapType() != EBootstrapType::MVS) {
                FillBuffer(weights, 1.0f);
            }
            Bootstrap(Config, random, weights, Lambda, derPtr);
            return weights;
        }

        void BootstrapAndFilter(TGpuAwareRandom& random,
                                TCudaBuffer<float, TMapping>& der,
                                TCudaBuffer<float, TMapping>& weights,
                                TCudaBuffer<ui32, TMapping>& indices)
        {
            if (Config.GetBootstrapType() != EBootstrapType::No) {
                auto tmp = BootstrappedWeights(random, &der);
                MultiplyVector(der, tmp);
                MultiplyVector(weights, tmp);

                if (AreZeroWeightsAfterBootstrap(Config.GetBootstrapType())) {
                    TCudaBuffer<ui32, TMapping> nzIndices;
                    FilterZeroEntries(&tmp, &nzIndices);

                    tmp.Reset(der.GetMapping());

                    tmp.Copy(der);
                    der.Reset(nzIndices.GetMapping());
                    Gather(der, tmp, nzIndices);

                    tmp.Copy(weights);
                    weights.Reset(nzIndices.GetMapping());
                    Gather(weights, tmp, nzIndices);

                    auto tmpUi32 = tmp.template ReinterpretCast<ui32>();
                    tmpUi32.Copy(indices);
                    indices.Reset(nzIndices.GetMapping());
                    Gather(indices, tmpUi32, nzIndices);
                }
            }
        }

        static bool BootstrapAndFilter(const NCatboostOptions::TBootstrapConfig& config,
                                       TGpuAwareRandom& random,
                                       const TMapping& mapping,
                                       TCudaBuffer<float, TMapping>* bootstrappedWeights,
                                       TCudaBuffer<ui32, TMapping>* bootstrappedIndices) {
            Y_ASSERT(config.GetBootstrapType() != EBootstrapType::MVS);
            Y_ASSERT(mapping.GetObjectsSlice().Size());
            if (config.GetBootstrapType() != EBootstrapType::No) {
                bootstrappedWeights->Reset(mapping);
                FillBuffer(*bootstrappedWeights, 1.0f);
                Bootstrap(config, random, *bootstrappedWeights, Nothing(), nullptr);

                if (AreZeroWeightsAfterBootstrap(config.GetBootstrapType())) {
                    FilterZeroEntries(bootstrappedWeights, bootstrappedIndices);
                    return false;
                } else {
                    bootstrappedIndices->Reset(bootstrappedWeights->GetMapping());
                    MakeSequence(*bootstrappedIndices);
                    return true;
                }
            } else {
                bootstrappedWeights->Reset(mapping);
                bootstrappedIndices->Reset(mapping);
                FillBuffer(*bootstrappedWeights, 1.0f);
                MakeSequence(*bootstrappedIndices);
                return true;
            }
        }

    private:
        const NCatboostOptions::TBootstrapConfig& Config;
        TMaybe<float> Lambda;
    };

    extern template class TBootstrap<NCudaLib::TStripeMapping>;

    extern template class TBootstrap<NCudaLib::TSingleMapping>;

    extern template class TBootstrap<NCudaLib::TMirrorMapping>;
}
