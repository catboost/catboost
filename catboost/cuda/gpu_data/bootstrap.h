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
#include <catboost/cuda/cuda_util/scan.h>

namespace NCatboostCuda {
    template <class TMapping>
    class TBootstrap {
    public:
        TBootstrap(const NCatboostOptions::TBootstrapConfig& config)
            : Config(config)
        {
        }

        static void Bootstrap(const NCatboostOptions::TBootstrapConfig& config,
                              TGpuAwareRandom& random,
                              TCudaBuffer<float, TMapping>& weights) {
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
                case EBootstrapType::No: {
                    break;
                }
                default: {
                    ythrow TCatBoostException() << "Unknown bootstrap type " << config.GetBootstrapType();
                }
            }
        }

        void Bootstrap(TGpuAwareRandom& random,
                       TCudaBuffer<float, TMapping>& weights) {
            Bootstrap(Config, random, weights);
        }

        TCudaBuffer<float, TMapping> BootstrappedWeights(TGpuAwareRandom& random,
                                                         const TMapping& mapping) {
            TCudaBuffer<float, TMapping> weights = TCudaBuffer<float, TMapping>::Create(mapping);
            FillBuffer(weights, 1.0f);
            Bootstrap(Config, random, weights);
            return weights;
        }

        void BootstrapAndFilter(TGpuAwareRandom& random,
                                TCudaBuffer<float, TMapping>& der,
                                TCudaBuffer<float, TMapping>& weights,
                                TCudaBuffer<ui32, TMapping>& indices) {
            if (Config.GetBootstrapType() != EBootstrapType::No) {
                auto tmp = BootstrappedWeights(random, der.GetMapping());

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
            CB_ENSURE(mapping.GetObjectsSlice().Size());
            if (config.GetBootstrapType() != EBootstrapType::No) {
                bootstrappedWeights->Reset(mapping);
                FillBuffer(*bootstrappedWeights, 1.0f);
                Bootstrap(config, random, *bootstrappedWeights);

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
    };

    extern template class TBootstrap<NCudaLib::TStripeMapping>;

    extern template class TBootstrap<NCudaLib::TSingleMapping>;

    extern template class TBootstrap<NCudaLib::TMirrorMapping>;
}
