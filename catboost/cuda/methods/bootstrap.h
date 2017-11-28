#pragma once

#include <catboost/libs/options/bootstrap_options.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_util/cpu_random.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/cuda_util/fill.h>

namespace NCatboostCuda
{
    template<class TMapping>
    class TBootstrap
    {
    public:
        TBootstrap(const TMapping& suggestedSeedsMapping,
                   const NCatboostOptions::TBootstrapConfig& config,
                   ui64 seed
        )
                : Config(config)
        {
            ui64 maxSeedCount = 512 * 256;

            auto mapping = suggestedSeedsMapping.Transform([&](const TSlice& slice) -> ui64
                                                           {
                                                               return std::min(maxSeedCount,
                                                                               NHelpers::CeilDivide(slice.Size(), 256) *
                                                                               256);
                                                           });

            Seeds.Reset(mapping);

            TRandom random(seed);
            WriteSeedsPointwise(Seeds, random);
        }

        TBootstrap& Bootstrap(TCudaBuffer<float, TMapping>& weights)
        {
            switch (Config.GetBootstrapType())
            {
                case EBootstrapType::Poisson:
                {
                    PoissonBootstrap(Seeds, weights, Config.GetPoissonLambda());
                    break;
                }
                case EBootstrapType::Bayesian:
                {
                    BayesianBootstrap(Seeds, weights, Config.GetBaggingTemperature());
                    break;
                }
                case EBootstrapType::Bernoulli:
                {
                    UniformBootstrap(Seeds, weights, Config.GetTakenFraction());
                    break;
                }
                case EBootstrapType::No:
                {
                    break;
                }
                default:
                {
                    ythrow TCatboostException() << "Unknown bootstrap type " << Config.GetBootstrapType();
                }
            }
            return *this;
        }

        TCudaBuffer<float, TMapping> BootstrapedWeights(const TMapping& mapping)
        {
            TCudaBuffer<float, TMapping> weights = TCudaBuffer<float, TMapping>::Create(mapping);
            FillBuffer(weights, 1.0f);
            Bootstrap(weights);
            return weights;
        }

    private:
        const NCatboostOptions::TBootstrapConfig& Config;
        TCudaBuffer<ui64, TMapping> Seeds;

        inline void WriteSeedsPointwise(TCudaBuffer<ui64, TMapping>& seeds,
                                        TRandom& random) const
        {
            TVector<ui64> seedsCpu(seeds.GetObjectsSlice().Size());
            for (ui32 i = 0; i < seeds.GetObjectsSlice().Size(); ++i)
            {
                seedsCpu[i] = random.NextUniformL();
            }
            seeds.Write(seedsCpu);
        }
    };
}
