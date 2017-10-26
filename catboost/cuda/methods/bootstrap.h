#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_util/cpu_random.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/cuda_util/fill.h>
namespace NCatboostCuda
{
    enum class EBootstrapType
    {
        Poisson,
        Bayesian,
        DiscreteUniform,
        NoBootstrap
    };

    template<EBootstrapType>
    struct IsDiscreteWeights
    {
        bool Value = true;
    };

    template<>
    struct IsDiscreteWeights<EBootstrapType::Bayesian>
    {
        bool Value = false;
    };

    class TBootstrapConfig
    {
    public:
        float GetTakenFraction() const
        {
            return TakenFraction;
        }

        float GetLambda() const
        {
            return TakenFraction < 1 ? -log(1 - TakenFraction) : -1;
        }

        EBootstrapType GetBootstrapType() const
        {
            if (TakenFraction == 1.0)
            {
                return EBootstrapType::NoBootstrap;
            }
            return BootstrapType;
        }

        ui64 GetSeed() const
        {
            return Seed;
        }

        float GetBaggingTemperature() const
        {
            return BaggingTemperature;
        }

        void Validate() const
        {
            CB_ENSURE((TakenFraction > 0) && (TakenFraction <= 1.0f), "Taken fraction should in in (0,1]");
            CB_ENSURE(BaggingTemperature, "Bagging temperature >= 0");
        }

        template<class TConfig>
        friend
        class TOptionsBinder;

        template<class TConfig>
        friend
        class TOptionsJsonConverter;

    private:
        float TakenFraction = 0.66;
        float BaggingTemperature = 1.0;
        EBootstrapType BootstrapType = EBootstrapType::Bayesian;
        ui64 Seed = 0;
    };

    template<class TMapping>
    class TBootstrap
    {
    public:
        TBootstrap(const TMapping& suggestedSeedsMapping,
                   const TBootstrapConfig& config)
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

            TRandom random(config.GetSeed());
            WriteSeedsPointwise(Seeds, random);
        }

        TBootstrap& Bootstrap(TCudaBuffer<float, TMapping>& weights)
        {
            switch (Config.GetBootstrapType())
            {
                case EBootstrapType::Poisson:
                {
                    PoissonBootstrap(Seeds, weights, Config.GetLambda());
                    break;
                }
                case EBootstrapType::Bayesian:
                {
                    BayesianBootstrap(Seeds, weights, Config.GetBaggingTemperature());
                    break;
                }
                case EBootstrapType::DiscreteUniform:
                {
                    UniformBootstrap(Seeds, weights, Config.GetTakenFraction());
                    break;
                }
                case EBootstrapType::NoBootstrap:
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
        TBootstrapConfig Config;
        TCudaBuffer<ui64, TMapping> Seeds;

        inline void WriteSeedsPointwise(TCudaBuffer<ui64, TMapping>& seeds,
                                        TRandom& random) const
        {
            yvector<ui64> seedsCpu(seeds.GetObjectsSlice().Size());
            for (ui32 i = 0; i < seeds.GetObjectsSlice().Size(); ++i)
            {
                seedsCpu[i] = random.NextUniformL();
            }
            seeds.Write(seedsCpu);
        }
    };
}
