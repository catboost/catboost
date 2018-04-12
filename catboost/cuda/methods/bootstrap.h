#pragma once

#include <catboost/libs/options/bootstrap_options.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/filter.h>
#include <catboost/cuda/cuda_util/helpers.h>

namespace NCatboostCuda {
    template <class TMapping>
    class TBootstrap {
    public:
        TBootstrap(const NCatboostOptions::TBootstrapConfig& config,
                   ui64 seed)
            : Config(config)
        {
            NCudaLib::TDistributedObject<ui64> maxSeedCount = CreateDistributedObject<ui64>(256 * 256);

            auto mapping = CreateMapping<TMapping>(maxSeedCount);
            Seeds.Reset(mapping);

            TRandom random(seed);
            GenerateSeedsPointwise(Seeds, random);
        }

        TBootstrap& Bootstrap(TCudaBuffer<float, TMapping>& weights) {
            switch (Config.GetBootstrapType()) {
                case EBootstrapType::Poisson: {
                    PoissonBootstrap(Seeds, weights, Config.GetPoissonLambda());
                    break;
                }
                case EBootstrapType::Bayesian: {
                    BayesianBootstrap(Seeds, weights, Config.GetBaggingTemperature());
                    break;
                }
                case EBootstrapType::Bernoulli: {
                    UniformBootstrap(Seeds, weights, Config.GetTakenFraction());
                    break;
                }
                case EBootstrapType::No: {
                    break;
                }
                default: {
                    ythrow TCatboostException() << "Unknown bootstrap type " << Config.GetBootstrapType();
                }
            }
            return *this;
        }

        TCudaBuffer<float, TMapping> BootstrappedWeights(const TMapping& mapping) {
            TCudaBuffer<float, TMapping> weights = TCudaBuffer<float, TMapping>::Create(mapping);
            FillBuffer(weights, 1.0f);
            Bootstrap(weights);
            return weights;
        }

        void BootstrapAndFilter(TCudaBuffer<float, TMapping>& der,
                                TCudaBuffer<float, TMapping>& weights,
                                TCudaBuffer<ui32, TMapping>& indices) {
            if (Config.GetBootstrapType() != EBootstrapType::No) {
                auto tmp = BootstrappedWeights(der.GetMapping());

                MultiplyVector(der, tmp);
                MultiplyVector(weights, tmp);

                //TODO(noxoomo): check it and uncomment
                if (AreZeroWeightsAfterBootstrap(Config.GetBootstrapType())) {
                    TCudaBuffer<ui32, TMapping> tmpIndices;
                    tmpIndices.Reset(tmp.GetMapping());
                    MakeSequence(tmpIndices);
                    RadixSort(tmp, tmpIndices, true);

                    auto nzSizes = NonZeroSizes(tmp);
                    TVector<ui32> nzSizesMaster;
                    nzSizes.Read(nzSizesMaster);

                    auto nzMapping = nzSizes.GetMapping().Transform([&](const TSlice& slice) {
                        CB_ENSURE(slice.Size() == 1);
                        return nzSizesMaster[slice.Left];
                    });

                    tmpIndices.Reset(nzMapping);

                    tmp.Copy(der);
                    der.Reset(nzMapping);
                    Gather(der, tmp, tmpIndices);

                    tmp.Copy(weights);
                    weights.Reset(nzMapping);
                    Gather(weights, tmp, tmpIndices);

                    auto tmpUi32 = tmp.template ReinterpretCast<ui32>();
                    tmpUi32.Copy(indices);
                    indices.Reset(nzMapping);
                    Gather(indices, tmpUi32, tmpIndices);
                }
            }
        }

    private:
        const NCatboostOptions::TBootstrapConfig& Config;
        TCudaBuffer<ui64, TMapping> Seeds;

        inline void GenerateSeedsPointwise(TCudaBuffer<ui64, TMapping>& seeds,
                                           TRandom& random) const {
            TVector<ui64> seedsCpu(seeds.GetObjectsSlice().Size());
            for (ui32 i = 0; i < seeds.GetObjectsSlice().Size(); ++i) {
                seedsCpu[i] = random.NextUniformL();
            }
            seeds.Write(seedsCpu);
        }
    };
}
