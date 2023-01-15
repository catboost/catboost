#pragma once

#include "leaves_estimation_config.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/targets/oracle_type.h>
#include <catboost/private/libs/algo_helpers/langevin_utils.h>
#include <catboost/cuda/cuda_util/gpu_random.h>

namespace NCatboostCuda {
    class ILeavesEstimationOracle {
    public:
        virtual ~ILeavesEstimationOracle() {
        }

        virtual TVector<float> MakeEstimationResult(const TVector<float>& point) const = 0;
        virtual ui32 PointDim() const = 0;
        virtual ui32 HessianBlockSize() const = 0;
        virtual void Regularize(TVector<float>* point) = 0;
        virtual void MoveTo(const TVector<float>& point) = 0;
        virtual void WriteValueAndFirstDerivatives(double* value,
                                                   TVector<double>* gradient) = 0;
        //second ders are blocked hessian. we split gradient in blocks, for each block we have matrix (e.g. matrix per leaf)
        //TODO(noxoomo): make this interface fill only lower-triangle of second der
        virtual void WriteSecondDerivatives(TVector<double>* secondDer) = 0;
        virtual void WriteWeights(TVector<double>* dst) = 0;
        virtual TVector<float> EstimateExact() = 0;
        virtual void AddLangevinNoiseToDerivatives(TVector<double>* derivatives,
                                                   NPar::ILocalExecutor* localExecutor) = 0;
    };

    class ILeavesEstimationOracleFactory {
    public:
        virtual ~ILeavesEstimationOracleFactory() {
        }

        virtual THolder<ILeavesEstimationOracle> Create(const TLeavesEstimationConfig& config,
                                                        TStripeBuffer<const float>&& baseline,
                                                        TStripeBuffer<ui32>&& bins,
                                                        ui32 binCount,
                                                        TGpuAwareRandom& random) const = 0;
    };

    inline void RegularizeImpl(const TLeavesEstimationConfig& config, const TConstArrayRef<double> binWeights, TVector<float>* point, ui32 approxDim = 1) {
        CB_ENSURE_INTERNAL(binWeights.size() * approxDim == point->size(),
            "Inappropriate point and weight vector sizes : " << binWeights.size() << " * " << approxDim  << " and " << point->size());
        for (size_t bin = 0; bin < binWeights.size(); ++bin) {
            if (binWeights[bin] < config.MinLeafWeight) {
                for (ui32 dim = 0; dim < approxDim; ++dim) {
                    (*point)[bin * approxDim + dim] = 0;
                }
            }
        }
    }

    inline void AddLangevinNoise(const TLeavesEstimationConfig& config,
                                 TVector<double>* derivatives,
                                 NPar::ILocalExecutor* localExecutor,
                                 ui64 randomSeed) {
        AddLangevinNoiseToDerivatives(config.DiffusionTemperature,
                                      config.LearningRate,
                                      randomSeed,
                                      derivatives,
                                      localExecutor);
    }

    inline void AddRigdeRegulaizationIfNecessary(const TLeavesEstimationConfig& config, const TVector<float>& point, double* value, TVector<double>* gradient) {
        const double lambda = config.Lambda;
        if (config.AddRidgeToTargetFunction) {
            double hingeLoss = 0;
            {
                for (const auto& val : point) {
                    hingeLoss += val * val;
                }
                hingeLoss *= lambda / 2;
            }
            (*value) -= hingeLoss;

            for (size_t i = 0; i < gradient->size(); ++i) {
                (*gradient)[i] -= lambda * point[i];
            }
        }
    }

}
