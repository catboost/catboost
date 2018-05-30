#pragma once

#include "leaves_estimation_config.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>

namespace NCatboostCuda {
    class INonDiagonalOracle {
    public:
        virtual ~INonDiagonalOracle() {
        }

        virtual ui32 PointDim() const = 0;
        virtual void Regularize(TVector<float>* point) = 0;
        virtual void MoveTo(TVector<float> point) = 0;
        virtual void WriteValueAndFirstDerivatives(double* value,
                                                   TVector<float>* gradient) = 0;
        virtual void WriteSecondDerivatives(TVector<float>* secondDer) = 0;
        virtual void WriteWeights(TVector<float>* dst) = 0;
    };

    class INonDiagonalOracleFactory {
    public:
        virtual ~INonDiagonalOracleFactory() {
        }

        virtual THolder<INonDiagonalOracle> Create(const TLeavesEstimationConfig& config,
                                                   TStripeBuffer<const float>&& baseline,
                                                   TStripeBuffer<const ui32>&& bins,
                                                   ui32 binCount) const = 0;
    };

}
