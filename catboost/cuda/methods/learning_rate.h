#pragma once

#include <catboost/libs/helpers/exception.h>

namespace NCatboostCuda {
    class TLearningRate {
    public:
        explicit TLearningRate(double rate)
            : LearningRate(rate)
        {
        }

        float Step(const int iteration) const {
            Y_UNUSED(iteration);
            return static_cast<float>(LearningRate);
        }

    private:
        double LearningRate;
    };
}
