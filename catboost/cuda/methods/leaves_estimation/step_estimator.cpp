#include "step_estimator.h"
#include <util/generic/yexception.h>
#include <catboost/libs/helpers/exception.h>

namespace NCatboostCuda {
    namespace {
        class TSkipStepEstimation: public IStepEstimator {
        public:
            bool IsSatisfied(double,
                             double,
                             const TVector<float>&) const override {
                return true;
            }
        };

        class TSimpleStepEstimator: public IStepEstimator {
        private:
            double FunctionValue;

        public:
            explicit TSimpleStepEstimator(const double functionValue)
                : FunctionValue(functionValue)
            {
            }

            bool IsSatisfied(double,
                             double nextFuncValue,
                             const TVector<float>&) const override {
                return FunctionValue <= nextFuncValue;
            }
        };

        class TArmijoStepEstimation: public IStepEstimator {
        private:
            const double C = 1e-5;

            double FunctionValue;
            const TVector<float>& Gradient;
            const TVector<float>& Direction;
            double DirGradDot;

        public:
            TArmijoStepEstimation(const double functionValue,
                                  const TVector<float>& gradient,
                                  const TVector<float>& direction)
                : FunctionValue(functionValue)
                , Gradient(gradient)
                , Direction(direction)
            {
                DirGradDot = 0;
                for (ui32 i = 0; i < Gradient.size(); ++i) {
                    DirGradDot += Gradient[i] * Direction[i];
                }
            }

            bool IsSatisfied(double step,
                             double nextFuncValue,
                             const TVector<float>& nextFuncGradient) const override {
                double directionNextGradDot = 0;
                for (ui32 i = 0; i < Gradient.size(); ++i) {
                    directionNextGradDot += Gradient[i] * nextFuncGradient[i];
                }
                return (nextFuncValue >= (FunctionValue + C * step * DirGradDot));
            }
        };
    }

    THolder<IStepEstimator> CreateStepEstimator(ELeavesEstimationStepBacktracking type,
                                                const double currentPoint,
                                                const TVector<float>& gradientAtPoint,
                                                const TVector<float>& moveDirection) {
        switch (type) {
            case ELeavesEstimationStepBacktracking::None: {
                return new TSkipStepEstimation;
            }
            case ELeavesEstimationStepBacktracking::AnyImprovment: {
                return new TSimpleStepEstimator(currentPoint);
            }
            case ELeavesEstimationStepBacktracking::Armijo: {
                return new TArmijoStepEstimation(currentPoint, gradientAtPoint, moveDirection);
            }
            default: {
                ythrow TCatboostException() << "Unknown step estimator type " << type;
            }
        }
    }
}
