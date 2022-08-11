#include "step_estimator.h"
#include <util/generic/yexception.h>
#include <catboost/libs/helpers/exception.h>

namespace NCatboostCuda {
    namespace {
        class TSkipStepEstimation: public IStepEstimator {
        public:
            bool IsSatisfied(double,
                             double,
                             const TVector<double>&) const override {
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
                             const TVector<double>&) const override {
                return FunctionValue <= nextFuncValue;
            }
        };

        class TArmijoStepEstimation: public IStepEstimator {
        private:
            const double C = 1e-5;

            double FunctionValue;
            const TVector<double>& Gradient;
            const TVector<float>& Direction;
            double DirGradDot;

        public:
            TArmijoStepEstimation(const double functionValue,
                                  const TVector<double>& gradient,
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
                             const TVector<double>& /*nextFuncGradient*/) const override {
                /*double directionNextGradDot = 0;
                for (ui32 i = 0; i < Gradient.size(); ++i) {
                    directionNextGradDot += Gradient[i] * nextFuncGradient[i];
                }*/
                return (nextFuncValue >= (FunctionValue + C * step * DirGradDot));
            }
        };
    }

    THolder<IStepEstimator> CreateStepEstimator(ELeavesEstimationStepBacktracking type,
                                                const double currentPoint,
                                                const TVector<double>& gradientAtPoint,
                                                const TVector<float>& moveDirection) {
        switch (type) {
            case ELeavesEstimationStepBacktracking::No: {
                return MakeHolder<TSkipStepEstimation>();
            }
            case ELeavesEstimationStepBacktracking::AnyImprovement: {
                return MakeHolder<TSimpleStepEstimator>(currentPoint);
            }
            case ELeavesEstimationStepBacktracking::Armijo: {
                return MakeHolder<TArmijoStepEstimation>(currentPoint, gradientAtPoint, moveDirection);
            }
            default: {
                ythrow TCatBoostException() << "Unknown step estimator type " << type;
            }
        }
    }
}
