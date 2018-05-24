#pragma once

#include "step_estimator.h"
#include "diagonal_descent.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <util/generic/algorithm.h>

namespace NCatboostCuda {

    template <class TDescentPoint>
    inline void AddRidgeRegularization(double lambda,
                                       bool addToTarget,
                                       TDescentPoint* pointInfo) {
        if (addToTarget) {
            double hingeLoss = 0;
            {
                for (const auto& val : pointInfo->Point) {
                    hingeLoss += val * val;
                }
                hingeLoss *= lambda / 2;
            }
            pointInfo->Value -= hingeLoss;
        }

        for (ui32 i = 0; i < pointInfo->Gradient.size(); ++i) {
            pointInfo->AddToHessianDiag(i, static_cast<float>(lambda));
            if (addToTarget) {
                pointInfo->Gradient[i] -= lambda * pointInfo->Point[i];
            }
        }
    }

    template <class TOracle>
    class TNewtonLikeWalker {
    public:
        using TDescentPoint = typename TOracle::TDescentPoint;
        TOracle& DerCalcer;
        const ui32 Iterations;
        ELeavesEstimationStepBacktracking StepEstimationType;
    public:
        TNewtonLikeWalker(TOracle& leavesProjector,
                          const ui32 iterations,
                          ELeavesEstimationStepBacktracking backtrackingType
        )
            : DerCalcer(leavesProjector)
            , Iterations(iterations)
            , StepEstimationType(backtrackingType) {
        }

        inline TVector<float> Estimate(const TVector<float>& startPoint) {
            const ui32 dim = (const ui32)startPoint.size();

            TDirectionEstimator estimator([&]() -> TDescentPoint {
                auto currentPointInfo = TOracle::Create(dim);
                currentPointInfo.Point = startPoint;
                DerCalcer.MoveTo(currentPointInfo.GetCurrentPoint());
                DerCalcer.ComputeValueAndDerivatives(currentPointInfo);
                return currentPointInfo;
            }());

            if (Iterations == 1) {
                TVector<float> result;
                estimator.MoveInOptimalDirection(result, 1.0);
                DerCalcer.Regularize(result);
                return result;
            }

            {
                const auto& pointInfo = estimator.GetCurrentPoint();
                double gradNorm = pointInfo.GradientNorm();
                MATRIXNET_INFO_LOG << "Initial gradient norm: " << gradNorm << " Func value: " << pointInfo.Value << Endl;
            }

            TDescentPoint nextPointInfo = TOracle::Create(dim);

            bool updated = false;
            for (ui32 iteration = 1; iteration < Iterations;) {
                const auto& currentPointInfo = estimator.GetCurrentPoint();
                const auto& moveDirection = estimator.GetDirection();

                double step = 1.0;
                auto stepEstimation = CreateStepEstimator(StepEstimationType,
                                                          currentPointInfo.Value,
                                                          currentPointInfo.Gradient,
                                                          moveDirection);

                for (; iteration < Iterations || (!updated && iteration < 100); ++iteration, step /= 2) {
                    auto& nextPoint = nextPointInfo.Point;
                    estimator.MoveInOptimalDirection(nextPoint, step);

                    DerCalcer.Regularize(nextPoint);
                    DerCalcer.MoveTo(nextPoint);
                    //compute value and diagonal ders. it's faster to do all at once
                    DerCalcer.ComputeValueAndDerivatives(nextPointInfo);

                    if (stepEstimation->IsSatisfied(step,
                                                   nextPointInfo.Value,
                                                   nextPointInfo.Gradient)) {
                        double gradNorm = nextPointInfo.GradientNorm();

                        MATRIXNET_INFO_LOG
                            << "Next point gradient norm: " << gradNorm << " Func value: " << nextPointInfo.Value
                            << " Moved with step: " << step << Endl;
                        estimator.NextPoint(nextPointInfo);
                        ++iteration;
                        updated = true;
                        break;
                    }
                }
            }

            return estimator.GetCurrentPoint().Point;
        }
    };
}
