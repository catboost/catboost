#pragma once

#include "step_estimator.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <util/generic/algorithm.h>
#include <catboost/libs/helpers/matrix.h>

namespace NCatboostCuda {
    struct TPointWithFuncInfo {
        double Value = 0.0;

        TVector<float> Point;
        TVector<float> Gradient;
        TVector<float> Hessian;

        double GradientNorm() const {
            const auto& gradient = Gradient;
            double gradNorm = 0;

            for (size_t leaf = 0; leaf < gradient.size(); ++leaf) {
                const double grad = gradient[leaf];
                gradNorm += grad * grad;
            }
            return sqrt(gradNorm);
        }
    };

    class TDirectionEstimator {
    public:
        TDirectionEstimator(TPointWithFuncInfo&& point)
            : CurrentPoint(std::move(point))
        {
            UpdateMoveDirection();
        }

        void NextPoint(const TPointWithFuncInfo& pointInfo) {
            CurrentPoint = pointInfo;
            UpdateMoveDirection();
        }

        const TVector<float>& GetDirection() {
            return MoveDirection;
        }

        const TPointWithFuncInfo& GetCurrentPoint() const {
            return CurrentPoint;
        }

        void MoveInOptimalDirection(TVector<float>& point,
                                    double step) const {
            point.resize(CurrentPoint.Point.size());

            Copy(CurrentPoint.Point.begin(),
                 CurrentPoint.Point.end(),
                 point.begin());

            for (ui32 leaf = 0; leaf < point.size(); ++leaf) {
                point[leaf] += step * MoveDirection[leaf];
            }
        }

    private:
        void UpdateMoveDirection() {
            if (CurrentPoint.Gradient.size() == CurrentPoint.Hessian.size()) {
                UpdateMoveDirectionDiagonal();
            } else {
                UpdateMoveDirectionNonDiagonal();
            }
        }
        void UpdateMoveDirectionDiagonal() {
            MoveDirection.resize(CurrentPoint.Point.size());

            for (ui32 i = 0; i < CurrentPoint.Gradient.size(); ++i) {
                MoveDirection[i] = CurrentPoint.Hessian[i] > 0 ? CurrentPoint.Gradient[i] / (CurrentPoint.Hessian[i] + 1e-20f) : 0;
            }
        }

        void UpdateMoveDirectionNonDiagonal() {
            const ui32 rowSize = CurrentPoint.Gradient.size();
            CB_ENSURE(rowSize * rowSize == CurrentPoint.Hessian.size());
            MoveDirection.resize(rowSize);

            TVector<double> sigma(CurrentPoint.Hessian.size());
            TVector<double> solution(CurrentPoint.Gradient.size());

            for (size_t i = 0; i < sigma.size(); ++i) {
                sigma[i] = CurrentPoint.Hessian[i];
            }

            for (size_t i = 0; i < solution.size(); ++i) {
                solution[i] = CurrentPoint.Gradient[i];
            }

            SolveLinearSystemCholesky(&sigma,
                                      &solution);

            for (uint i = 0; i < solution.size(); ++i) {
                MoveDirection[i] = (float)solution[i];
            }
        }

    private:
        TPointWithFuncInfo CurrentPoint;
        TVector<float> MoveDirection;
    };

    template <class TOracle>
    class TNewtonLikeWalker {
    public:
        TOracle& Oracle;
        const ui32 Iterations;
        ELeavesEstimationStepBacktracking StepEstimationType;

    public:
        TNewtonLikeWalker(TOracle& oracle,
                          const ui32 iterations,
                          ELeavesEstimationStepBacktracking backtrackingType)
            : Oracle(oracle)
            , Iterations(iterations)
            , StepEstimationType(backtrackingType)
        {
        }

        inline TVector<float> Estimate(TVector<float> startPoint) {
            startPoint.resize(Oracle.PointDim());

            TDirectionEstimator estimator([&]() -> TPointWithFuncInfo {
                TPointWithFuncInfo point;
                point.Point = startPoint;
                Oracle.MoveTo(point.Point);
                Oracle.WriteValueAndFirstDerivatives(&point.Value,
                                                     &point.Gradient);
                Oracle.WriteSecondDerivatives(&point.Hessian);

                return point;
            }());

            if (Iterations == 1) {
                TVector<float> result;
                estimator.MoveInOptimalDirection(result, 1.0);
                Oracle.Regularize(&result);
                return result;
            }

            {
                const auto& pointInfo = estimator.GetCurrentPoint();
                double gradNorm = pointInfo.GradientNorm();
                MATRIXNET_INFO_LOG << "Initial gradient norm: " << gradNorm << " Func value: " << pointInfo.Value << Endl;
            }

            TPointWithFuncInfo nextPointWithFuncInfo = estimator.GetCurrentPoint();

            bool updated = false;
            for (ui32 iteration = 0; iteration < Iterations;) {
                const auto& currentPointInfo = estimator.GetCurrentPoint();
                const auto& moveDirection = estimator.GetDirection();

                double step = 1.0;
                auto stepEstimation = CreateStepEstimator(StepEstimationType,
                                                          currentPointInfo.Value,
                                                          currentPointInfo.Gradient,
                                                          moveDirection);

                for (; iteration < Iterations || (!updated && iteration < 100); ++iteration, step /= 2) {
                    auto& nextPoint = nextPointWithFuncInfo.Point;
                    estimator.MoveInOptimalDirection(nextPoint, step);

                    Oracle.Regularize(&nextPoint);
                    Oracle.MoveTo(nextPoint);
                    Oracle.WriteValueAndFirstDerivatives(&nextPointWithFuncInfo.Value,
                                                         &nextPointWithFuncInfo.Gradient);

                    if (stepEstimation->IsSatisfied(step,
                                                    nextPointWithFuncInfo.Value,
                                                    nextPointWithFuncInfo.Gradient))
                    {
                        Oracle.WriteSecondDerivatives(&nextPointWithFuncInfo.Hessian);
                        double gradNorm = nextPointWithFuncInfo.GradientNorm();

                        MATRIXNET_INFO_LOG
                            << "Next point gradient norm: " << gradNorm << " Func value: " << nextPointWithFuncInfo.Value
                            << " Moved with step: " << step << Endl;
                        estimator.NextPoint(nextPointWithFuncInfo);
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
