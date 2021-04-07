#include "descent_helpers.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/private/libs/lapack/linear_system.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>

namespace NCatboostCuda {
    namespace {
        struct TPointWithFuncInfo {
            int HessianBlockSize = 0;
            double Value = 0.0;

            TVector<float> Point;
            TVector<double> Gradient;
            TVector<double> Hessian;

            explicit TPointWithFuncInfo(int hessianBlockSize)
                : HessianBlockSize(hessianBlockSize)
            {
            }

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
            TDirectionEstimator(TPointWithFuncInfo&& point, NPar::ILocalExecutor* localExecutor)
                : CurrentPoint(std::move(point))
                , LocalExecutor(localExecutor)
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
                if (CurrentPoint.HessianBlockSize == 1) {
                    UpdateMoveDirectionDiagonal();
                } else {
                    UpdateMoveDirectionBlockedHessian();
                }
            }
            void UpdateMoveDirectionDiagonal() {
                CB_ENSURE(CurrentPoint.Gradient.size() == CurrentPoint.Hessian.size());
                MoveDirection.resize(CurrentPoint.Point.size());

                for (ui32 i = 0; i < CurrentPoint.Gradient.size(); ++i) {
                    MoveDirection[i] = CurrentPoint.Hessian[i] > 0 ? CurrentPoint.Gradient[i] / (CurrentPoint.Hessian[i] + 1e-20f) : 0;
                }
            }

            void UpdateMoveDirectionBlockedHessian() {
                const ui32 numBlocks = CurrentPoint.Gradient.size() / CurrentPoint.HessianBlockSize;
                const ui32 rowSize = CurrentPoint.HessianBlockSize;
                CB_ENSURE(rowSize * rowSize * numBlocks == CurrentPoint.Hessian.size(), rowSize << " " << numBlocks);
                CB_ENSURE(rowSize * numBlocks == CurrentPoint.Point.size());

                MoveDirection.resize(rowSize * numBlocks);
                NPar::ParallelFor(*LocalExecutor, 0, numBlocks, [&](ui32 blockId) {
                    TVector<double> sigma(rowSize * rowSize);
                    TVector<double> solution(rowSize);

                    for (size_t i = 0; i < sigma.size(); ++i) {
                        sigma[i] = CurrentPoint.Hessian[blockId * rowSize * rowSize + i];
                    }

                    for (size_t i = 0; i < solution.size(); ++i) {
                        solution[i] = CurrentPoint.Gradient[blockId * rowSize + i];
                    }

                    SolveLinearSystemCholesky(&sigma,
                                              &solution);

                    for (size_t i = 0; i < solution.size(); ++i) {
                        MoveDirection[blockId * rowSize + i] = (float)solution[i];
                    }
                });
            }

        private:
            TPointWithFuncInfo CurrentPoint;
            TVector<float> MoveDirection;

            NPar::ILocalExecutor* LocalExecutor;
        };
    }

    TVector<float> TNewtonLikeWalker::Estimate(
        TVector<float> startPoint,
        NPar::ILocalExecutor* localExecutor) {
        startPoint.resize(Oracle.PointDim());
        const int hessianBlockSize = Oracle.HessianBlockSize();

        TDirectionEstimator estimator(
            [&]() -> TPointWithFuncInfo {
                TPointWithFuncInfo point(hessianBlockSize);
                point.Point = startPoint;
                Oracle.MoveTo(point.Point);
                Oracle.WriteValueAndFirstDerivatives(&point.Value,
                                                     &point.Gradient);
                Oracle.AddLangevinNoiseToDerivatives(&point.Gradient, localExecutor);

                Oracle.WriteSecondDerivatives(&point.Hessian);
                Oracle.AddLangevinNoiseToDerivatives(&point.Hessian, localExecutor);

                return point;
            }(),
            localExecutor);

        if (Iterations == 1) {
            TVector<float> result;
            estimator.MoveInOptimalDirection(result, 1.0);
            Oracle.Regularize(&result);
            return Oracle.MakeEstimationResult(result);
        }

        {
            const auto& pointInfo = estimator.GetCurrentPoint();
            double gradNorm = pointInfo.GradientNorm();
            CATBOOST_DEBUG_LOG << "Initial gradient norm: " << gradNorm << " Func value: " << pointInfo.Value << Endl;
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
                Oracle.AddLangevinNoiseToDerivatives(&nextPointWithFuncInfo.Gradient, localExecutor);

                if (stepEstimation->IsSatisfied(step,
                                                nextPointWithFuncInfo.Value,
                                                nextPointWithFuncInfo.Gradient))
                {
                    Oracle.WriteSecondDerivatives(&nextPointWithFuncInfo.Hessian);
                    Oracle.AddLangevinNoiseToDerivatives(&nextPointWithFuncInfo.Gradient, localExecutor);
                    double gradNorm = nextPointWithFuncInfo.GradientNorm();

                    CATBOOST_DEBUG_LOG
                        << "Next point gradient norm: " << gradNorm << " Func value: " << nextPointWithFuncInfo.Value
                        << " Moved with step: " << step << Endl;
                    estimator.NextPoint(nextPointWithFuncInfo);
                    ++iteration;
                    updated = true;
                    break;
                }
            }
        }

        return Oracle.MakeEstimationResult(estimator.GetCurrentPoint().Point);
    }
}
