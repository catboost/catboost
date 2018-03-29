#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <util/generic/algorithm.h>

namespace NCatboostCuda {
    struct TPointwiseDescentPoint {
        double Value = 0.0;

        TVector<float> Point;
        TVector<float> Gradient;
        TVector<float> Hessian;

        TPointwiseDescentPoint(ui32 partCount) {
            SetSize(partCount);
        }

        void SetSize(ui32 leafCount) {
            Point.resize(leafCount);
            Gradient.resize(leafCount);
            Hessian.resize(leafCount);
        }

        const TVector<float>& GetCurrentPoint() const {
            return Point;
        }

        void CleanDerivativeInformation() {
            Fill(Gradient.begin(), Gradient.end(), 0.0f);
            Fill(Hessian.begin(), Hessian.end(), 0.0f);
        }

        void AddToHessianDiag(ui32 x, float val) {
            Hessian[x] += val;
        }

        double GradientNorm() const {
            const auto& gradient = Gradient;
            double gradNorm = 0;

            for (ui32 leaf = 0; leaf < gradient.size(); ++leaf) {
                const double grad = gradient[leaf];
                gradNorm += grad * grad;
            }
            return sqrt(gradNorm);
        }
    };

    inline void AddRidgeRegularization(double lambda,
                                       TPointwiseDescentPoint& pointInfo,
                                       bool tohessianOnly = true) {
        if (!tohessianOnly) {
            double hingeLoss = 0;
            {
                for (const auto& val : pointInfo.Point) {
                    hingeLoss += val * val;
                }
                hingeLoss *= lambda / 2;
            }
            pointInfo.Value -= hingeLoss;
        }

        for (ui32 i = 0; i < pointInfo.Gradient.size(); ++i) {
            pointInfo.AddToHessianDiag(i, static_cast<float>(lambda));
            if (!tohessianOnly) {
                pointInfo.Gradient[i] -= lambda * pointInfo.Point[i];
            }
        }
    }

    class TSimpleStepEstimator {
    private:
        double FunctionValue;

    public:
        TSimpleStepEstimator(const double functionValue,
                             const TVector<float>& gradient,
                             const TVector<float>& direction)
            : FunctionValue(functionValue)
        {
            (void)gradient;
            (void)direction;
        }

        bool IsSatisfied(double,
                         double nextFuncValue,
                         const TVector<float>&) const {
            return FunctionValue <= nextFuncValue;
        }
    };

    class TArmijoStepEstimation {
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
                         const TVector<float>& nextFuncGradient) const {
            double directionNextGradDot = 0;
            for (ui32 i = 0; i < Gradient.size(); ++i) {
                directionNextGradDot += Gradient[i] * nextFuncGradient[i];
            }
            return (nextFuncValue >= (FunctionValue + C * step * DirGradDot));
        }
    };

    class TDirectionEstimator {
    public:
        TDirectionEstimator(TPointwiseDescentPoint&& point)
            : CurrentPoint(std::move(point))
        {
            UpdateMoveDirection();
        }

        void NextPoint(const TPointwiseDescentPoint& pointInfo) {
            CurrentPoint.Value = pointInfo.Value;

            Copy(pointInfo.Point.begin(), pointInfo.Point.end(), CurrentPoint.Point.begin());
            Copy(pointInfo.Gradient.begin(), pointInfo.Gradient.end(), CurrentPoint.Gradient.begin());

            Copy(pointInfo.Hessian.begin(), pointInfo.Hessian.end(), CurrentPoint.Hessian.begin());
            UpdateMoveDirection();
        }

        const TVector<float>& GetDirection() {
            return MoveDirection;
        }

        const TPointwiseDescentPoint& GetCurrentPoint() const {
            return CurrentPoint;
        }

        void MoveInOptimalDirection(TVector<float>& point,
                                    double step) const {
            point.resize(CurrentPoint.Point.size());

            Copy(CurrentPoint.Point.begin(), CurrentPoint.Point.end(), point.begin());

            for (ui32 leaf = 0; leaf < point.size(); ++leaf) {
                point[leaf] += step * MoveDirection[leaf];
            }
        }

    private:
        void UpdateMoveDirection() {
            MoveDirection.resize(CurrentPoint.Point.size());

            for (ui32 i = 0; i < CurrentPoint.Gradient.size(); ++i) {
                MoveDirection[i] = CurrentPoint.Hessian[i] > 0 ? CurrentPoint.Gradient[i] / (CurrentPoint.Hessian[i] + 1e-20f) : 0;
            }
        }

    private:
        TPointwiseDescentPoint CurrentPoint;
        TVector<float> MoveDirection;
    };

    template <class TOracle,
              class TBacktrackingStepEstimator>
    class TNewtonLikeWalker {
    public:
        using TDescentPoint = typename TOracle::TDescentPoint;
        TOracle& DerCalcer;
        const ui32 Iterations;

    public:
        TNewtonLikeWalker(TOracle& leavesProjector,
                          const ui32 iterations)
            : DerCalcer(leavesProjector)
            , Iterations(iterations)
        {
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
                MATRIXNET_INFO_LOG
                    << "Initial gradient norm: " << gradNorm << " Func value: " << pointInfo.Value << Endl;
            }

            TDescentPoint nextPointInfo = TOracle::Create(dim);

            bool updated = false;
            for (ui32 iteration = 1; iteration < Iterations;) {
                const auto& currentPointInfo = estimator.GetCurrentPoint();
                const auto& moveDirection = estimator.GetDirection();

                double step = 1.0;
                TBacktrackingStepEstimator stepEstimation(currentPointInfo.Value,
                                                          currentPointInfo.Gradient,
                                                          moveDirection);

                for (; iteration < Iterations || (!updated && iteration < 100); ++iteration, step /= 2) {
                    auto& nextPoint = nextPointInfo.Point;
                    estimator.MoveInOptimalDirection(nextPoint, step);

                    DerCalcer.Regularize(nextPoint);
                    DerCalcer.MoveTo(nextPoint);
                    //compute value and diagonal ders. it's faster to do all at once
                    DerCalcer.ComputeValueAndDerivatives(nextPointInfo);

                    if (stepEstimation.IsSatisfied(step,
                                                   nextPointInfo.Value,
                                                   nextPointInfo.Gradient))
                    {
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
