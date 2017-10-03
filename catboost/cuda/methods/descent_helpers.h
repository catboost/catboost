#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>

struct TPointwiseDescentPoint {
    double Value = 0.0;

    yvector<float> Point;
    yvector<float> Gradient;
    yvector<float> Hessian;

    TPointwiseDescentPoint(ui32 partCount) {
        SetSize(partCount);
    }

    void SetSize(ui32 leafCount) {
        Point.resize(leafCount);
        Gradient.resize(leafCount);
        Hessian.resize(leafCount);
    }

    const yvector<float>& GetCurrentPoint() const {
        return Point;
    }

    void CleanDerivativeInformation() {
        Fill(Gradient.begin(), Gradient.end(), 0.0f);
        Fill(Hessian.begin(), Hessian.end(), 0.0f);
    }

    void AddToHessianDiag(ui32 x, float val) {
        Hessian[x] += val;
    }
};

class TSimpleStepEstimator {
private:
    double FunctionValue;

public:
    TSimpleStepEstimator(const double functionValue,
                         const yvector<float>& gradient,
                         const yvector<float>& direction)
        : FunctionValue(functionValue)
    {
        (void)gradient;
        (void)direction;
    }

    bool IsSatisfied(double,
                     double nextFuncValue,
                     const yvector<float>&) const {
        return FunctionValue <= nextFuncValue;
    }
};

class TArmijoStepEstimation {
private:
    const double C = 1e-5;

    double FunctionValue;
    const yvector<float>& Gradient;
    const yvector<float>& Direction;
    double DirGradDot;

public:
    TArmijoStepEstimation(const double functionValue,
                          const yvector<float>& gradient,
                          const yvector<float>& direction)
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
                     const yvector<float>& nextFuncGradient) const {
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

    const yvector<float>& GetDirection() {
        return MoveDirection;
    }

    const TPointwiseDescentPoint& GetCurrentPoint() const {
        return CurrentPoint;
    }

    void MoveInOptimalDirection(yvector<float>& point,
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
    yvector<float> MoveDirection;
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

    inline yvector<float> Estimate(const yvector<float>& startPoint) {
        const ui32 dim = (const ui32)startPoint.size();

        TDirectionEstimator estimator([&]() -> TDescentPoint {
            auto currentPointInfo = TOracle::Create(dim);
            currentPointInfo.Point = startPoint;
            DerCalcer.MoveTo(currentPointInfo.GetCurrentPoint());
            DerCalcer.ComputeValueAndDerivatives(currentPointInfo);
            return currentPointInfo;
        }());

        if (Iterations == 1) {
            yvector<float> result;
            estimator.MoveInOptimalDirection(result, 1.0);
            return result;
        }

        {
            const auto& pointInfo = estimator.GetCurrentPoint();
            double gradNorm = DerCalcer.GradientNorm(pointInfo);
            MATRIXNET_INFO_LOG << "Initial gradient norm: " << gradNorm << " Func value: " << pointInfo.Value << Endl;
        }

        TDescentPoint nextPointInfo = TOracle::Create(dim);

        for (ui32 iteration = 1; iteration < Iterations;) {
            const auto& currentPointInfo = estimator.GetCurrentPoint();
            const auto& moveDirection = estimator.GetDirection();

            double step = 1.0;
            TBacktrackingStepEstimator stepEstimation(currentPointInfo.Value,
                                                      currentPointInfo.Gradient,
                                                      moveDirection);

            for (; iteration < Iterations; ++iteration, step /= 2) {
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
                    double gradNorm = DerCalcer.GradientNorm(nextPointInfo);

                    MATRIXNET_INFO_LOG << "Next point gradient norm: " << gradNorm << " Func value: " << nextPointInfo.Value
                                       << " Moved with step: " << step << Endl;
                    estimator.NextPoint(nextPointInfo);
                    ++iteration;
                    break;
                }
            }
        }

        return estimator.GetCurrentPoint().Point;
    }
};
