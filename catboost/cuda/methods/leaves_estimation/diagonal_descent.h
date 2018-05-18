#pragma once

#include <util/generic/ymath.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>

namespace NCatboostCuda {

      struct TDiagonalDescentPoint {
        double Value = 0.0;

        TVector<float> Point;
        TVector<float> Gradient;
        TVector<float> Hessian;

        TDiagonalDescentPoint(ui32 partCount) {
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

            for (size_t leaf = 0; leaf < gradient.size(); ++leaf) {
                const double grad = gradient[leaf];
                gradNorm += grad * grad;
            }
            return sqrt(gradNorm);
        }
    };

    class TDirectionEstimator {
    public:

        TDirectionEstimator(TDiagonalDescentPoint&& point)
            : CurrentPoint(std::move(point)) {
            UpdateMoveDirection();
        }

        void NextPoint(const TDiagonalDescentPoint& pointInfo) {
            CurrentPoint.Value = pointInfo.Value;

            Copy(pointInfo.Point.begin(), pointInfo.Point.end(), CurrentPoint.Point.begin());
            Copy(pointInfo.Gradient.begin(), pointInfo.Gradient.end(), CurrentPoint.Gradient.begin());

            Copy(pointInfo.Hessian.begin(), pointInfo.Hessian.end(), CurrentPoint.Hessian.begin());
            UpdateMoveDirection();
        }

        const TVector<float>& GetDirection() {
            return MoveDirection;
        }

        const TDiagonalDescentPoint& GetCurrentPoint() const {
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
        TDiagonalDescentPoint CurrentPoint;
        TVector<float> MoveDirection;
    };
}
