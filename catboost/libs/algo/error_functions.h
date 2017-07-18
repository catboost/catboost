#pragma once

#include "params.h"
#include "metric.h"
#include "ders_holder.h"

#include <catboost/libs/metrics/auc.h>

#include <library/containers/2d_array/2d_array.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/yassert.h>
#include <util/string/iterator.h>


void CalcSoftmax(const yvector<double>& approx, yvector<double>* softmax);

template<typename TChild>
class IDerCalcer {
public:
    void CalcFirstDerRange(int count, const double* approxes, const float* targets, const float* weights, double* ders) const {
        if (weights) {
            for (int i = 0; i < count; ++i) {
                ders[i] = CalcDer(approxes[i], targets[i]) * static_cast<double>(weights[i]);
            }
        } else {
            for (int i = 0; i < count; ++i) {
                ders[i] = CalcDer(approxes[i], targets[i]);
            }
        }
    }

    void CalcDersRange(int count, const double* approxes, const float* targets, const float* weights, TDer1Der2* ders) const {
        if (weights) {
            for (int i = 0; i < count; ++i) {
                CalcWeightedDers(approxes[i], targets[i], weights[i], &ders[i]);
            }
        } else {
            for (int i = 0; i < count; ++i) {
                CalcDers(approxes[i], targets[i], &ders[i]);
            }
        }
    }

    void CalcDersMulti(const yvector<double>&, float, float, yvector<double>*, TArray2D<double>*) const {
        CB_ENSURE(false, "Not implemented");
    }

private:
    double CalcDer(double approx, float target) const {
        return static_cast<const TChild*>(this)->CalcDer(approx, target);
    }

    double CalcDer2(double approx, float target) const {
        return static_cast<const TChild*>(this)->CalcDer2(approx, target);
    }

    void CalcDers(double approx, float target, TDer1Der2* ders) const {
        ders->Der1 = CalcDer(approx, target);
        ders->Der2 = CalcDer2(approx, target);
    }

    void CalcWeightedDers(double approx, float target, float w, TDer1Der2* ders) const {
        ders->Der1 = CalcDer(approx, target) * static_cast<double>(w);
        ders->Der2 = CalcDer2(approx, target) * static_cast<double>(w);
    }
};

class TBinclassError : public IDerCalcer<TBinclassError>  {
public:
    double CalcDer(double approx, float target) const {
        const double approxExp = exp(approx);
        const double p = approxExp / (1 + approxExp);
        return target > 0 ? (1 - p) : -p;
    }

    double CalcDer2(double approx, float = 0) const {
        const double approxExp = exp(approx);
        const double p = approxExp / (1 + approxExp);
        return -p * (1 - p);
    }

    void CalcDers(double approx, float target, TDer1Der2* ders) const {
        const double approxExp = exp(approx);
        const double p = approxExp / (1 + approxExp);
        ders->Der1 = target > 0 ? (1 - p) : -p;
        ders->Der2 = -p * (1 - p);
    }
    void CalcFirstDerRange(int count, const double* approxes, const float* targets, const float* weights, double* ders) const;
    void CalcDersRange(int count, const double* approxes, const float* targets, const float* weights, TDer1Der2* ders) const;
};

class TQuadError : public IDerCalcer<TQuadError> {
public:
    static constexpr double RMSE_DER2 = -1.0;

    double CalcDer(double approx, float target) const {
        return target - approx;
    }

    double CalcDer2(double = 0, float = 0) const {
        return RMSE_DER2;
    }
};

class TCrossEntropyError : public IDerCalcer<TCrossEntropyError> {
public:
    double CalcDer(double approx, float prob) const {
        // p * 1/(1+exp(x)) + (1-p) * (-exp(x)/(1+exp(x))) =
        // (p - (1-p)exp(x)) / (1+exp(x))
        double approxExp = exp(approx);
        return (prob - (1 - prob) * approxExp) / (1 + approxExp);
    }

    double CalcDer2(double approx, float = 0) const {
        double approxExp = exp(approx);
        double p = approxExp / (1 + approxExp);
        return -p * (1 - p);
    }

    void CalcDers(double approx, float prob, TDer1Der2* ders) const {
        const double approxExp = exp(approx);
        const double p = approxExp / (1 + approxExp);
        ders->Der1 = (prob - (1 - prob) * approxExp) / (1 + approxExp);
        ders->Der2 = -p * (1 - p);
    }

    void CalcDersRange(int count, const double* approxes, const float* probs, const float* weights, TDer1Der2* ders) const;
};

class TQuantileError : public IDerCalcer<TQuantileError> {
public:
    const double QUANTILE_DER2 = 0.0;

    double Alpha;
    SAVELOAD(Alpha);

    TQuantileError()
        : Alpha(0.5)
    {
    }

    explicit TQuantileError(double alpha)
        : Alpha(alpha)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
    }

    double CalcDer(double approx, float target) const {
        return (target - approx > 0) ? Alpha : -(1 - Alpha);
    }

    double CalcDer2(double = 0, float = 0) const {
        return QUANTILE_DER2;
    }
};

class TLogLinearQuantileError : public IDerCalcer<TLogLinearQuantileError> {
public:
    const double QUANTILE_DER2 = 0.0;

    double Alpha;
    SAVELOAD(Alpha);

    TLogLinearQuantileError()
        : Alpha(0.5)
    {
    }

    explicit TLogLinearQuantileError(double alpha)
        : Alpha(alpha)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
    }

    double CalcDer(double approx, float target) const {
        double expApprox = exp(approx);
        return (target - expApprox > 0) ? Alpha * expApprox : -(1 - Alpha) * expApprox;
    }

    double CalcDer2(double = 0, float = 0) const {
        return QUANTILE_DER2;
    }
};

class TPoissonError : public IDerCalcer<TPoissonError> {
public:
    double CalcDer(double approx, float target) const {
        return target - exp(approx);
    }

    double CalcDer2(double approx, float) const {
        return -exp(approx);
    }

    void CalcDers(double approx, float target, TDer1Der2* ders) const {
        const double expApprox = exp(approx);
        ders->Der1 = target - expApprox;
        ders->Der2 = -expApprox;
    }
};

class TMAPError : public IDerCalcer<TMAPError> {
public:
    const double MAPE_DER2 = 0.0;

    double CalcDer(double approx, float target) const {
        return (target - approx > 0) ? 1 / target : -1 / target;
    }

    double CalcDer2(double = 0, float = 0) const {
        return MAPE_DER2;
    }
};

class TMultiClassError : public IDerCalcer<TMultiClassError> {
public:
    double CalcDer(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for MultiClass error.");
    }

    double CalcDer2(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for MultiClass error.");
    }

    void CalcDersMulti(const yvector<double>& approx, float target, float weight, yvector<double>* der, TArray2D<double>* der2) const {
        int approxDimension = approx.ysize();

        yvector<double> softmax(approxDimension);
        CalcSoftmax(approx, &softmax);

        for (int dim = 0; dim < approxDimension; ++dim) {
            (*der)[dim] = -softmax[dim];
        }
        int targetClass = static_cast<int>(target);
        (*der)[targetClass] += 1;

        if (der2 != nullptr) {
            for (int dimY = 0; dimY < approxDimension; ++dimY) {
                for (int dimX = 0; dimX < approxDimension; ++dimX) {
                    (*der2)[dimY][dimX] = softmax[dimY] * softmax[dimX];
                }
                (*der2)[dimY][dimY] -= softmax[dimY];
            }
        }
        if (weight != 1) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                (*der)[dim] *= weight;
            }
            if (der2 != nullptr) {
                for (int dimY = 0; dimY < approxDimension; ++dimY) {
                    for (int dimX = 0; dimX < approxDimension; ++dimX) {
                        (*der2)[dimY][dimX] *= weight;
                    }
                }
            }
        }
    }
};

class TCustomError : public IDerCalcer<TCustomError> {
public:
    explicit TCustomError(const TCustomObjectiveDescriptor& descriptor)
        : Descriptor(descriptor)
    { }

    void CalcDersMulti(const yvector<double>& approx, float target, float weight,
                       yvector<double>* der, TArray2D<double>* der2) const
    {
        Descriptor.CalcDersMulti(approx, target, weight, der, der2, Descriptor.CustomData);
    }

    void CalcDersRange(int count, const double* approxes, const float* targets,
                       const float* weights, TDer1Der2* ders) const
    {
        Descriptor.CalcDersRange(count, approxes, targets, weights, ders, Descriptor.CustomData);
    }

    void CalcFirstDerRange(int count, const double* approxes, const float* targets,
                           const float* weights, double* ders) const
    {
        yvector<TDer1Der2> derivatives(count, {0.0, 0.0});
        CalcDersRange(count, approxes, targets, weights, derivatives.data());
        for (int i = 0; i < count; ++i) {
            ders[i] = derivatives[i].Der1;
        }
    }
private:
    TCustomObjectiveDescriptor Descriptor;
};
