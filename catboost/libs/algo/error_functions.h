#pragma once

#include "params.h"
#include "metric.h"
#include "ders_holder.h"
#include "approx_util.h"

#include <catboost/libs/metrics/auc.h>

#include <library/containers/2d_array/2d_array.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/yassert.h>
#include <util/string/iterator.h>


void CalcSoftmax(const yvector<double>& approx, yvector<double>* softmax);

template<typename TChild, bool StoreExpApproxParam>
class IDerCalcer {
public:
    static const constexpr bool StoreExpApprox = StoreExpApproxParam;

    void CalcFirstDerRange(int start, int count, const double* approxes, const double* approxDeltas, const float* targets, const float* weights, double* ders) const {
        if (approxDeltas != nullptr) {
            for (int i = start; i < start + count; ++i) {
                ders[i] = CalcDer(UpdateApprox<StoreExpApprox>(approxes[i], approxDeltas[i]), targets[i]);
            }
        } else {
            for (int i = start; i < start + count; ++i) {
                ders[i] = CalcDer(approxes[i], targets[i]);
            }
        }
        if (weights != nullptr) {
            for (int i = start; i < start + count; ++i) {
                ders[i] *= static_cast<double>(weights[i]);
            }
        }
    }

    void CalcDersRange(int start, int count, const double* approxes, const double* approxDeltas, const float* targets, const float* weights, TDer1Der2* ders) const {
        if (approxDeltas != nullptr) {
            for (int i = start; i < start + count; ++i) {
                CalcDers(UpdateApprox<StoreExpApprox>(approxes[i], approxDeltas[i]), targets[i], &ders[i]);
            }
        } else {
            for (int i = start; i < start + count; ++i) {
                CalcDers(approxes[i], targets[i], &ders[i]);
            }
        }
        if (weights != nullptr) {
            for (int i = start; i < start + count; ++i) {
                ders[i].Der1 *= weights[i];
                ders[i].Der2 *= weights[i];
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
};

class TBinclassError : public IDerCalcer<TBinclassError, /*StoreExpApproxParam*/ true> {
public:
    explicit TBinclassError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approxExp, float target) const {
        const double p = approxExp / (1 + approxExp);
        return target > 0 ? (1 - p) : -p;
    }

    double CalcDer2(double approxExp, float = 0) const {
        const double p = approxExp / (1 + approxExp);
        return -p * (1 - p);
    }

    void CalcDers(double approxExp, float target, TDer1Der2* ders) const {
        const double p = approxExp / (1 + approxExp);
        ders->Der1 = target > 0 ? (1 - p) : -p;
        ders->Der2 = -p * (1 - p);
    }
    void CalcFirstDerRange(int start, int count, const double* approxes, const double* approxDeltas, const float* targets, const float* weights, double* ders) const;
    void CalcDersRange(int start, int count, const double* approxes, const double* approxDeltas, const float* targets, const float* weights, TDer1Der2* ders) const;
};

class TQuadError : public IDerCalcer<TQuadError, /*StoreExpApproxParam*/ false> {
public:
    static constexpr double RMSE_DER2 = -1.0;

    explicit TQuadError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approx, float target) const {
        return target - approx;
    }

    double CalcDer2(double = 0, float = 0) const {
        return RMSE_DER2;
    }
};

class TCrossEntropyError : public IDerCalcer<TCrossEntropyError, /*StoreExpApproxParam*/ true> {
public:
    explicit TCrossEntropyError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approxExp, float prob) const {
        // p * 1/(1+exp(x)) + (1-p) * (-exp(x)/(1+exp(x))) =
        // (p - (1-p)exp(x)) / (1+exp(x))
        return (prob - (1 - prob) * approxExp) / (1 + approxExp);
    }

    double CalcDer2(double approxExp, float = 0) const {
        double p = approxExp / (1 + approxExp);
        return -p * (1 - p);
    }

    void CalcDers(double approxExp, float prob, TDer1Der2* ders) const {
        const double p = approxExp / (1 + approxExp);
        ders->Der1 = (prob - (1 - prob) * approxExp) / (1 + approxExp);
        ders->Der2 = -p * (1 - p);
    }

    void CalcDersRange(int start, int count, const double* approxes, const double* approxDelta, const float* probs, const float* weights, TDer1Der2* ders) const;
};

class TQuantileError : public IDerCalcer<TQuantileError, /*StoreExpApproxParam*/ false> {
public:
    const double QUANTILE_DER2 = 0.0;

    double Alpha;
    SAVELOAD(Alpha);

    explicit TQuantileError(bool storeExpApprox)
        : Alpha(0.5)
    {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    TQuantileError(double alpha, bool storeExpApprox)
        : Alpha(alpha)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approx, float target) const {
        return (target - approx > 0) ? Alpha : -(1 - Alpha);
    }

    double CalcDer2(double = 0, float = 0) const {
        return QUANTILE_DER2;
    }
};

class TLogLinearQuantileError : public IDerCalcer<TLogLinearQuantileError, /*StoreExpApproxParam*/ true> {
public:
    const double QUANTILE_DER2 = 0.0;

    double Alpha;
    SAVELOAD(Alpha);

    explicit TLogLinearQuantileError(bool storeExpApprox)
        : Alpha(0.5)
    {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    TLogLinearQuantileError(double alpha, bool storeExpApprox)
        : Alpha(alpha)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approxExp, float target) const {
        return (target - approxExp > 0) ? Alpha * approxExp : -(1 - Alpha) * approxExp;
    }

    double CalcDer2(double = 0, float = 0) const {
        return QUANTILE_DER2;
    }
};

class TPoissonError : public IDerCalcer<TPoissonError, /*StoreExpApproxParam*/ true> {
public:
    explicit TPoissonError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approxExp, float target) const {
        return target - approxExp;
    }

    double CalcDer2(double approxExp, float) const {
        return -approxExp;
    }

    void CalcDers(double approxExp, float target, TDer1Der2* ders) const {
        ders->Der1 = target - approxExp;
        ders->Der2 = -approxExp;
    }
};

class TMAPError : public IDerCalcer<TMAPError, /*StoreExpApproxParam*/ false> {
public:
    const double MAPE_DER2 = 0.0;

    explicit TMAPError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approx, float target) const {
        return (target - approx > 0) ? 1 / target : -1 / target;
    }

    double CalcDer2(double = 0, float = 0) const {
        return MAPE_DER2;
    }
};

class TMultiClassError : public IDerCalcer<TMultiClassError, /*StoreExpApproxParam*/ false> {
public:
    explicit TMultiClassError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

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

class TMultiClassOneVsAllError : public IDerCalcer<TMultiClassError, /*StoreExpApproxParam*/ false> {
public:
    explicit TMultiClassOneVsAllError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for MultiClassOneVsAll error.");
    }

    double CalcDer2(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for MultiClassOneVsAll error.");
    }

    void CalcDersMulti(const yvector<double>& approx, float target, float weight, yvector<double>* der, TArray2D<double>* der2) const {
        int approxDimension = approx.ysize();

        yvector<double> prob(approxDimension);
        for (int dim = 0; dim < approxDimension; ++dim) {
            double expApprox = exp(approx[dim]);
            prob[dim] = expApprox / (1 + expApprox);
            (*der)[dim] = -prob[dim];
        }
        int targetClass = static_cast<int>(target);
        (*der)[targetClass] += 1;

        if (der2 != nullptr) {
            for (int dimY = 0; dimY < approxDimension; ++dimY) {
                for (int dimX = 0; dimX < approxDimension; ++dimX) {
                    (*der2)[dimY][dimX] = 0;
                }
                (*der2)[dimY][dimY] = -prob[dimY] * (1 - prob[dimY]);
            }
        }

        if (weight != 1) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                (*der)[dim] *= weight;
                (*der2)[dim][dim] *= weight;
            }
        }
    }
};

class TCustomError : public IDerCalcer<TCustomError, /*StoreExpApproxParam*/ false> {
public:
    TCustomError(const TFitParams& params)
        : Descriptor(*params.ObjectiveDescriptor)
    {
        CB_ENSURE(params.StoreExpApprox == StoreExpApprox, "Approx format does not match");
    }

    void CalcDersMulti(const yvector<double>& approx, float target, float weight,
                       yvector<double>* der, TArray2D<double>* der2) const
    {
        Descriptor.CalcDersMulti(approx, target, weight, der, der2, Descriptor.CustomData);
    }

    void CalcDersRange(int start, int count, const double* approxes, const double* approxDeltas, const float* targets,
                       const float* weights, TDer1Der2* ders) const
    {
        if (approxDeltas != nullptr) {
            yvector<double> updatedApproxes(count);
            for (int i = start; i < start + count; ++i) {
                updatedApproxes[i - start] = approxes[i] + approxDeltas[i];
            }
            Descriptor.CalcDersRange(count, updatedApproxes.data(), targets + start, weights + start, ders + start, Descriptor.CustomData);
        } else {
            Descriptor.CalcDersRange(count, approxes + start, targets + start, weights + start, ders + start, Descriptor.CustomData);
        }
    }

    void CalcFirstDerRange(int start, int count, const double* approxes, const double* approxDeltas, const float* targets,
                           const float* weights, double* ders) const
    {
        yvector<TDer1Der2> derivatives(count, {0.0, 0.0});
        CalcDersRange(start, count, approxes, approxDeltas, targets, weights, derivatives.data() - start);
        for (int i = start; i < start + count; ++i) {
            ders[i] = derivatives[i - start].Der1;
        }
    }
private:
    TCustomObjectiveDescriptor Descriptor;
};
