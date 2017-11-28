#pragma once

#include "approx_util.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/metrics/ders_holder.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/auc.h>
#include <catboost/libs/data/pair.h>
#include <catboost/libs/helpers/eval_helpers.h>

#include <library/containers/2d_array/2d_array.h>
#include <library/threading/local_executor/local_executor.h>
#include <library/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/yassert.h>
#include <util/string/iterator.h>

template<typename TChild, bool StoreExpApproxParam>
class IDerCalcer {
public:
    static const constexpr bool StoreExpApprox = StoreExpApproxParam;

    void CalcFirstDerRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        double* ders
    ) const {
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

    void CalcDersRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDer1Der2* ders
    ) const {
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

    void CalcDersMulti(
        const TVector<double>& /*approx*/,
        float /*target*/,
        float /*weight*/,
        TVector<double>* /*der*/,
        TArray2D<double>* /*der2*/
    ) const {
        CB_ENSURE(false, "Not implemented");
    }

    void CalcDersPairs(
        const TVector<double>& /*approxes*/,
        const TVector<TVector<TCompetitor>>& /*competitors*/,
        int /*start*/,
        int /*count*/,
        TVector<double>* /*ders*/
    ) const {
        CB_ENSURE(false, "Not implemented");
    }

    void CalcDersForQueries(
        int /*start*/,
        int /*count*/,
        const TVector<double>& /*approx*/,
        const TVector<float>& /*target*/,
        const TVector<float>& /*weight*/,
        const TVector<ui32>& /*queriesId*/,
        const THashMap<ui32, ui32>& /*queriesSize*/,
        TVector<TDer1Der2>* /*ders*/
    ) const {
        CB_ENSURE(false, "Not implemented");
    }

    EErrorType GetErrorType() const {
        return EErrorType::PerObjectError;
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

class TLoglossError : public IDerCalcer<TLoglossError, /*StoreExpApproxParam*/ true> {
public:
    explicit TLoglossError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approxExp, float target) const {
        const double p = approxExp / (1 + approxExp);
        return target - p;
    }

    double CalcDer2(double approxExp, float = 0) const {
        const double p = approxExp / (1 + approxExp);
        return -p * (1 - p);
    }

    void CalcDers(double approxExp, float target, TDer1Der2* ders) const {
        const double p = approxExp / (1 + approxExp);
        ders->Der1 = target - p;
        ders->Der2 = -p * (1 - p);
    }

    void CalcFirstDerRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        double* ders
    ) const;

    void CalcDersRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDer1Der2* ders
    ) const;
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

    void CalcDersRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDelta,
        const float* probs,
        const float* weights,
        TDer1Der2* ders
    ) const;
};

class TRMSEError : public IDerCalcer<TRMSEError, /*StoreExpApproxParam*/ false> {
public:
    static constexpr double RMSE_DER2 = -1.0;

    explicit TRMSEError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double approx, float target) const {
        return target - approx;
    }

    double CalcDer2(double = 0, float = 0) const {
        return RMSE_DER2;
    }
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

class TLogLinQuantileError : public IDerCalcer<TLogLinQuantileError, /*StoreExpApproxParam*/ true> {
public:
    const double QUANTILE_DER2 = 0.0;

    double Alpha;
    SAVELOAD(Alpha);

    explicit TLogLinQuantileError(bool storeExpApprox)
        : Alpha(0.5)
    {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    TLogLinQuantileError(double alpha, bool storeExpApprox)
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

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        TArray2D<double>* der2
    ) const {
        int approxDimension = approx.ysize();

        TVector<double> softmax(approxDimension);
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

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        TArray2D<double>* der2
    ) const {
        int approxDimension = approx.ysize();

        TVector<double> prob(approxDimension);
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
            }
            if (der2 != nullptr) {
                for (int dim = 0; dim < approxDimension; ++dim) {
                    (*der2)[dim][dim] *= weight;
                }
            }
        }
    }
};

class TPairLogitError : public IDerCalcer<TPairLogitError, /*StoreExpApproxParam*/ true> {
public:
    explicit TPairLogitError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for PairLogit error.");
    }

    double CalcDer2(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for PairLogit error.");
    }

    EErrorType GetErrorType() const {
        return EErrorType::PairwiseError;
    }

    void CalcDersPairs(
        const TVector<double>& expApproxes,
        const TVector<TVector<TCompetitor>>& competitors,
        int start,
        int count,
        TVector<double>* ders
    ) const {
        for (int docId = start; docId < start + count; ++docId) {
            double curDer = 0;
            for (const auto& competitor : competitors[docId]) {
                if (competitor.Weight > 0) {
                    curDer += competitor.Weight * expApproxes[competitor.Id] / (expApproxes[competitor.Id] + expApproxes[docId]);
                } else {
                    curDer += competitor.Weight * expApproxes[docId] / (expApproxes[competitor.Id] + expApproxes[docId]);
                }
            }
            (*ders)[docId - start] = curDer;
        }
    }
};

class TQueryRmseError : public IDerCalcer<TQueryRmseError, /*StoreExpApproxParam*/ false> {
public:
    explicit TQueryRmseError(bool storeExpApprox) {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
    }

    double CalcDer(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for QueryRMSE error.");
    }

    double CalcDer2(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for QueryRMSE error.");
    }

    EErrorType GetErrorType() const {
        return EErrorType::QuerywiseError;
    }

    void CalcDersForQueries(
        int start,
        int count,
        const TVector<double>& approxes,
        const TVector<float>& targets,
        const TVector<float>& weights,
        const TVector<ui32>& queriesId,
        const THashMap<ui32, ui32>& queriesSize,
        TVector<TDer1Der2>* ders
    ) const {
        int offset = 0;
        while (offset < count) {
            ui32 querySize = queriesSize.find(queriesId[start + offset])->second;
            double queryAvrg = CalcQueryAvrg(start + offset, querySize, approxes, targets, weights);
            for (ui32 docId = start + offset; docId < start + offset + querySize; ++docId) {
                (*ders)[docId - start].Der1 = targets[docId] - approxes[docId] - queryAvrg;
                if (!weights.empty()) {
                    (*ders)[docId - start].Der1 *= weights[docId];
                }
            }
            offset += querySize;
        }
    }
private:
    double CalcQueryAvrg(
        int start,
        int count,
        const TVector<double>& approxes,
        const TVector<float>& targets,
        const TVector<float>& weights
    ) const {
        double querySum = 0;
        double queryCount = 0;
        for (int docId = start; docId < start + count; ++docId) {
            double w = weights.empty() ? 1 : weights[docId];
            querySum += (targets[docId] - approxes[docId]) * w;
            queryCount += w;
        }

        double queryAvrg = 0;
        if (queryCount > 0) {
            queryAvrg = querySum / queryCount;
        }
        return queryAvrg;
    }
};

class TCustomError : public IDerCalcer<TCustomError, /*StoreExpApproxParam*/ false> {
public:
    TCustomError(
        const NCatboostOptions::TCatBoostOptions& params,
        const TMaybe<TCustomObjectiveDescriptor>& descriptor
    )
        : Descriptor(*descriptor)
    {
        CB_ENSURE(IsStoreExpApprox(params) == StoreExpApprox, "Approx format does not match");
    }

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        TArray2D<double>* der2
    ) const {
        Descriptor.CalcDersMulti(approx, target, weight, der, der2, Descriptor.CustomData);
    }

    void CalcDersRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDer1Der2* ders
    ) const {
        if (approxDeltas != nullptr) {
            TVector<double> updatedApproxes(count);
            for (int i = start; i < start + count; ++i) {
                updatedApproxes[i - start] = approxes[i] + approxDeltas[i];
            }
            Descriptor.CalcDersRange(count, updatedApproxes.data(), targets + start, weights + start, ders + start, Descriptor.CustomData);
        } else {
            Descriptor.CalcDersRange(count, approxes + start, targets + start, weights + start, ders + start, Descriptor.CustomData);
        }
    }

    void CalcFirstDerRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        double* ders
    ) const {
        TVector<TDer1Der2> derivatives(count, {0.0, 0.0});
        CalcDersRange(start, count, approxes, approxDeltas, targets, weights, derivatives.data() - start);
        for (int i = start; i < start + count; ++i) {
            ders[i] = derivatives[i - start].Der1;
        }
    }
private:
    TCustomObjectiveDescriptor Descriptor;
};

class TUserDefinedPerObjectError : public IDerCalcer<TUserDefinedPerObjectError, /*StoreExpApproxParam*/ false> {
public:

    double Alpha;
    SAVELOAD(Alpha);

    TUserDefinedPerObjectError(const ymap<TString, TString>& params, bool storeExpApprox)
        : Alpha(0.0)
    {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
        if (params.has("alpha")) {
            Alpha = FromString<double>(params.at("alpha"));
        }
    }

    double CalcDer(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for TUserDefinedPerObjectError error.");
        return 0.0;
    }

    double CalcDer2(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for TUserDefinedPerObjectError error.");
        return 0.0;
    }
};

class TUserDefinedQuerywiseError : public IDerCalcer<TUserDefinedQuerywiseError, /*StoreExpApproxParam*/ false> {
public:

    double Alpha;
    SAVELOAD(Alpha);

    TUserDefinedQuerywiseError(const ymap<TString, TString>& params, bool storeExpApprox)
        : Alpha(0.0)
    {
        CB_ENSURE(storeExpApprox == StoreExpApprox, "Approx format does not match");
        if (params.has("alpha")) {
            Alpha = FromString<double>(params.at("alpha"));
        }
    }

    double CalcDer(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for TUserDefinedQuerywiseError error.");
    }

    double CalcDer2(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented for TUserDefinedQuerywiseError error.");
    }

    void CalcDersForQueries(
        int /*start*/,
        int /*count*/,
        const TVector<double>& /*approxes*/,
        const TVector<float>& /*targets*/,
        const TVector<float>& /*weights*/,
        const TVector<ui32>& /*queriesId*/,
        const THashMap<ui32, ui32>& /*queriesSize*/,
        TVector<TDer1Der2>* /*ders*/
    ) const {
        CB_ENSURE(false, "Not implemented for TUserDefinedQuerywiseError error.");
    }

    EErrorType GetErrorType() const {
        return EErrorType::QuerywiseError;
    }
};

