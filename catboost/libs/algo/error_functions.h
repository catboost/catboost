#pragma once

#include "approx_updater_helpers.h"
#include "custom_objective_descriptor.h"
#include "ders_holder.h"
#include "hessian.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/eval_result/eval_helpers.h>

#include <library/containers/2d_array/2d_array.h>
#include <library/fast_exp/fast_exp.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/yassert.h>
#include <util/string/iterator.h>

class IDerCalcer {
public:
    explicit IDerCalcer(
        bool isExpApprox,
        ui32 maxDerivativeOrder = 3,
        EErrorType errorType = EErrorType::PerObjectError,
        EHessianType hessianType = EHessianType::Symmetric)
    : IsExpApprox(isExpApprox)
    , MaxSupportedDerivativeOrder(maxDerivativeOrder)
    , ErrorType(errorType)
    , HessianType(hessianType)
    {
        Y_ASSERT(maxDerivativeOrder >= 1 && maxDerivativeOrder <= 3);
    }

    virtual ~IDerCalcer() = default;

    bool GetIsExpApprox() const {
        return IsExpApprox;
    }

    ui32 GetMaxSupportedDerivativeOrder() const {
        return MaxSupportedDerivativeOrder;
    }

    EErrorType GetErrorType() const {
        return ErrorType;
    }

    EHessianType GetHessianType() const {
        return HessianType;
    }

    virtual void CalcFirstDerRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        double* firstDers
    ) const {
        CalcDersRange(start, count, /*maxDerivativeOrder*/ 1, approxes, approxDeltas, targets, weights, /*ders*/ nullptr, firstDers);
    }

    virtual void CalcDersRange(
        int start,
        int count,
        bool calcThirdDer,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDers* ders
    ) const {
        const int maxDerivativeOrder = calcThirdDer ? 3 : Min(MaxSupportedDerivativeOrder, 2u);
        CalcDersRange(start, count, maxDerivativeOrder, approxes, approxDeltas, targets, weights, ders, /*firstDers*/ nullptr);
    }

    virtual void CalcDersMulti(
        const TVector<double>& /*approx*/,
        float /*target*/,
        float /*weight*/,
        TVector<double>* /*der*/,
        THessianInfo* /*der2*/
    ) const {
        CB_ENSURE(false, "Not implemented");
    }

    virtual void CalcDersForQueries(
        int /*queryStartIndex*/,
        int /*queryEndIndex*/,
        const TVector<double>& /*approx*/,
        const TVector<float>& /*target*/,
        const TVector<float>& /*weight*/,
        const TVector<TQueryInfo>& /*queriesInfo*/,
        TArrayRef<TDers> /*ders*/,
        NPar::TLocalExecutor* /*localExecutor*/
    ) const {
        CB_ENSURE(false, "Not implemented");
    }

private:
    const bool IsExpApprox;
    const ui32 MaxSupportedDerivativeOrder;
    const EErrorType ErrorType;
    const EHessianType HessianType;

    virtual double CalcDer(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented");
    }

    virtual double CalcDer2(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented");
    }

    virtual double CalcDer3(double /*approx*/, float /*target*/) const {
        CB_ENSURE(false, "Not implemented");
    }

    template <int MaxDerivativeOrder, bool UseTDers, bool UseExpApprox, bool HasDelta>
    void CalcDersRangeImpl(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDers* ders,
        double* firstDers
    ) const;

    void CalcDersRange(
        int start,
        int count,
        int maxDerivativeOrder,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDers* ders,
        double* firstDers
    ) const;
};

class TCrossEntropyError final : public IDerCalcer {
public:
    explicit TCrossEntropyError(bool isExpApprox)
    : IDerCalcer(isExpApprox)
    {
    }

    void CalcFirstDerRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        double* ders
    ) const override;

    void CalcDersRange(
        int start,
        int count,
        bool calcThirdDer,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDers* ders
    ) const override;
};

class TRMSEError final : public IDerCalcer {
public:
    static constexpr double RMSE_DER2 = -1.0;
    static constexpr double RMSE_DER3 = 0.0;

    explicit TRMSEError(bool isExpApprox)
    : IDerCalcer(isExpApprox)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        return target - approx;
    }

    double CalcDer2(double /*approx*/, float /*target*/) const override {
        return RMSE_DER2;
    }

    double CalcDer3(double /*approx*/, float /*target*/) const override {
        return RMSE_DER3;
    }
};

class TQuantileError final : public IDerCalcer {
public:
    static constexpr double QUANTILE_DER2_AND_DER3 = 0.0;

    const double Alpha;

    explicit TQuantileError(bool isExpApprox)
    : IDerCalcer(isExpApprox)
    , Alpha(0.5)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    TQuantileError(double alpha, bool isExpApprox)
    : IDerCalcer(isExpApprox)
    , Alpha(alpha)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        return (target - approx > 0) ? Alpha : -(1 - Alpha);
    }

    double CalcDer2(double /*approx*/, float /*target*/) const override {
        return QUANTILE_DER2_AND_DER3;
    }

    double CalcDer3(double /*approx*/, float /*target*/) const override {
        return QUANTILE_DER2_AND_DER3;
    }
};

class TLqError final : public IDerCalcer {
public:
    const double Q;

    TLqError(double q, bool isExpApprox)
    : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ q >= 2 ?  3 : 1)
    , Q(q)
    {
        Y_ASSERT(Q >= 1);
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        const double absLoss = abs(approx - target);
        const double absLossQ = std::pow(absLoss, Q - 1);
        return Q * (approx - target > 0 ? 1 : -1)  * absLossQ;
    }

    double CalcDer2(double approx, float target) const override {
        const double absLoss = abs(target - approx);
        return Q * (Q - 1) * std::pow(absLoss, Q - 2);
    }

    double CalcDer3(double approx, float target) const override {
        const double absLoss = abs(target - approx);
        return Q * (Q - 1) *  (Q - 2) * std::pow(absLoss, Q - 3) * (approx - target > 0 ? 1 : -1);
    }
};

class TLogLinQuantileError final : public IDerCalcer {
public:
    static constexpr double QUANTILE_DER2_AND_DER3 = 0.0;

    const double Alpha;

    explicit TLogLinQuantileError(bool isExpApprox)
    : IDerCalcer(isExpApprox)
    , Alpha(0.5)
    {
        CB_ENSURE(isExpApprox == true, "Approx format does not match");
    }

    TLogLinQuantileError(double alpha, bool isExpApprox)
    : IDerCalcer(isExpApprox)
    , Alpha(alpha)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
        CB_ENSURE(isExpApprox == true, "Approx format does not match");
    }

private:
    double CalcDer(double approxExp, float target) const override {
        return (target - approxExp > 0) ? Alpha * approxExp : -(1 - Alpha) * approxExp;
    }

    double CalcDer2(double /*approx*/, float /*target*/) const override {
        return QUANTILE_DER2_AND_DER3;
    }

    double CalcDer3(double /*approx*/, float /*target*/) const override {
        return QUANTILE_DER2_AND_DER3;
    }
};

class TMAPError final : public IDerCalcer {
public:
    static constexpr double MAPE_DER2_AND_DER3 = 0.0;

    explicit TMAPError(bool isExpApprox)
    : IDerCalcer(isExpApprox)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        return (target - approx > 0) ? 1 / target : -1 / target;
    }

    double CalcDer2(double /*approx*/, float /*target*/) const override {
        return MAPE_DER2_AND_DER3;
    }

    double CalcDer3(double /*approx*/, float /*target*/) const override {
        return MAPE_DER2_AND_DER3;
    }
};

class TPoissonError final : public IDerCalcer {
public:
    explicit TPoissonError(bool isExpApprox)
    : IDerCalcer(isExpApprox)
    {
        CB_ENSURE(isExpApprox == true, "Approx format does not match");
    }

private:
    double CalcDer(double approxExp, float target) const override {
        return target - approxExp;
    }

    double CalcDer2(double approxExp, float) const override {
        return -approxExp;
    }

    double CalcDer3(double approxExp, float /*target*/) const override {
        return -approxExp;
    }
};

class TMultiClassError final : public IDerCalcer {
public:
    explicit TMultiClassError(bool isExpApprox)
    : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        const int approxDimension = approx.ysize();

        TVector<double> softmax(approxDimension);
        CalcSoftmax(approx, &softmax);

        for (int dim = 0; dim < approxDimension; ++dim) {
            (*der)[dim] = -softmax[dim];
        }
        int targetClass = static_cast<int>(target);
        (*der)[targetClass] += 1;

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Symmetric &&
                     der2->ApproxDimension == approxDimension);
            int idx = 0;
            for (int dimY = 0; dimY < approxDimension; ++dimY) {
                der2->Data[idx++] = softmax[dimY] * (softmax[dimY] - 1);
                for (int dimX = dimY + 1; dimX < approxDimension; ++dimX) {
                    der2->Data[idx++] = softmax[dimY] * softmax[dimX];
                }
            }
        }

        if (weight != 1) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                (*der)[dim] *= weight;
            }
            if (der2 != nullptr) {
                int idx = 0;
                for (int dimY = 0; dimY < approxDimension; ++dimY) {
                    for (int dimX = dimY; dimX < approxDimension; ++dimX) {
                        der2->Data[idx++] *= weight;
                    }
                }
            }
        }
    }
};

class TMultiClassOneVsAllError final : public IDerCalcer {
public:
    explicit TMultiClassOneVsAllError(bool isExpApprox)
    : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2, EErrorType::PerObjectError, EHessianType::Diagonal)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        const int approxDimension = approx.ysize();

        TVector<double> prob = approx;
        FastExpInplace(prob.data(), prob.ysize());
        for (int dim = 0; dim < approxDimension; ++dim) {
            prob[dim] /= (1 + prob[dim]);
            (*der)[dim] = -prob[dim];
        }
        int targetClass = static_cast<int>(target);
        (*der)[targetClass] += 1;

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal &&
                     der2->ApproxDimension == approxDimension);
            for (int dim = 0; dim < approxDimension; ++ dim) {
                der2->Data[dim] = -prob[dim] * (1 - prob[dim]);
            }
        }

        if (weight != 1) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                (*der)[dim] *= weight;
            }
            if (der2 != nullptr) {
                for (int dim = 0; dim < approxDimension; ++dim) {
                    der2->Data[dim] *= weight;
                }
            }
        }
    }
};

class TPairLogitError final : public IDerCalcer {
public:
    explicit TPairLogitError(bool isExpApprox)
    : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2, EErrorType::PairwiseError)
    {
        CB_ENSURE(isExpApprox == true, "Approx format does not match");
    }

    void CalcDersForQueries(
        int queryStartIndex,
        int queryEndIndex,
        const TVector<double>& expApproxes,
        const TVector<float>& /*targets*/,
        const TVector<float>& /*weights*/,
        const TVector<TQueryInfo>& queriesInfo,
        TArrayRef<TDers> ders,
        NPar::TLocalExecutor* localExecutor
    ) const override {
        CB_ENSURE(queryStartIndex < queryEndIndex);
        const int start = queriesInfo[queryStartIndex].Begin;
        NPar::ParallelFor(*localExecutor, queryStartIndex, queryEndIndex, [&] (ui32 queryIndex) {
            const int begin = queriesInfo[queryIndex].Begin;
            const int end = queriesInfo[queryIndex].End;
            TDers* dersData = ders.data() + begin - start;
            Fill(dersData, dersData + end - begin, TDers{/*1st*/0.0, /*2nd*/0.0, /*3rd*/0.0});
            for (int docId = begin; docId < end; ++docId) {
                double winnerDer = 0.0;
                double winnerSecondDer = 0.0;
                for (const auto& competitor : queriesInfo[queryIndex].Competitors[docId - begin]) {
                    const double p = expApproxes[competitor.Id + begin] / (expApproxes[competitor.Id + begin] + expApproxes[docId]);
                    winnerDer += competitor.Weight * p;
                    dersData[competitor.Id].Der1 -= competitor.Weight * p;
                    winnerSecondDer += competitor.Weight * p * (p - 1);
                    dersData[competitor.Id].Der2 += competitor.Weight * p * (p - 1);
                }
                dersData[docId - begin].Der1 += winnerDer;
                dersData[docId - begin].Der2 += winnerSecondDer;
            }
        });
    }
};

class TQueryRmseError final : public IDerCalcer {
public:
    explicit TQueryRmseError(bool isExpApprox)
    : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2, EErrorType::QuerywiseError)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    void CalcDersForQueries(
        int queryStartIndex,
        int queryEndIndex,
        const TVector<double>& approxes,
        const TVector<float>& targets,
        const TVector<float>& weights,
        const TVector<TQueryInfo>& queriesInfo,
        TArrayRef<TDers> ders,
        NPar::TLocalExecutor* localExecutor
    ) const override {
        const int start = queriesInfo[queryStartIndex].Begin;
        NPar::ParallelFor(*localExecutor, queryStartIndex, queryEndIndex, [&] (ui32 queryIndex) {
            const int begin = queriesInfo[queryIndex].Begin;
            const int end = queriesInfo[queryIndex].End;
            const int querySize = end - begin;

            const double queryAvrg = CalcQueryAvrg(begin, querySize, approxes, targets, weights);
            for (int docId = begin; docId < end; ++docId) {
                ders[docId - start].Der1 = targets[docId] - approxes[docId] - queryAvrg;
                ders[docId - start].Der2 = -1;
                if (!weights.empty()) {
                    ders[docId - start].Der1 *= weights[docId];
                    ders[docId - start].Der2 *= weights[docId];
                }
            }
        });
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

class TQuerySoftMaxError final : public IDerCalcer {
public:

    const double LambdaReg;

    explicit TQuerySoftMaxError(double lambdaReg, bool isExpApprox)
    : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2, EErrorType::QuerywiseError)
    , LambdaReg(lambdaReg)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    void CalcDersForQueries(
        int queryStartIndex,
        int queryEndIndex,
        const TVector<double>& approxes,
        const TVector<float>& targets,
        const TVector<float>& weights,
        const TVector<TQueryInfo>& queriesInfo,
        TArrayRef<TDers> ders,
        NPar::TLocalExecutor* localExecutor
    ) const override {
        int start = queriesInfo[queryStartIndex].Begin;
        NPar::ParallelFor(*localExecutor, queryStartIndex, queryEndIndex, [&](int queryIndex) {
            int begin = queriesInfo[queryIndex].Begin;
            int end = queriesInfo[queryIndex].End;
            CalcDersForSingleQuery(start, begin - start, end - begin, approxes, targets, weights, ders);
        });
    }

private:
    void CalcDersForSingleQuery(
        int start,
        int offset,
        int count,
        TConstArrayRef<double> approxes,
        TConstArrayRef<float> targets,
        TConstArrayRef<float> weights,
        TArrayRef<TDers> ders
    ) const;
};

class TCustomError final : public IDerCalcer {
public:
    TCustomError(
        const NCatboostOptions::TCatBoostOptions& params,
        const TMaybe<TCustomObjectiveDescriptor>& descriptor
    )
        : IDerCalcer(/*IsExpApprox*/ false)
        , Descriptor(*descriptor)
    {
        CB_ENSURE(IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()) == false, "Approx format does not match");
    }

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        Descriptor.CalcDersMulti(approx, target, weight, der, der2, Descriptor.CustomData);
    }

    void CalcDersRange(
        int start,
        int count,
        bool /*calcThirdDer*/,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDers* ders
    ) const override {
        memset(ders + start, 0, sizeof(*ders) * count);
        if (approxDeltas != nullptr) {
            TVector<double> updatedApproxes(count);
            for (int i = start; i < start + count; ++i) {
                updatedApproxes[i - start] = approxes[i] + approxDeltas[i];
            }
            Descriptor.CalcDersRange(count, updatedApproxes.data(), targets + start, weights ? (weights + start) : nullptr, ders + start, Descriptor.CustomData);
        } else {
            Descriptor.CalcDersRange(count, approxes + start, targets + start, weights ? (weights + start) : nullptr, ders + start, Descriptor.CustomData);
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
    ) const override {
        TVector<TDers> derivatives(count, {0.0, 0.0, 0.0});
        CalcDersRange(start, count, /*calcThirdDer=*/false, approxes, approxDeltas, targets, weights, derivatives.data() - start);
        for (int i = start; i < start + count; ++i) {
            ders[i] = derivatives[i - start].Der1;
        }
    }
private:
    TCustomObjectiveDescriptor Descriptor;
};

inline double GetNumericParameter(const TMap<TString, TString>& params, const TString& key) {
    if (params.contains(key)) {
        return FromString<double>(params.at(key));
    } else {
        return 0.0;
    }
}

class TUserDefinedPerObjectError final : public IDerCalcer {
public:
    const double Alpha;

    TUserDefinedPerObjectError(const TMap<TString, TString>& params, bool isExpApprox)
    : IDerCalcer(isExpApprox)
    , Alpha(GetNumericParameter(params, "alpha"))
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }
};

class TUserDefinedQuerywiseError final : public IDerCalcer {
public:
    const double Alpha;

    TUserDefinedQuerywiseError(const TMap<TString, TString>& params, bool isExpApprox)
    : IDerCalcer(isExpApprox, EErrorType::QuerywiseError)
    , Alpha(GetNumericParameter(params, "alpha"))
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }
};


void CheckDerivativeOrderForTrain(ui32 derivativeOrder, ELeavesEstimation estimationMethod);
void CheckDerivativeOrderForObjectImportance(ui32 derivativeOrder, ELeavesEstimation estimationMethod);
