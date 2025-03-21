#pragma once

#include "approx_updater_helpers.h"
#include "custom_objective_descriptor.h"
#include "ders_holder.h"
#include "hessian.h"
#include "survival_aft_utils.h"

#include <catboost/private/libs/data_types/pair.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/string/split.h>
#include <util/system/yassert.h>

#include <cmath>
#include <memory>

class IDerCalcer {
public:
    explicit IDerCalcer(
        bool isExpApprox,
        ui32 maxDerivativeOrder = 3,
        EErrorType errorType = EErrorType::PerObjectError,
        EHessianType hessianType = EHessianType::Symmetric
    )
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
        CalcDersRange(
            start,
            count,
            /*maxDerivativeOrder*/ 1,
            approxes,
            approxDeltas,
            targets,
            weights,
            /*ders*/ nullptr,
            firstDers);
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
        CalcDersRange(
            start,
            count,
            maxDerivativeOrder,
            approxes,
            approxDeltas,
            targets,
            weights,
            ders,
            /*firstDers*/ nullptr);
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
        ui64 /*randomSeed*/,
        NPar::ILocalExecutor* /*localExecutor*/
    ) const {
        CB_ENSURE(false, "Not implemented");
    }

private:
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

private:
    const bool IsExpApprox;
    const ui32 MaxSupportedDerivativeOrder;
    const EErrorType ErrorType;
    const EHessianType HessianType;
};

class TMultiDerCalcer : public IDerCalcer {
public:
    static constexpr int MaxDerivativeOrder = 2;

    explicit TMultiDerCalcer(EHessianType hessianType = EHessianType::Symmetric)
        : IDerCalcer(/*isExpApprox*/false, MaxDerivativeOrder, EErrorType::PerObjectError, hessianType)
    {
    }

    virtual void CalcDers(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const = 0;
};

class TMultiRMSEError final : public TMultiDerCalcer {
public:
    explicit TMultiRMSEError()
        : TMultiDerCalcer(EHessianType::Diagonal) {
    }

    void CalcDers(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        const int dim = target.size();
        for (auto i : xrange(dim)) {
            (*der)[i] = weight * (target[i] - approx[i]);
        }

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal &&
                     der2->ApproxDimension == dim);

            for (auto i : xrange(dim)) {
                der2->Data[i] = -weight;
            }
        }
    }
};

class TMultiRMSEErrorWithMissingValues final : public TMultiDerCalcer {
public:
    explicit TMultiRMSEErrorWithMissingValues()
        : TMultiDerCalcer(EHessianType::Diagonal) {
    }

    void CalcDers(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        const int dim = target.size();
        for (auto i : xrange(dim)) {
            if (IsNan(target[i])) {
                (*der)[i] = 0.0;
            } else {
                (*der)[i] = weight * (target[i] - approx[i]);
            }
        }

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal &&
                     der2->ApproxDimension == dim);

            for (auto i : xrange(dim)) {
                if (IsNan(target[i])) {
                    der2->Data[i] = 0.0;
                } else {
                    der2->Data[i] = -weight;
                }
            }
        }
    }
};

class TSurvivalAftError final : public TMultiDerCalcer {
public:
    const double Scale;
    std::unique_ptr<NCB::IDistribution> Distribution;

public:
    explicit TSurvivalAftError(std::unique_ptr<NCB::IDistribution> distribution, double scale)
        : TMultiDerCalcer(EHessianType::Diagonal)
        , Scale(scale)
        , Distribution(std::move(distribution))
    {
        CB_ENSURE(Scale > 0, "Scale should be positive");
    }

    void CalcDers(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override;
};

class TRMSEWithUncertaintyError final : public TMultiDerCalcer {
public:
    explicit TRMSEWithUncertaintyError()
        : TMultiDerCalcer(EHessianType::Diagonal)
    {
    }

    void CalcDers(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        // computes ngboost "natural gradient"
        // should be regular gradient multiplied by Fisher information
        // see https://arxiv.org/pdf/1910.03225v1.pdf
        const int dim = 2;
        Y_ASSERT(target.size() == 1);
        const double diff = (target[0] - approx[0]);
        double prec = -2 * approx[1];
        NCB::FastExpWithInfInplace(&prec, /*count*/ 1);
        (*der)[0] = weight * diff;
        (*der)[1] = weight * (Sqr(diff) * prec - 1);

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal &&
                     der2->ApproxDimension == dim);

            der2->Data[0] = -weight;
            der2->Data[1] = -2 * weight * Sqr(diff) * prec;
        }
    }
};

class TMultiQuantileError final : public IDerCalcer {
public:
    static constexpr double QUANTILE_DER2_AND_DER3 = 0.0;

public:
    const TVector<double> Alpha;
    const double Delta;

public:
    explicit TMultiQuantileError(bool isExpApprox)
        : IDerCalcer(isExpApprox)
        , Alpha({0.5})
        , Delta(1e-6)
    {
    }

    TMultiQuantileError(const TVector<double>& alpha, double delta, bool isExpApprox)
        : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2, /*errorType*/ PerObjectError, EHessianType::Diagonal)
        , Alpha(alpha)
        , Delta(delta)
    {
        Y_ASSERT(AllOf(Alpha, [] (double a) { return a > -1e-6 && a < 1.0 + 1e-6; }));
        Y_ASSERT(Delta >= 0 && Delta <= 1e-2);
    }

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override;

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

public:
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

class TLogCoshError final : public IDerCalcer {
public:
    explicit TLogCoshError(bool isExpApprox)
        : IDerCalcer(isExpApprox)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        return -tanh(approx - target);
    }

    double CalcDer2(double approx, float target) const override {
        return -1 / (cosh(approx - target) * cosh(approx - target));
    }

    double CalcDer3(double approx, float target) const override {
        return 2 * tanh(approx - target) / (cosh(approx - target) * cosh(approx - target));
    }
};

class TCoxError final : public IDerCalcer {
public:
    explicit TCoxError(bool isExpApprox, ui32 maxDerivativeOrder = 3)
        : IDerCalcer(isExpApprox, maxDerivativeOrder)
    {
        CB_ENSURE_INTERNAL(!isExpApprox, "Cox objective requires isExpApprox == false");
    }

    void CalcDersRange(
        int start,
        int count,
        bool calcThirdDer,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        TDers* ders
    ) const;

    void CalcFirstDerRange(
        int start,
        int count,
        const double* approxes,
        const double* approxDeltas,
        const float* targets,
        const float* weights,
        double* firstDers
    ) const;
};

class TQuantileError final : public IDerCalcer {
public:
    static constexpr double QUANTILE_DER2_AND_DER3 = 0.0;

public:
    const double Alpha;
    const double Delta;

public:
    explicit TQuantileError(bool isExpApprox)
        : IDerCalcer(isExpApprox)
        , Alpha(0.5)
        , Delta(1e-6)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    TQuantileError(double alpha, double delta, bool isExpApprox)
        : IDerCalcer(isExpApprox)
        , Alpha(alpha)
        , Delta(delta)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
        Y_ASSERT(Delta >= 0 && Delta <= 1e-2);
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        const double val = target - approx;
        if (abs(val) < Delta) return 0;
        return (target - approx > 0) ? Alpha : -(1 - Alpha);
    }

    double CalcDer2(double /*approx*/, float /*target*/) const override {
        return QUANTILE_DER2_AND_DER3;
    }

    double CalcDer3(double /*approx*/, float /*target*/) const override {
        return QUANTILE_DER2_AND_DER3;
    }
};

class TExpectileError final : public IDerCalcer {
public:
    static constexpr double EXPECTILE_DER3 = 0.0;

public:
    const double Alpha;

public:
    explicit TExpectileError(bool isExpApprox)
        : IDerCalcer(isExpApprox)
        , Alpha(0.5)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    TExpectileError(double alpha, bool isExpApprox)
        : IDerCalcer(isExpApprox)
        , Alpha(alpha)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        double e = target - approx;
        return (e > 0) ? 2.0 * Alpha * e : 2.0 * (1 - Alpha) * e;
    }

    double CalcDer2(double approx, float target) const override {
        double e = target - approx;
        return (e > 0) ? -2.0 * Alpha : -2.0 * (1 - Alpha);
    }

    double CalcDer3(double /*approx*/, float /*target*/) const override {
        return EXPECTILE_DER3;
    }
};

class TLqError final : public IDerCalcer {
public:
    const double Q;

public:
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
        return Q * (target - approx > 0 ? 1 : -1)  * absLossQ;
    }

    double CalcDer2(double approx, float target) const override {
        const double absLoss = abs(target - approx);
        return -Q * (Q - 1) * std::pow(absLoss, Q - 2);
    }

    double CalcDer3(double approx, float target) const override {
        const double absLoss = abs(target - approx);
        return Q * (Q - 1) *  (Q - 2) * std::pow(absLoss, Q - 3) * (target - approx > 0 ? 1 : -1);
    }
};

class TLogLinQuantileError final : public IDerCalcer {
public:
    static constexpr double QUANTILE_DER2_AND_DER3 = 0.0;

public:
    const double Alpha;

public:
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

public:
    explicit TMAPError(bool isExpApprox)
        : IDerCalcer(isExpApprox)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        return (target - approx > 0) ? 1 / Max(1.f, Abs(target)) : -1 / Max(1.f, Abs(target));
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
        const auto derRef = MakeArrayRef(*der);
        CalcSoftmax(approx, derRef);

        if (der2 != nullptr) {
            const int approxDimension = approx.ysize();
            Y_ASSERT(der2->HessianType == EHessianType::Symmetric &&
                     der2->ApproxDimension == approxDimension);
            const auto der2Ref = MakeArrayRef(der2->Data);
            int idx = 0;
            for (int dimY : xrange(approxDimension)) {
                const double derY = derRef[dimY];
                der2Ref[idx++] = derY * (derY - 1);
                for (int dimX : xrange(dimY + 1, approxDimension)) {
                    der2Ref[idx++] = derY * derRef[dimX];
                }
            }
        }

        for (auto& value : derRef) {
            value = -value;
        }
        const int targetClass = static_cast<int>(target);
        derRef[targetClass] += 1;

        if (weight != 1) {
            for (auto& value : derRef) {
                value *= weight;
            }
            if (der2 != nullptr) {
                const auto der2Ref = MakeArrayRef(der2->Data);
                for (auto& value : der2Ref) {
                    value *= weight;
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
        NCB::FastExpWithInfInplace(prob.data(), prob.ysize());
        for (int dim = 0; dim < approxDimension; ++dim) {
            prob[dim] /= (1 + prob[dim]);
            (*der)[dim] = -prob[dim];
        }
        int targetClass = static_cast<int>(target);
        (*der)[targetClass] += 1;

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal && der2->ApproxDimension == approxDimension);
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

class TMultiCrossEntropyError final : public TMultiDerCalcer {
public:
    explicit TMultiCrossEntropyError()
        : TMultiDerCalcer(EHessianType::Diagonal){}

    void CalcDers(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        const int approxDimension = approx.ysize();
        TArrayRef<double> derRef(*der);
        CopyN(approx.data(), approx.ysize(), derRef.data());

        NCB::FastExpWithInfInplace(derRef.data(), derRef.ysize());
        for (int dim = 0; dim < approxDimension; ++dim) {
            derRef[dim] = -derRef[dim] / (1 + derRef[dim]);
        }

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal && der2->ApproxDimension == approxDimension);
            for (int dim = 0; dim < approxDimension; ++ dim) {
                der2->Data[dim] = derRef[dim] * (1 + derRef[dim]);
            }
        }

        for (int dim = 0; dim < approxDimension; ++dim) {
            derRef[dim] += target[dim];
        }

        if (weight != 1) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                derRef[dim] *= weight;
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
        ui64 /*randomSeed*/,
        NPar::ILocalExecutor* localExecutor
    ) const override {
        CB_ENSURE(queryStartIndex < queryEndIndex);
        const int start = queriesInfo[queryStartIndex].Begin;
        NPar::ParallelFor(
            *localExecutor,
            queryStartIndex,
            queryEndIndex,
            [&] (ui32 queryIndex) {
                const int begin = queriesInfo[queryIndex].Begin;
                const int end = queriesInfo[queryIndex].End;
                TDers* dersData = ders.data() + begin - start;
                Fill(dersData, dersData + end - begin, TDers{/*1st*/0.0, /*2nd*/0.0, /*3rd*/0.0});
                for (int docId = begin; docId < end; ++docId) {
                    double winnerDer = 0.0;
                    double winnerSecondDer = 0.0;
                    for (const auto& competitor : queriesInfo[queryIndex].Competitors[docId - begin]) {
                        const double p = expApproxes[competitor.Id + begin] /
                            (expApproxes[competitor.Id + begin] + expApproxes[docId]);
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
        ui64 /*randomSeed*/,
        NPar::ILocalExecutor* localExecutor
    ) const override {
        const int start = queriesInfo[queryStartIndex].Begin;
        NPar::ParallelFor(
            *localExecutor,
            queryStartIndex,
            queryEndIndex,
            [&] (ui32 queryIndex) {
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

class TGroupQuantileError final : public IDerCalcer {
public:
    static constexpr double QUANTILE_DER2_AND_DER3 = 0.0;

public:
    const double Alpha;
    const double Delta;

public:
    explicit TGroupQuantileError(bool isExpApprox)
        : IDerCalcer(isExpApprox, EErrorType::QuerywiseError)
        , Alpha(0.5)
        , Delta(1e-6)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

    explicit TGroupQuantileError(double alpha, double delta, bool isExpApprox)
        : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 3, /* errorType */ EErrorType::QuerywiseError)
        , Alpha(alpha)
        , Delta(delta)
    {
        Y_ASSERT(Alpha > -1e-6 && Alpha < 1.0 + 1e-6);
        Y_ASSERT(Delta >= 0 && Delta <= 1e-2);
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
        ui64 /*randomSeed*/,
        NPar::ILocalExecutor* localExecutor
    ) const override {
        const int start = queriesInfo[queryStartIndex].Begin;
        NPar::ParallelFor(
            *localExecutor,
            queryStartIndex,
            queryEndIndex,
            [&] (ui32 queryIndex) {
                const int begin = queriesInfo[queryIndex].Begin;
                const int end = queriesInfo[queryIndex].End;
                const int querySize = end - begin;

                const double queryAvrg = CalcQueryAvrg(begin, querySize, approxes, targets, weights);
                for (int docId = begin; docId < end; ++docId) {
                    const double val = targets[docId] - approxes[docId] - queryAvrg;
                    ders[docId - start].Der2 = QUANTILE_DER2_AND_DER3;
                    ders[docId - start].Der3 = QUANTILE_DER2_AND_DER3;
                    if (!weights.empty()) {
                        ders[docId - start].Der2 *= weights[docId];
                        ders[docId - start].Der3 *= weights[docId];
                    }
                    if (abs(val) < Delta) {
                        ders[docId - start].Der1 = 0;
                        continue;
                    }
                    ders[docId - start].Der1 = (val > 0) ? Alpha : -(1 - Alpha);
                    if (!weights.empty()) {
                        ders[docId - start].Der1 *= weights[docId];
                    }
                }
            }
        );
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
    const double Beta;

public:
    explicit TQuerySoftMaxError(double lambdaReg, double beta, bool isExpApprox)
        : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2, EErrorType::QuerywiseError)
        , LambdaReg(lambdaReg)
        , Beta(beta)
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
        ui64 /*randomSeed*/,
        NPar::ILocalExecutor* localExecutor
    ) const override {
        int start = queriesInfo[queryStartIndex].Begin;
        NPar::ParallelFor(
            *localExecutor,
            queryStartIndex,
            queryEndIndex,
            [&](int queryIndex) {
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
        CB_ENSURE(
            IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()) == false,
            "Approx format does not match");
    }

    void CalcDersMulti(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        Descriptor.CalcDersMultiClass(approx, target, weight, der, der2, Descriptor.CustomData);
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
            Descriptor.CalcDersRange(
                count,
                updatedApproxes.data(),
                targets + start,
                weights ? (weights + start) : nullptr,
                ders + start,
                Descriptor.CustomData);
        } else {
            Descriptor.CalcDersRange(
                count,
                approxes + start,
                targets + start,
                weights ? (weights + start) : nullptr,
                ders + start,
                Descriptor.CustomData);
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
        CalcDersRange(
            start,
            count, /*calcThirdDer=*/
            false,
            approxes,
            approxDeltas,
            targets,
            weights,
            derivatives.data() - start);
        for (int i = start; i < start + count; ++i) {
            ders[i] = derivatives[i - start].Der1;
        }
    }
private:
    TCustomObjectiveDescriptor Descriptor;
};

class TMultiTargetCustomError final : public TMultiDerCalcer {
public:

    TMultiTargetCustomError(
        const NCatboostOptions::TCatBoostOptions& params,
        const TMaybe<TCustomObjectiveDescriptor>& descriptor
    )
        : Descriptor(*descriptor) {
        CB_ENSURE(
            IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()) == false,
            "Approx format does not match");
    }

    void CalcDers(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* der,
        THessianInfo* der2
    ) const override {
        Descriptor.CalcDersMultiTarget(approx, target, weight, der, der2, Descriptor.CustomData);
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

public:
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

public:
    TUserDefinedQuerywiseError(const TMap<TString, TString>& params, bool isExpApprox)
        : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2, EErrorType::QuerywiseError)
        , Alpha(GetNumericParameter(params, "alpha"))
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }
};

/* TStochasticFilterError */
class TStochasticFilterError final : public IDerCalcer {
public:
    const double Sigma;
    const int NumEstimations;

public:
    TStochasticFilterError(double sigma, int numEstimations, bool isExpApprox)
        : IDerCalcer(false, 1, EErrorType::QuerywiseError)
        , Sigma(sigma)
        , NumEstimations(numEstimations)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
        CB_ENSURE(Sigma > 0, "Scale parameter 'sigma' for DCG-RR loss must be positive");
        CB_ENSURE(NumEstimations > 0, "Number of estimations must be positive integer");
    }

    void CalcDersForQueries(
        int queryStartIndex,
        int queryEndIndex,
        const TVector<double>& approx,
        const TVector<float>& target,
        const TVector<float>& /*weights*/,
        const TVector<TQueryInfo>& queriesInfo,
        TArrayRef<TDers> ders,
        ui64 randomSeed,
        NPar::ILocalExecutor* localExecutor
    ) const override {
        NPar::ILocalExecutor::TExecRangeParams blockParams(queryStartIndex, queryEndIndex);
        blockParams.SetBlockCount(CB_THREAD_LIMIT);
        const int blockSize = blockParams.GetBlockSize();
        const int blockCount = blockParams.GetBlockCount();
        const TVector<ui64> randomSeeds = GenRandUI64Vector(blockCount, randomSeed);
        const int start = queriesInfo[queryStartIndex].Begin;

        NPar::ParallelFor(
            *localExecutor,
            0,
            static_cast<ui32>(blockCount),
            [&](int blockId) {
                TRestorableFastRng64 rand(randomSeeds[blockId]);
                rand.Advance(10);
                const int from = blockId * blockSize;
                const int to = Min<int>((blockId + 1) * blockSize, queryEndIndex);
                for (int queryIndex = from; queryIndex < to; ++queryIndex) {
                    int begin = queriesInfo[queryIndex].Begin;
                    int end = queriesInfo[queryIndex].End;
                    CalcQueryDers(begin, begin - start, end - begin, approx, target, ders, &rand);
                }
            });
    }

private:
    void CalcQueryDers(
        int offset,
        int offsetDer,
        int querySize,
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        TArrayRef<TDers> ders,
        TRestorableFastRng64* Rand
    ) const {
        Fill(ders.begin() + offsetDer, ders.begin() + offsetDer + querySize, TDers { 0.f, 0.f, 0.f });
        const double baselineLoss = CalcBaseline(offset, querySize, approx, target);
        TVector<double> probs(querySize, 0.);

        for (int i = 0; i < NumEstimations; ++i) {
            int pos = 0;
            double loss = 0.0;

            for (int j = 0; j < querySize; ++j) {
                const double prob = Sigmoid(approx[offset + j] * Sigma);
                const bool isFiltered = prob >= Rand->GenRandReal1();
                loss += isFiltered ? target[offset + j] / (pos + 1) : 0.;
                pos += isFiltered;
                probs[j] = isFiltered ? (1 - prob) : -prob;
            }
            for (int j = 0; j < querySize; ++j) {
                ders[offsetDer + j].Der1 += probs[j] * (loss - baselineLoss) / NumEstimations;
            }
        }
    }

    double CalcBaseline(
        int offset,
        int count,
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target
    ) const {
        double baselineValue = 0.0;
        int pos = 0;
        for (int i = 0; i < count; ++i) {
            if (approx[offset + i] >= 0) {
                baselineValue += target[offset + i] / (++pos);
            }
        }
        return baselineValue;
    }
};

class TLambdaMartError final : public IDerCalcer{
    ELossFunction TargetMetric;
    int TopSize;
    ENdcgMetricType NumeratorType;          // for (N)DCG
    ENdcgDenominatorType DenominatorType;   // for (N)DCG
    double Sigma;
    bool Norm;

public:
    TLambdaMartError(
        ELossFunction targetMetric,
        const TMap<TString, TString>& metricParams,
        double sigma,
        bool norm);

    void CalcDersForQueries(
        int queryStartIndex,
        int queryEndIndex,
        const TVector<double>& approxes,
        const TVector<float>& target,
        const TVector<float>& /*weights*/,
        const TVector<TQueryInfo>& queriesInfo,
        TArrayRef<TDers> ders,
        ui64 /*randomSeed*/,
        NPar::ILocalExecutor* localExecutor
    ) const override;

private:
    void CalcDersForSingleQuery(
        TConstArrayRef<double> approxes,
        TConstArrayRef<float> targets,
        TArrayRef<TDers> ders
    ) const;

    inline double CalcNumerator(float target) const {
        return NumeratorType == ENdcgMetricType::Exp ? (Exp2(target) - 1) : target;
    }

    inline double CalcDenominator(size_t pos) const {
        return DenominatorType == ENdcgDenominatorType::LogPosition ? Log2(2.0 + pos) : (1.0 + pos);
    }

    inline size_t GetQueryTopSize(size_t docCount) const {
        if (TopSize == -1 || TopSize > (int)docCount) {
            return docCount;
        }
        return TopSize;
    }

    double CalcIdealMetric(TConstArrayRef<float> target, size_t queryTopSize) const;

    void CalcDersNDCG(
        TConstArrayRef<double> approxes,
        TConstArrayRef<float> targets,
        TConstArrayRef<size_t> order,
        TArrayRef<TDers> ders,
        double& sumDer1
    ) const;

    void CalcDersMRR(
        TConstArrayRef<double> approxes,
        TConstArrayRef<float> targets,
        TConstArrayRef<size_t> order,
        TArrayRef<TDers> ders,
        double& sumDer1
    ) const;

    void CalcDersERR(
        TConstArrayRef<double> approxes,
        TConstArrayRef<float> targets,
        TConstArrayRef<size_t> order,
        TArrayRef<TDers> ders,
        double& sumDer1
    ) const;

    void CalcDersMAP(
        TConstArrayRef<double> approxes,
        TConstArrayRef<float> targets,
        TConstArrayRef<size_t> order,
        TArrayRef<TDers> ders,
        double& sumDer1
    ) const;
};

class TStochasticRankError final : public IDerCalcer {
    ELossFunction TargetMetric;
    int TopSize;
    ENdcgMetricType NumeratorType;          // for (N)DCG
    ENdcgDenominatorType DenominatorType;   // for (N)DCG
    double Decay;                           // for PFound

    double Sigma;           // scale
    size_t NumEstimations;  // Monte Carlo method samples
    double Mu;              // ties resolving coefficient
    double Nu;              // approx norm addition
    double Lambda;          // SFA coefficient

    static constexpr double INV_SQRT_2PI = 0.398942280401432677939946;

public:
    TStochasticRankError(
        ELossFunction targetMetric,
        const TMap<TString, TString>& metricParams,
        double sigma,
        size_t numEstimations,
        double mu,
        double nu,
        double lambda);

    void CalcDersForQueries(
        int queryStartIndex,
        int queryEndIndex,
        const TVector<double>& approxes,
        const TVector<float>& target,
        const TVector<float>& /*weights*/,
        const TVector<TQueryInfo>& queriesInfo,
        TArrayRef<TDers> ders,
        ui64 randomSeed,
        NPar::ILocalExecutor* localExecutor
    ) const override;

private:
    void CalcDersForSingleQuery(
        TConstArrayRef<double> approxes,
        TConstArrayRef<float> targets,
        ui64 randomSeed,
        TArrayRef<TDers> ders
    ) const;

    void CalcMonteCarloEstimateForSingleQueryPermutation(
        TConstArrayRef<float> targets,
        const TVector<double>& approxes,
        const TVector<double>& scores,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        const double noiseSum,
        TArrayRef<TDers> ders
    ) const;

    double CalcDCGMetricDiff(
        size_t oldPos,
        size_t newPos,
        const TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        const TVector<double>& scores,
        const TVector<double>& cumSum,
        const TVector<double>& cumSumUp,
        const TVector<double>& cumSumLow
    ) const;

    double CalcPFoundMetricDiff(
        size_t oldPos,
        size_t newPos,
        size_t queryTopSize,
        const TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        const TVector<double>& cumSum
    ) const;

    double CalcERRMetricDiff(
        size_t oldPos,
        size_t newPos,
        size_t queryTopSize,
        const TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        const TVector<double>& cumSum,
        const TVector<double>& cumSumUp,
        const TVector<double>& cumSumLow
    ) const;

    double CalcMRRMetricDiff(
        size_t oldPos,
        size_t newPos,
        const TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        int firstRelevPos,
        int secondRelevPos
    ) const;

    double CalcMetricDiff(
        size_t oldPos,
        size_t newPos,
        size_t queryTopSize,
        const TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        const TVector<double>& scores,
        const TVector<double>& cumSum,
        const TVector<double>& cumSumUp,
        const TVector<double>& cumSumLow,
        int firstRelevPos,
        int secondRelevPos
    ) const;

    void CalcDCGCumulativeStatistics(
        TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        const TVector<double>& scores,
        TArrayRef<double> cumSumRef,
        TArrayRef<double> cumSumUpRef,
        TArrayRef<double> cumSumLowRef
    ) const;

    void CalcPFoundCumulativeStatistics(
        TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        TArrayRef<double> cumSum
    ) const;

    void CalcERRCumulativeStatistics(
        TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        const TVector<double>& posWeights,
        TArrayRef<double> cumSumRef,
        TArrayRef<double> cumSumUpRef,
        TArrayRef<double> cumSumLowRef
    ) const;

    void CalcMRRStatistics(
        TConstArrayRef<float> targets,
        const TVector<size_t>& order,
        int* firstRelevPos,
        int* secondRelevPos
    ) const;

    TVector<double> ComputeDCGPosWeights(
        TConstArrayRef<float> targets
    ) const;

    TVector<double> ComputePFoundPosWeights(
        TConstArrayRef<float> targets,
        const TVector<size_t>& order
    ) const;

    TVector<double> ComputeERRPosWeights(
        TConstArrayRef<float> targets,
        const TVector<size_t>& order
    ) const;

    double CalcDCG(const TVector<float>& sortedTargets, const TVector<double>& posWeights) const;

    inline double CalcNumerator(float target) const {
        return NumeratorType == ENdcgMetricType::Exp ? (Exp2(target) - 1) : target;
    }

    inline double CalcDenominator(size_t pos) const {
        return DenominatorType == ENdcgDenominatorType::LogPosition ? Log2(2.0 + pos) : (1.0 + pos);
    }

    inline size_t GetQueryTopSize(size_t docCount) const {
        if (TopSize == -1 || TopSize > (int)docCount) {
            return docCount;
        }
        return TopSize;
    }

    inline double NormalDensity(double x, double mean, double sigma) const {
        const long double z = Sqr((x - mean) / sigma);
        return std::expl(-z / 2.0) * INV_SQRT_2PI / sigma;
    }

    inline double NormalDensityDiff(double x1, double x2, double mean, double sigma) const {
        return NormalDensity(x1, mean, sigma) - NormalDensity(x2, mean, sigma);
    }
};

class THuberError final : public IDerCalcer {
    static constexpr double HUBER_DER2 = -1.0;
    static constexpr double HUBER_DER3 = 0.0;

    const double Delta;
public:

    explicit THuberError(double delta, bool isExpApprox)
        : IDerCalcer(isExpApprox)
        , Delta(delta)
    {
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        double diff = target - approx;
        if (fabs(diff) < Delta) {
            return diff;
        } else {
            return diff > 0.0 ? Delta : -Delta;
        }
    }

    double CalcDer2(double approx, float target) const override {
        double diff = target - approx;
        if (fabs(diff) < Delta) {
            return HUBER_DER2;
        } else {
            return 0.0;
        }
    }

    double CalcDer3(double /*approx*/, float /*target*/) const override {
        return HUBER_DER3;
    }
};

class TTweedieError final : public IDerCalcer {
public:
    const double VariancePower;

public:
    TTweedieError(double variance_power, bool isExpApprox)
        : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 3)
        , VariancePower(variance_power)
    {
        Y_ASSERT(VariancePower > 1 && VariancePower < 2);
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        double der = target * std::exp((1 - VariancePower) * approx);
        der -= std::exp((2 - VariancePower) * approx);
        return der;
    }

    double CalcDer2(double approx, float target) const override {
        double der2 = target * std::exp((1 - VariancePower) * approx) * (1 - VariancePower);
        der2 -= std::exp((2 - VariancePower) * approx) * (2 - VariancePower);
        return der2;
    }

    double CalcDer3(double approx, float target) const override {
        double der3 = target * std::exp((1 - VariancePower) * approx) * Sqr(1 - VariancePower);
        der3 -= std::exp((2 - VariancePower) * approx) * Sqr(2 - VariancePower);
        return der3;
    }
};

void CheckDerivativeOrderForObjectImportance(ui32 derivativeOrder, ELeavesEstimation estimationMethod);

class TFocalError final : public IDerCalcer {
public:
    const double FocalAlpha;
    const double FocalGamma;

public:
    TFocalError(double alpha, double gamma, bool isExpApprox)
        : IDerCalcer(isExpApprox, /*maxDerivativeOrder*/ 2)
        , FocalAlpha(alpha), FocalGamma(gamma)
    {
        Y_ASSERT(FocalAlpha > 0 && FocalAlpha < 1 && FocalGamma > 0);
        CB_ENSURE(isExpApprox == false, "Approx format does not match");
    }

private:
    double CalcDer(double approx, float target) const override {
        double approx_exp, at, p, pt, y, der;
        approx_exp = 1 / (1 + exp(-approx));
        at = target == 1 ? FocalAlpha : 1 - FocalAlpha;
        p = std::clamp(approx_exp, 0.0000000000001, 0.9999999999999);
        pt = target == 1 ? p : 1 - p;
        y = 2 * target - 1;
        der = at * y * pow((1 - pt), FocalGamma);
        der = der * (FocalGamma * pt * log(pt) + pt - 1);
        return -der;
    }

    double CalcDer2(double approx, float target) const override {
        double approx_exp, at, p, pt, y, u, du, v, dv, der2;
        approx_exp = 1 / (1 + exp(-approx));
        at = target == 1 ? FocalAlpha : 1 - FocalAlpha;
        p = std::clamp(approx_exp, 0.0000000000001, 0.9999999999999);
        pt = target == 1 ? p : 1 - p;
        y = 2 * target - 1;
        u = at * y * pow((1 - pt), FocalGamma);
        du = -at * y * FocalGamma * pow((1 - pt), FocalGamma - 1);
        v = FocalGamma * pt * log(pt) + pt - 1;
        dv = FocalGamma * log(pt) + FocalGamma + 1;
        der2 = (du * v + u * dv) * y * (pt * (1 - pt));
        return -der2;
    }

};
