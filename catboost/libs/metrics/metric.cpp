#include "metric.h"
#include "auc.h"
#include "balanced_accuracy.h"
#include "brier_score.h"
#include "classification_utils.h"
#include "dcg.h"
#include "doc_comparator.h"
#include "hinge_loss.h"
#include "kappa.h"
#include "llp.h"
#include "pfound.h"
#include "precision_recall_at_k.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/loss_description.h>

#include <library/fast_exp/fast_exp.h>
#include <library/fast_log/fast_log.h>

#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/ymath.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/iterator.h>
#include <util/string/printf.h>
#include <util/system/yassert.h>

#include <limits>

/* TMetric */

static inline double OverflowSafeLogitProb(double approx) {
    double expApprox = exp(approx);
    return approx < 200 ? expApprox / (1.0 + expApprox) : 1.0;
}

template <typename T>
static inline TString BuildDescription(const TMetricParam<T>& param) {
    if (param.IsUserDefined()) {
        return TStringBuilder() << param.GetName() << "=" << ToString(param.Get());
    }
    return {};
}

template <>
inline TString BuildDescription<bool>(const TMetricParam<bool>& param) {
    if (param.IsUserDefined()) {
        return TStringBuilder() << param.GetName() << "=" << (param.Get() ? "true" : "false");
    }
    return {};
}

template <typename T>
static inline TString BuildDescription(const char* fmt, const TMetricParam<T>& param) {
    if (param.IsUserDefined()) {
        return TStringBuilder() << param.GetName() << "=" << Sprintf(fmt, param.Get());
    }
    return {};
}

template <typename T, typename... TRest>
static inline TString BuildDescription(const TMetricParam<T>& param, const TRest&... rest) {
    const TString& head = BuildDescription(param);
    const TString& tail = BuildDescription(rest...);
    const TString& sep = (head.empty() || tail.empty()) ? "" : ";";
    return TStringBuilder() << head << sep << tail;
}

template <typename T, typename... TRest>
static inline TString BuildDescription(const char* fmt, const TMetricParam<T>& param, const TRest&... rest) {
    const TString& head = BuildDescription(fmt, param);
    const TString& tail = BuildDescription(rest...);
    const TString& sep = (head.empty() || tail.empty()) ? "" : ";";
    return TStringBuilder() << head << sep << tail;
}

template <typename... TParams>
static inline TString BuildDescription(ELossFunction lossFunction, const TParams&... params) {
    const TString& tail = BuildDescription(params...);
    const TString& sep = tail.empty() ? "" : ":";
    return TStringBuilder() << ToString(lossFunction) << sep << tail;
}

template <typename... TParams>
static inline TString BuildDescription(const TString& description, const TParams&... params) {
    Y_ASSERT(!description.empty());
    const TString& tail = BuildDescription(params...);
    const TString& sep = tail.empty() ? "" : description.Contains(':') ? ";" : ":";
    return TStringBuilder() << description << sep << tail;
}

static inline TMetricParam<double> MakeBorderParam(double border) {
    return {"border", border, border != GetDefaultClassificationBorder()};
}

EErrorType TMetric::GetErrorType() const {
    return EErrorType::PerObjectError;
}

double TMetric::GetFinalError(const TMetricHolder& error) const {
    Y_ASSERT(error.Stats.size() == 2);
    return error.Stats[0] / (error.Stats[1] + 1e-38);
}

TVector<TString> TMetric::GetStatDescriptions() const {
    return {"SumError", "SumWeight"};
}

const TMap<TString, TString>& TMetric::GetHints() const {
    return Hints;
}

void TMetric::AddHint(const TString& key, const TString& value) {
    Hints[key] = value;
}

bool TMetric::NeedTarget() const {
    return GetErrorType() != EErrorType::PairwiseError;
}

static inline TConstArrayRef<double> GetRowRef(const TVector<TVector<double>>& matrix, size_t rowIdx) {
    if (matrix.empty()) {
        return TArrayRef<double>();
    } else {
        return MakeArrayRef(matrix[rowIdx]);
    }
}

static constexpr ui32 EncodeFlags(bool flagOne, bool flagTwo, bool flagThree = false, bool flagFour = false) {
    return flagOne + flagTwo * 2 + flagThree * 4 + flagFour * 8;
}

/* CrossEntropy */

namespace {
    struct TCrossEntropyMetric: public TAdditiveMetric<TCrossEntropyMetric> {
        explicit TCrossEntropyMetric(ELossFunction lossFunction, double border);
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        ELossFunction LossFunction;
        double Border;
    };
}

THolder<IMetric> MakeCrossEntropyMetric(ELossFunction lossFunction, double border) {
    return MakeHolder<TCrossEntropyMetric>(lossFunction, border);
}

TCrossEntropyMetric::TCrossEntropyMetric(ELossFunction lossFunction, double border)
        : LossFunction(lossFunction)
        , Border(border)
{
    Y_ASSERT(lossFunction == ELossFunction::Logloss || lossFunction == ELossFunction::CrossEntropy);
    if (lossFunction == ELossFunction::CrossEntropy) {
        CB_ENSURE(border == GetDefaultClassificationBorder(), "Border is meaningless for crossEntropy metric");
    }
}

TMetricHolder TCrossEntropyMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    // p * log(1/(1+exp(-f))) + (1-p) * log(1 - 1/(1+exp(-f))) =
    // p * log(exp(f) / (exp(f) + 1)) + (1-p) * log(exp(-f)/(1+exp(-f))) =
    // p * log(exp(f) / (exp(f) + 1)) + (1-p) * log(1/(exp(f) + 1)) =
    // p * (log(val) - log(val + 1)) + (1-p) * (-log(val + 1)) =
    // p*log(val) - p*log(val+1) - log(val+1) + p*log(val+1) =
    // p*log(val) - log(val+1)

    CB_ENSURE(approx.size() == 1, "Metric logloss supports only single-dimensional data");
    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight, auto isLogloss, float border, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder holder(2);
        for (int i : xrange(begin, end)) {
            const float w = hasWeight ? weight[i] : 1;
            const float prob = isLogloss ? target[i] > border : target[i];
            if (isExpApprox) {
                double expApprox = approx[i];
                double nonExpApprox = FastLogf(expApprox);
                if (hasDelta) {
                    expApprox *= approxDelta[i];
                    nonExpApprox += FastLogf(approxDelta[i]);
                }
                holder.Stats[0] += w * ((1 - prob) * nonExpApprox + FastLogf(1 + 1 / expApprox));
            } else {
                double nonExpApprox = approx[i];
                if (hasDelta) {
                    nonExpApprox += approxDelta[i];
                }
                const double expApprox = exp(nonExpApprox);
                holder.Stats[0] += w * ((1 - prob) * nonExpApprox + log(1 + 1 / expApprox));
            }
            holder.Stats[1] += w;
        }
        return holder;
    };
    switch (EncodeFlags(isExpApprox, !approxDelta.empty(), !weight.empty(), LossFunction == ELossFunction::Logloss)) {
        case EncodeFlags(false, false, false, false):
            return impl(std::false_type(), std::false_type(), std::false_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, false, false, true):
            return impl(std::false_type(), std::false_type(), std::false_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, false, true, false):
            return impl(std::false_type(), std::false_type(), std::true_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, false, true, true):
            return impl(std::false_type(), std::false_type(), std::true_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, false, false):
            return impl(std::false_type(), std::true_type(), std::false_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, false, true):
            return impl(std::false_type(), std::true_type(), std::false_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, true, false):
            return impl(std::false_type(), std::true_type(), std::true_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, true, true):
            return impl(std::false_type(), std::true_type(), std::true_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, false, false):
            return impl(std::true_type(), std::false_type(), std::false_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, false, true):
            return impl(std::true_type(), std::false_type(), std::false_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, true, false):
            return impl(std::true_type(), std::false_type(), std::true_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, true, true):
            return impl(std::true_type(), std::false_type(), std::true_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, false, false):
            return impl(std::true_type(), std::true_type(), std::false_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, false, true):
            return impl(std::true_type(), std::true_type(), std::false_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, true, false):
            return impl(std::true_type(), std::true_type(), std::true_type(), std::false_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, true, true):
            return impl(std::true_type(), std::true_type(), std::true_type(), std::true_type(), Border, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

TString TCrossEntropyMetric::GetDescription() const {
    return BuildDescription(LossFunction, UseWeights, "%.3g", MakeBorderParam(Border));

}

void TCrossEntropyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* CtrFactor */

namespace {
    class TCtrFactorMetric : public TAdditiveMetric<TCtrFactorMetric> {
    public:
        explicit TCtrFactorMetric(double border = GetDefaultClassificationBorder())
            : Border(border) {
        }
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        double Border;
    };
}

THolder<IMetric> MakeCtrFactorMetric(double border) {
    return MakeHolder<TCtrFactorMetric>(border);
}

TMetricHolder TCtrFactorMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric CtrFactor supports only single-dimensional data");
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder holder(2);
    const double* approxPtr = approxVec.data();
    const float* targetPtr = target.data();
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        const float targetVal = targetPtr[i] > Border;
        holder.Stats[0] += w * targetVal;

        const double p = OverflowSafeLogitProb(approxPtr[i]);
        holder.Stats[1] += w * p;
    }
    return holder;
}

TString TCtrFactorMetric::GetDescription() const {
    return BuildDescription(ELossFunction::CtrFactor, UseWeights);
}

void TCtrFactorMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::FixedValue;
    *bestValue = 1;
}

/* RMSE */

namespace {
    struct TRMSEMetric: public TAdditiveMetric<TRMSEMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakeRMSEMetric() {
    return MakeHolder<TRMSEMetric>();
}

TMetricHolder TRMSEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric RMSE supports only single-dimensional data");
    Y_ASSERT(!isExpApprox);

    const auto impl = [=] (auto hasDelta, auto hasWeight, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        for (int k : xrange(begin, end)) {
            double targetMismatch = approx[k] - target[k];
            if (hasDelta) {
                targetMismatch += approxDelta[k];
            }
            const float w = hasWeight ? weight[k] : 1;
            error.Stats[0] += Sqr(targetMismatch) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    switch (EncodeFlags(!approxDelta.empty(), !weight.empty())) {
        case EncodeFlags(false, false):
            return impl(std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true):
            return impl(std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false):
            return impl(std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true):
            return impl(std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

double TRMSEMetric::GetFinalError(const TMetricHolder& error) const {
    return sqrt(error.Stats[0] / (error.Stats[1] + 1e-38));
}

TString TRMSEMetric::GetDescription() const {
    return BuildDescription(ELossFunction::RMSE, UseWeights);
}

void TRMSEMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Lq */

namespace {
    struct TLqMetric: public TAdditiveMetric<TLqMetric> {
        explicit TLqMetric(double q)
            : Q(q) {
            CB_ENSURE(Q >= 1, "Lq metric is defined for q >= 1, got " << q);
        }

        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        double Q;
    };
}

THolder<IMetric> MakeLqMetric(double q) {
    return MakeHolder<TLqMetric>(q);
}

TMetricHolder TLqMetric::EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric Lq supports only single-dimensional data");
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        for (int k : xrange(begin, end)) {
            double targetMismatch = approx[k] - target[k];
            if (hasDelta) {
                targetMismatch += approxDelta[k];
            }
            const float w = hasWeight ? weight[k] : 1;
            error.Stats[0] += std::pow(Abs(targetMismatch), Q) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    switch (EncodeFlags(!approxDelta.empty(), !weight.empty())) {
        case EncodeFlags(false, false):
            return impl(std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true):
            return impl(std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false):
            return impl(std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true):
            return impl(std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}


TString TLqMetric::GetDescription() const {
    TMetricParam<double> q("q", Q, true);
    return BuildDescription(ELossFunction::Lq, UseWeights, "%.3g", q);
}

void TLqMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Quantile */

namespace {
    class TQuantileMetric : public TAdditiveMetric<TQuantileMetric> {
    public:
        explicit TQuantileMetric(ELossFunction lossFunction, double alpha);
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        ELossFunction LossFunction;
        double Alpha;
    };
}

THolder<IMetric> MakeQuantileMetric(ELossFunction lossFunction, double alpha) {
    return MakeHolder<TQuantileMetric>(lossFunction, alpha);
}

TQuantileMetric::TQuantileMetric(ELossFunction lossFunction, double alpha)
        : LossFunction(lossFunction)
        , Alpha(alpha)
{
    Y_ASSERT(lossFunction == ELossFunction::Quantile || lossFunction == ELossFunction::MAE);
    CB_ENSURE(lossFunction == ELossFunction::Quantile || alpha == 0.5, "Alpha parameter should not be used for MAE loss");
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6, "Alpha parameter for quantile metric should be in interval [0, 1]");
}

TMetricHolder TQuantileMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric quantile supports only single-dimensional data");
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight, double alpha, bool isMAE, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        for (int i : xrange(begin, end)) {
            double val = target[i] - approx[i];
            if (hasDelta) {
                val -= approxDelta[i];
            }
            const double multiplier = (val > 0) ? alpha : -(1 - alpha);
            const float w = hasWeight ? weight[i] : 1;
            error.Stats[0] += (multiplier * val) * w;
            error.Stats[1] += w;
        }
        if (isMAE) {
            error.Stats[0] *= 2;
        }
        return error;
    };
    switch (EncodeFlags(!approxDelta.empty(), !weight.empty())) {
        case EncodeFlags(false, false):
            return impl(std::false_type(), std::false_type(), Alpha, LossFunction == ELossFunction::MAE, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true):
            return impl(std::false_type(), std::true_type(), Alpha, LossFunction == ELossFunction::MAE, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false):
            return impl(std::true_type(), std::false_type(), Alpha, LossFunction == ELossFunction::MAE, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true):
            return impl(std::true_type(), std::true_type(), Alpha, LossFunction == ELossFunction::MAE, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

TString TQuantileMetric::GetDescription() const {
    if (LossFunction == ELossFunction::Quantile) {
        const TMetricParam<double> alpha("alpha", Alpha, /*userDefined*/true);
        return BuildDescription(LossFunction, UseWeights, "%.3g", alpha);
    } else {
        return BuildDescription(LossFunction, UseWeights);
    }
}

void TQuantileMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* LogLinQuantile */

namespace {
    class TLogLinQuantileMetric : public TAdditiveMetric<TLogLinQuantileMetric> {
    public:
        explicit TLogLinQuantileMetric(double alpha);
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        double Alpha;
    };
}

THolder<IMetric> MakeLogLinQuantileMetric(double alpha) {
    return MakeHolder<TLogLinQuantileMetric>(alpha);
}

TLogLinQuantileMetric::TLogLinQuantileMetric(double alpha)
        : Alpha(alpha)
{
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6, "Alpha parameter for log-linear quantile metric should be in interval (0, 1)");
}

TMetricHolder TLogLinQuantileMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric log-linear quantile supports only single-dimensional data");
    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight, double alpha, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        for (int i : xrange(begin, end)) {
            double expApprox = approx[i];
            if (isExpApprox) {
                if (hasDelta) {
                    expApprox *= approxDelta[i];
                }
            } else {
                if (hasDelta) {
                    expApprox += approxDelta[i];
                }
                FastExpInplace(&expApprox, 1);
            }
            const double val = target[i] - expApprox;
            const double multiplier = (val > 0) ? alpha : -(1 - alpha);
            const float w = hasWeight ? weight[i] : 1;
            error.Stats[0] += (multiplier * val) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    switch (EncodeFlags(isExpApprox, !approxDelta.empty(), !weight.empty())) {
        case EncodeFlags(false, false, false):
            return impl(std::false_type(), std::false_type(), std::false_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, false, true):
            return impl(std::false_type(), std::false_type(), std::true_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, false):
            return impl(std::false_type(), std::true_type(), std::false_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, true):
            return impl(std::false_type(), std::true_type(), std::true_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, false):
            return impl(std::true_type(), std::false_type(), std::false_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, true):
            return impl(std::true_type(), std::false_type(), std::true_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, false):
            return impl(std::true_type(), std::true_type(), std::false_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, true):
            return impl(std::true_type(), std::true_type(), std::true_type(), Alpha, approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

TString TLogLinQuantileMetric::GetDescription() const {
    const TMetricParam<double> alpha("alpha", Alpha, /*userDefined*/true);
    return BuildDescription(ELossFunction::LogLinQuantile, UseWeights, "%.3g", alpha);
}

void TLogLinQuantileMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* MAPE */

namespace {
    struct TMAPEMetric : public TAdditiveMetric<TMAPEMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakeMAPEMetric() {
    return MakeHolder<TMAPEMetric>();
}

TMetricHolder TMAPEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric MAPE quantile supports only single-dimensional data");
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        for (int k : xrange(begin, end)) {
            const float w = hasWeight ? weight[k] : 1;
            const double delta = hasDelta ? approxDelta[k] : 0;
            error.Stats[0] += Abs(1 - (approx[k] + delta) / target[k]) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    switch (EncodeFlags(!approxDelta.empty(), !weight.empty())) {
        case EncodeFlags(false, false):
            return impl(std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true):
            return impl(std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false):
            return impl(std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true):
            return impl(std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

TString TMAPEMetric::GetDescription() const {
    return BuildDescription(ELossFunction::MAPE, UseWeights);
}

void TMAPEMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Greater K */

namespace {
    struct TNumErrorsMetric: public TAdditiveMetric<TNumErrorsMetric> {
        explicit TNumErrorsMetric(double k)
            : GreaterThan(k) {
            CB_ENSURE(k > 0, "Error: NumErrors metric requires num_erros > 0 parameter, got " << k);
        }

        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        double GreaterThan;
    };
}

THolder<IMetric> MakeNumErrorsMetric(double k) {
    return MakeHolder<TNumErrorsMetric>(k);
}

TMetricHolder TNumErrorsMetric::EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric NumErrors supports only single-dimensional data");
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Stats[0] += (Abs(approxVec[k] - target[k]) > GreaterThan ? 1 : 0) * w;
        error.Stats[1] += w;
    }

    return error;
}

TString TNumErrorsMetric::GetDescription() const {
    TMetricParam<double> k("greater_than", GreaterThan, true);
    return BuildDescription(ELossFunction::NumErrors, UseWeights, "%.3g", k);
}

void TNumErrorsMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Poisson */

namespace {
    struct TPoissonMetric : public TAdditiveMetric<TPoissonMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakePoissonMetric() {
    return MakeHolder<TPoissonMetric>();
}

TMetricHolder TPoissonMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    // Error function:
    // Sum_d[approx(d) - target(d) * log(approx(d))]
    // approx(d) == exp(Sum(tree_value))

    Y_ASSERT(approx.size() == 1);
    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        for (int i : xrange(begin, end)) {
            double expApprox = approx[i], nonExpApprox;
            if (isExpApprox) {
                if (hasDelta) {
                    expApprox *= approxDelta[i];
                }
                nonExpApprox = FastLogf(expApprox);
            } else {
                if (hasDelta) {
                    expApprox += approxDelta[i];
                }
                nonExpApprox = expApprox;
                FastExpInplace(&expApprox, 1);
            }
            const float w = hasWeight ? weight[i] : 1;
            error.Stats[0] += (expApprox - target[i] * nonExpApprox) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    switch (EncodeFlags(isExpApprox, !approxDelta.empty(), !weight.empty())) {
        case EncodeFlags(false, false, false):
            return impl(std::false_type(), std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, false, true):
            return impl(std::false_type(), std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, false):
            return impl(std::false_type(), std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, true):
            return impl(std::false_type(), std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, false):
            return impl(std::true_type(), std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, true):
            return impl(std::true_type(), std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, false):
            return impl(std::true_type(), std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, true):
            return impl(std::true_type(), std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

TString TPoissonMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Poisson, UseWeights);
}

void TPoissonMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Mean squared logarithmic error */

namespace {
    struct TMSLEMetric : public TAdditiveMetric<TMSLEMetric> {
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakeMSLEMetric() {
    return MakeHolder<TMSLEMetric>();
}

TMetricHolder TMSLEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric Mean squared logarithmic error supports only single-dimensional data");
    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(2);
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        error.Stats[0] += Sqr(log(1 + approxVec[i]) - log(1 + target[i])) * w;
        error.Stats[1] += w;
    }

    return error;
}

double TMSLEMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[0] / (error.Stats[1] + 1e-38);
}

TString TMSLEMetric::GetDescription() const {
    return BuildDescription(ELossFunction::MSLE, UseWeights);
}

void TMSLEMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Median absolute error */

namespace {
    struct TMedianAbsoluteErrorMetric : public TNonAdditiveMetric {
        TMetricHolder Eval(
                const TVector<TVector<double>>& approx,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end,
                NPar::TLocalExecutor& executor) const override {
                    return Eval(approx, /*approxDelta*/{}, /*isExpApprox*/false, target, weight, queriesInfo, begin, end, executor);
                }
        TMetricHolder Eval(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end,
                NPar::TLocalExecutor& executor) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        TMedianAbsoluteErrorMetric() {
            UseWeights.MakeIgnored();
        }
    };
}

THolder<IMetric> MakeMedianAbsoluteErrorMetric() {
    return MakeHolder<TMedianAbsoluteErrorMetric>();
}

TMetricHolder TMedianAbsoluteErrorMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /*executor*/
) const {
    CB_ENSURE(approx.size() == 1, "Metric Median absolute error supports only single-dimensional data");
    Y_ASSERT(!isExpApprox);
    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    TVector<double> values;
    values.reserve(end - begin);
    if (approxDelta.empty()) {
        for (int i = begin; i < end; ++i) {
            values.push_back(fabs(approxVec[i] - target[i]));
        }
    } else {
        for (int i = begin; i < end; ++i) {
            values.push_back(fabs(approxVec[i] + approxDelta[0][i] - target[i]));
        }
    }
    int median = (end - begin) / 2;
    PartialSort(values.begin(), values.begin() + median + 1, values.end());
    if (target.size() % 2 == 0) {
        error.Stats[0] = (values[median - 1] + values[median]) / 2;
    } else {
        error.Stats[0] = values[median];
    }
    error.Stats[1] = 1;

    return error;
}

TString TMedianAbsoluteErrorMetric::GetDescription() const {
    return ToString(ELossFunction::MedianAbsoluteError);
}

void TMedianAbsoluteErrorMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Symmetric mean absolute percentage error */

namespace {
    struct TSMAPEMetric : public TAdditiveMetric<TSMAPEMetric> {
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakeSMAPEMetric() {
    return MakeHolder<TSMAPEMetric>();
}

TMetricHolder TSMAPEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Symmetric mean absolute percentage error supports only single-dimensional data");
    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(2);
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        double denominator = (fabs(approxVec[i]) + fabs(target[i]));
        error.Stats[0] += denominator == 0 ? 0 : 200 * w * fabs(target[i] - approxVec[i]) / denominator;
        error.Stats[1] += w;
    }

    return error;
}

double TSMAPEMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[0] / (error.Stats[1] + 1e-38);
}

TString TSMAPEMetric::GetDescription() const {
    return BuildDescription(ELossFunction::SMAPE, UseWeights);
}

void TSMAPEMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* loglikelihood of prediction */

namespace {
    struct TLLPMetric : public TAdditiveMetric<TLLPMetric> {
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        TVector<TString> GetStatDescriptions() const override;
    };
}

THolder<IMetric> MakeLLPMetric() {
    return MakeHolder<TLLPMetric>();
}

TMetricHolder TLLPMetric::EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return CalcLlp(approx.front(), target, weight, begin, end);
}

double TLLPMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcLlp(error);
}

TString TLLPMetric::GetDescription() const {
    return BuildDescription(ELossFunction::LogLikelihoodOfPrediction, UseWeights);
}

void TLLPMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

TVector<TString> TLLPMetric::GetStatDescriptions() const {
    return {"intermediate result", "clicks", "shows"};
}

/* MultiClass */

namespace {
    struct TMultiClassMetric : public TAdditiveMetric<TMultiClassMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakeMultiClassMetric() {
    return MakeHolder<TMultiClassMetric>();
}

static void GetMultiDimensionalApprox(int idx, const TVector<TVector<double>>& approx, const TVector<TVector<double>>& approxDelta, TArrayRef<double> evaluatedApprox) {
    const auto approxDimension = approx.size();
    Y_ASSERT(approxDimension == evaluatedApprox.size());
    if (!approxDelta.empty()) {
        for (auto dimensionIdx : xrange(approxDimension)) {
            evaluatedApprox[dimensionIdx] = approx[dimensionIdx][idx] + approxDelta[dimensionIdx][idx];
        }
    } else {
        for (auto dimensionIdx : xrange(approxDimension)) {
            evaluatedApprox[dimensionIdx] = approx[dimensionIdx][idx];
        }
    }
}

TMetricHolder TMultiClassMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    // Y_ASSERT(target.size() == approx[0].size());
    const int approxDimension = approx.ysize();
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(2);

    TVector<double> evaluatedApprox(approxDimension);
    for (int k = begin; k < end; ++k) {
        GetMultiDimensionalApprox(k, approx, approxDelta, evaluatedApprox);

        double maxApprox = std::numeric_limits<double>::min();
        for (int dim = 1; dim < approxDimension; ++dim) {
            if (evaluatedApprox[dim] > maxApprox) {
                maxApprox = evaluatedApprox[dim];
            }
        }

        double sumExpApprox = 0;
        for (int dim = 0; dim < approxDimension; ++dim) {
            sumExpApprox += exp(evaluatedApprox[dim] - maxApprox);
        }

        const int targetClass = static_cast<int>(target[k]);
        Y_ASSERT(targetClass >= 0 && targetClass < approxDimension);
        const double targetClassApprox = evaluatedApprox[targetClass];

        const float w = weight.empty() ? 1 : weight[k];
        error.Stats[0] += (targetClassApprox - maxApprox - log(sumExpApprox)) * w;
        error.Stats[1] += w;
    }

    return error;
}

TString TMultiClassMetric::GetDescription() const {
    return BuildDescription(ELossFunction::MultiClass, UseWeights);
}

void TMultiClassMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* MultiClassOneVsAll */

namespace {
    struct TMultiClassOneVsAllMetric : public TAdditiveMetric<TMultiClassOneVsAllMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakeMultiClassOneVsAllMetric() {
    return MakeHolder<TMultiClassOneVsAllMetric>();
}

TMetricHolder TMultiClassOneVsAllMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    const int approxDimension = approx.ysize();
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(2);
    TVector<double> evaluatedApprox(approxDimension);
    for (int k = begin; k < end; ++k) {
        GetMultiDimensionalApprox(k, approx, approxDelta, evaluatedApprox);
        double sumDimErrors = 0;
        for (int dim = 0; dim < approxDimension; ++dim) {
            const double expApprox = exp(evaluatedApprox[dim]);
            sumDimErrors += IsFinite(expApprox) ? -log(1 + expApprox) : -evaluatedApprox[dim];
        }

        const int targetClass = static_cast<int>(target[k]);
        Y_ASSERT(targetClass >= 0 && targetClass < approxDimension);
        sumDimErrors += evaluatedApprox[targetClass];

        const float w = weight.empty() ? 1 : weight[k];
        error.Stats[0] += sumDimErrors / approxDimension * w;
        error.Stats[1] += w;
    }
    return error;
}

TString TMultiClassOneVsAllMetric::GetDescription() const {
    return BuildDescription(ELossFunction::MultiClassOneVsAll, UseWeights);
}

void TMultiClassOneVsAllMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* PairLogit */

namespace {
    struct TPairLogitMetric : public TAdditiveMetric<TPairLogitMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakePairLogitMetric() {
    return MakeHolder<TPairLogitMetric>();
}

TMetricHolder TPairLogitMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> /*target*/,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric PairLogit supports only single-dimensional data");

    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        TVector<double> approxExpShifted;
        for (int queryIndex : xrange(queryStartIndex, queryEndIndex)) {
            const int begin = queriesInfo[queryIndex].Begin;
            const int end = queriesInfo[queryIndex].End;
            const int querySize = end - begin;
            if (querySize > approxExpShifted.ysize()) {
                approxExpShifted.yresize(querySize);
            }
            for (auto idx : xrange(begin, end)) {
                if (hasDelta) {
                    if (isExpApprox) {
                        approxExpShifted[idx - begin] = approx[idx] * approxDelta[idx];
                    } else {
                        approxExpShifted[idx - begin] = approx[idx] + approxDelta[idx];
                    }
                } else {
                    approxExpShifted[idx - begin] = approx[idx];
                }
            }
            if (!isExpApprox) {
                const double maxQueryApprox = *MaxElement(approxExpShifted.begin(), approxExpShifted.begin() + querySize);
                for (double& approx : approxExpShifted) {
                    approx -= maxQueryApprox;
                }
                FastExpInplace(approxExpShifted.data(), querySize);
            }

            for (int docId = 0; docId < queriesInfo[queryIndex].Competitors.ysize(); ++docId) {
                for (const auto& competitor : queriesInfo[queryIndex].Competitors[docId]) {
                    const double weight = hasWeight ? competitor.Weight : 1.0;
                    error.Stats[0] += -weight * log(approxExpShifted[docId] / (approxExpShifted[docId] + approxExpShifted[competitor.Id]));
                    error.Stats[1] += weight;
                }
            }
        }
        return error;
    };
    switch (EncodeFlags(isExpApprox, !approxDelta.empty(), UseWeights.Get())) {
        case EncodeFlags(false, false, false):
            return impl(std::false_type(), std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, false, true):
            return impl(std::false_type(), std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, false):
            return impl(std::false_type(), std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true, true):
            return impl(std::false_type(), std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, false):
            return impl(std::true_type(), std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false, true):
            return impl(std::true_type(), std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, false):
            return impl(std::true_type(), std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true, true):
            return impl(std::true_type(), std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

EErrorType TPairLogitMetric::GetErrorType() const {
    return EErrorType::PairwiseError;
}

TString TPairLogitMetric::GetDescription() const {
    return BuildDescription(ELossFunction::PairLogit, UseWeights);
}

void TPairLogitMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* QueryRMSE */

namespace {
    struct TQueryRMSEMetric : public TAdditiveMetric<TQueryRMSEMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        template <bool HasDelta, bool HasWeight>
        double CalcQueryAvrg(
            int start,
            int count,
            TConstArrayRef<double> approxes,
            TConstArrayRef<double> approxDelta,
            TConstArrayRef<float> targets,
            TConstArrayRef<float> weights
        ) const;
    };
}

THolder<IMetric> MakeQueryRMSEMetric() {
    return MakeHolder<TQueryRMSEMetric>();
}

TMetricHolder TQueryRMSEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric QueryRMSE supports only single-dimensional data");
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TMetricHolder error(2);
        for (int queryIndex : xrange(queryStartIndex, queryEndIndex)) {
            const int begin = queriesInfo[queryIndex].Begin;
            const int end = queriesInfo[queryIndex].End;
            const double queryAvrg = CalcQueryAvrg<hasDelta, hasWeight>(begin, end - begin, approx, approxDelta, target, weight);
            for (int docId : xrange(begin, end)) {
                const double w = hasWeight ? weight[docId] : 1;
                const double delta = hasDelta ? approxDelta[docId] : 0;
                error.Stats[0] += Sqr(target[docId] - approx[docId] - delta - queryAvrg) * w;
                error.Stats[1] += w;
            }
        }
        return error;
    };
    switch (EncodeFlags(!approxDelta.empty(), !weight.empty())) {
        case EncodeFlags(false, false):
            return impl(std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true):
            return impl(std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false):
            return impl(std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true):
            return impl(std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

template <bool HasDelta, bool HasWeight>
double TQueryRMSEMetric::CalcQueryAvrg(
    int start,
    int count,
    TConstArrayRef<double> approxes,
    TConstArrayRef<double> approxDelta,
    TConstArrayRef<float> targets,
    TConstArrayRef<float> weights
) const {
    double qsum = 0;
    double qcount = 0;
    for (int docId : xrange(start, start + count)) {
        const double w = HasWeight ? weights[docId] : 1;
        const double delta = HasDelta ? approxDelta[docId] : 0;
        qsum += (targets[docId] - approxes[docId] - delta) * w;
        qcount += w;
    }

    double qavrg = 0;
    if (qcount > 0) {
        qavrg = qsum / qcount;
    }
    return qavrg;
}

EErrorType TQueryRMSEMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TQueryRMSEMetric::GetDescription() const {
    return BuildDescription(ELossFunction::QueryRMSE, UseWeights);
}

double TQueryRMSEMetric::GetFinalError(const TMetricHolder& error) const {
    return sqrt(error.Stats[0] / (error.Stats[1] + 1e-38));
}

void TQueryRMSEMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* PFound */

namespace {
    struct TPFoundMetric : public TAdditiveMetric<TPFoundMetric> {
        explicit TPFoundMetric(int topSize, double decay);
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int TopSize;
        double Decay;
    };
}

THolder<IMetric> MakePFoundMetric(int topSize, double decay) {
    return MakeHolder<TPFoundMetric>(topSize, decay);
}

TPFoundMetric::TPFoundMetric(int topSize, double decay)
        : TopSize(topSize)
        , Decay(decay) {
}

TMetricHolder TPFoundMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    const auto impl = [=] (auto hasDelta, auto isExpApprox, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        TPFoundCalcer calcer(TopSize, Decay);
        for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
            const int queryBegin = queriesInfo[queryIndex].Begin;
            const int queryEnd = queriesInfo[queryIndex].End;
            const ui32* subgroupIdData = nullptr;
            const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.0;
            if (!queriesInfo[queryIndex].SubgroupId.empty()) {
                subgroupIdData = queriesInfo[queryIndex].SubgroupId.data();
            }
            const double* approxDeltaData = nullptr;
            if (hasDelta) {
                approxDeltaData = approxDelta.data();
            }
            calcer.AddQuery<isExpApprox, hasDelta>(target.data() + queryBegin, approx.data() + queryBegin, approxDeltaData + queryBegin, queryWeight, subgroupIdData, queryEnd - queryBegin);
        }
        return calcer.GetMetric();
    };
    switch (EncodeFlags(!approxDelta.empty(), isExpApprox)) {
        case EncodeFlags(false, false):
            return impl(std::false_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true):
            return impl(std::false_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false):
            return impl(std::true_type(), std::false_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true):
            return impl(std::true_type(), std::true_type(), approx[0], GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }

}

EErrorType TPFoundMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TPFoundMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != -1);
    const TMetricParam<double> decay("decay", Decay, Decay != 0.85);
    return BuildDescription(ELossFunction::PFound, UseWeights, topSize, "%.3g", decay);
}

double TPFoundMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TPFoundMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* NDCG@N */

namespace {
    struct TNdcgMetric: public TAdditiveMetric<TNdcgMetric> {
        explicit TNdcgMetric(int topSize, ENdcgMetricType type);
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int TopSize;
        ENdcgMetricType MetricType;
    };
}

THolder<IMetric> MakeNdcgMetric(int topSize, ENdcgMetricType type) {
    return MakeHolder<TNdcgMetric>(topSize, type);
}

TNdcgMetric::TNdcgMetric(int topSize, ENdcgMetricType type)
    : TopSize(topSize)
    , MetricType(type) {
}

TMetricHolder TNdcgMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    TMetricHolder error(2);
    TVector<NMetrics::TSample> samples;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        const auto queryBegin = queriesInfo[queryIndex].Begin;
        const auto queryEnd = queriesInfo[queryIndex].End;
        const auto querySize = queryEnd - queryBegin;
        const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.f;
        NMetrics::TSample::FromVectors(
            MakeArrayRef(target.data() + queryBegin, querySize),
            MakeArrayRef(approx.front().data() + queryBegin, querySize),
            &samples);
        error.Stats[0] += queryWeight * CalcNdcg(samples, MetricType, TopSize);
        error.Stats[1] += queryWeight;
    }
    return error;
}

TString TNdcgMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != -1);
    const TMetricParam<ENdcgMetricType> type("type", MetricType, true);
    return BuildDescription(ELossFunction::NDCG, UseWeights, topSize, type);
}

EErrorType TNdcgMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TNdcgMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TNdcgMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}


/* QuerySoftMax */

namespace {
    struct TQuerySoftMaxMetric : public TAdditiveMetric<TQuerySoftMaxMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        TMetricHolder EvalSingleQuery(
            int start,
            int count,
            TConstArrayRef<double> approxes,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> targets,
            TConstArrayRef<float> weights,
            TArrayRef<double> softmax
        ) const;
    };
}

THolder<IMetric> MakeQuerySoftMaxMetric() {
    return MakeHolder<TQuerySoftMaxMetric>();
}

TMetricHolder TQuerySoftMaxMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric QuerySoftMax supports only single-dimensional data");

    TMetricHolder error(2);
    TVector<double> softmax;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        const int begin = queriesInfo[queryIndex].Begin;
        const int end = queriesInfo[queryIndex].End;
        if (softmax.ysize() < end - begin) {
            softmax.yresize(end - begin);
        }
        error.Add(EvalSingleQuery(begin, end - begin, approx[0], approxDelta, isExpApprox, target, weight, softmax));
    }
    return error;
}

EErrorType TQuerySoftMaxMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TMetricHolder TQuerySoftMaxMetric::EvalSingleQuery(
    int start,
    int count,
    TConstArrayRef<double> approxes,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> targets,
    TConstArrayRef<float> weights,
    TArrayRef<double> softmax
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight, TConstArrayRef<double> approx, TConstArrayRef<double> approxDelta) {
        double sumWeightedTargets = 0;
        for (int dim : xrange(count)) {
            if (targets[start + dim] > 0) {
                const double weight = hasWeight ? weights[start + dim] : 1;
                sumWeightedTargets += weight * targets[start + dim];
            }
        }
        TMetricHolder error(2);
        if (sumWeightedTargets <= 0) {
            return error;
        }
        error.Stats[1] = sumWeightedTargets;

        for (int dim : xrange(count)) {
            const double delta = hasDelta ? approxDelta[start + dim] : 0;
            softmax[dim] = approx[start + dim] + delta;
        }
        double maxApprox = -std::numeric_limits<double>::max();
        for (int dim : xrange(count)) {
            if (!hasWeight || weights[start + dim] > 0) {
                maxApprox = Max(maxApprox, softmax[dim]);
            }
        }
        for (int dim : xrange(count)) {
            softmax[dim] -= maxApprox;
        }
        FastExpInplace(softmax.data(), count);
        double sumExpApprox = 0;
        for (int dim : xrange(count)) {
            const double weight = hasWeight ? weights[start + dim] : 1;
            if (weight > 0) {
                softmax[dim] *= weight;
                sumExpApprox += softmax[dim];
            }
        }
        for (int dim : xrange(count)) {
            if (targets[start + dim] > 0) {
                const double weight = hasWeight ? weights[start + dim] : 1;
                if (weight > 0) {
                    error.Stats[0] -= weight * targets[start + dim] * log(softmax[dim] / sumExpApprox);
                }
            }
        }
        return error;
    };
    switch (EncodeFlags(!approxDelta.empty(), !weights.empty())) {
        case EncodeFlags(false, false):
            return impl(std::false_type(), std::false_type(), approxes, GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(false, true):
            return impl(std::false_type(), std::true_type(), approxes, GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, false):
            return impl(std::true_type(), std::false_type(), approxes, GetRowRef(approxDelta, /*rowIdx*/0));
        case EncodeFlags(true, true):
            return impl(std::true_type(), std::true_type(), approxes, GetRowRef(approxDelta, /*rowIdx*/0));
        default:
            Y_VERIFY(false);
    }
}

TString TQuerySoftMaxMetric::GetDescription() const {
    return BuildDescription(ELossFunction::QuerySoftMax, UseWeights);
}

void TQuerySoftMaxMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* R2 */

namespace {
    struct TR2Metric: public TAdditiveMetric<TR2Metric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakeR2Metric() {
    return MakeHolder<TR2Metric>();
}

TMetricHolder TR2Metric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric R2 supports only single-dimensional data");
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    double avrgTarget = Accumulate(approxVec.begin() + begin, approxVec.begin() + end, 0.0);
    Y_ASSERT(begin < end);
    avrgTarget /= end - begin;

    TMetricHolder error(2);  // Stats[0] == mse, Stats[1] == targetVariance

    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Stats[0] += Sqr(approxVec[k] - target[k]) * w;
        error.Stats[1] += Sqr(target[k] - avrgTarget) * w;
    }
    return error;
}

double TR2Metric::GetFinalError(const TMetricHolder& error) const {
    return 1 - error.Stats[0] / error.Stats[1];
}

TString TR2Metric::GetDescription() const {
    return BuildDescription(ELossFunction::R2, UseWeights);
}

void TR2Metric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* AUC */

namespace {
    struct TAUCMetric: public TNonAdditiveMetric {
        explicit TAUCMetric(double border = GetDefaultClassificationBorder())
                : Border(border) {
            UseWeights.SetDefaultValue(false);
        }

        explicit TAUCMetric(int positiveClass)
            : PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor) const override {
                return Eval(approx, /*approxDelta*/{}, /*isExpApprox*/false, target, weight, queriesInfo, begin, end, executor);
        }
        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int PositiveClass = 1;
        bool IsMultiClass = false;
        double Border = GetDefaultClassificationBorder();
    };
}

THolder<IMetric> MakeBinClassAucMetric(double border) {
    return MakeHolder<TAUCMetric>(border);
}

THolder<IMetric> MakeMultiClassAucMetric(int positiveClass) {
    return MakeHolder<TAUCMetric>(positiveClass);
}

TMetricHolder TAUCMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weightIn,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    const auto& approxVec = approx.ysize() == 1 ? approx.front() : approx[PositiveClass];
    Y_ASSERT(approxVec.size() == target.size());
    auto weight = UseWeights ? weightIn : TConstArrayRef<float>{};

    TVector<double> approxCopy(approxVec.begin() + begin, approxVec.begin() + end);
    if (!approxDelta.empty()) {
        for (int idx : xrange(begin, end)) {
            approxCopy[idx - begin] += approxDelta[0][idx];
        }
    }
    TVector<double> targetCopy(target.begin() + begin, target.begin() + end);

    if (!IsMultiClass) {
        for (ui32 i = 0; i < targetCopy.size(); ++i) {
            targetCopy[i] = targetCopy[i] > Border;
        }
    }

    if (approx.ysize() > 1) {
        int positiveClass = PositiveClass;
        ForEach(targetCopy.begin(), targetCopy.end(), [positiveClass](double& x) {
            x = (x == static_cast<double>(positiveClass));
        });
    }

    TVector<NMetrics::TSample> samples;
    if (weight.empty()) {
        samples = NMetrics::TSample::FromVectors(targetCopy, approxCopy);
    } else {
        TVector<double> weightCopy(weight.begin() + begin, weight.begin() + end);
        samples = NMetrics::TSample::FromVectors(targetCopy, approxCopy, weightCopy);
    }

    TMetricHolder error(2);
    error.Stats[0] = CalcAUC(&samples);
    error.Stats[1] = 1.0;
    return error;
}

TString TAUCMetric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::AUC, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::AUC, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

void TAUCMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Accuracy */

namespace {
    struct TAccuracyMetric : public TAdditiveMetric<TAccuracyMetric> {
        explicit TAccuracyMetric(double border)
            : Border(border)
        {
        }
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        double Border = GetDefaultClassificationBorder();
    };
}

THolder<IMetric> MakeAccuracyMetric(double border) {
    return MakeHolder<TAccuracyMetric>(border);
}

TMetricHolder TAccuracyMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(target.size() == approx[0].size());
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return GetAccuracy(approx, target, weight, begin, end, Border);
}

TString TAccuracyMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Accuracy, UseWeights, "%.3g", MakeBorderParam(Border));
}

void TAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Precision */

namespace {
    struct TPrecisionMetric : public TAdditiveMetric<TPrecisionMetric> {
        explicit TPrecisionMetric(double border)
                : Border(border) {
        }

        explicit TPrecisionMetric(int positiveClass)
            : PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int PositiveClass = 1;
        bool IsMultiClass = false;
        double Border = GetDefaultClassificationBorder();
    };
}

THolder<IMetric> MakeBinClassPrecisionMetric(double border) {
    return MakeHolder<TPrecisionMetric>(border);
}

THolder<IMetric> MakeMultiClassPrecisionMetric(int positiveClass) {
    CB_ENSURE(positiveClass >= 0, "Class id should not be negative");
    return MakeHolder<TPrecisionMetric>(positiveClass);
}

TMetricHolder TPrecisionMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    double targetPositive;
    TMetricHolder error(2); // Stats[0] == truePositive, Stats[1] == approxPositive
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass, Border,
                     &error.Stats[0], &targetPositive, &error.Stats[1]);

    return error;
}

TString TPrecisionMetric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::Precision, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::Precision, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

void TPrecisionMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TPrecisionMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1;
}

/* Recall */

namespace {
    struct TRecallMetric: public TAdditiveMetric<TRecallMetric> {
        explicit TRecallMetric(double border = GetDefaultClassificationBorder())
            : Border(border)
        {
        }

        explicit TRecallMetric(int positiveClass)
            : PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int PositiveClass = 1;
        bool IsMultiClass = false;
        double Border = GetDefaultClassificationBorder();
    };
}

THolder<IMetric> MakeBinClassRecallMetric(double border) {
    return MakeHolder<TRecallMetric>(border);
}

THolder<IMetric> MakeMultiClassRecallMetric(int positiveClass) {
    CB_ENSURE(positiveClass >= 0, "Class id should not be negative");
    return MakeHolder<TRecallMetric>(positiveClass);
}

TMetricHolder TRecallMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    double approxPositive;
    TMetricHolder error(2);  // Stats[0] == truePositive, Stats[1] == targetPositive
    GetPositiveStats(
            approx,
            target,
            weight,
            begin,
            end,
            PositiveClass,
            Border,
            &error.Stats[0],
            &error.Stats[1],
            &approxPositive
    );

    return error;
}

TString TRecallMetric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::Recall, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::Recall, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

double TRecallMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1;
}

void TRecallMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Balanced Accuracy */

namespace {
    struct TBalancedAccuracyMetric: public TAdditiveMetric<TBalancedAccuracyMetric> {
        explicit TBalancedAccuracyMetric(double border)
                : Border(border)
        {
        }
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int PositiveClass = 1;
        double Border = GetDefaultClassificationBorder();
    };
}

THolder<IMetric> MakeBinClassBalancedAccuracyMetric(double border) {
    return MakeHolder<TBalancedAccuracyMetric>(border);
}

TMetricHolder TBalancedAccuracyMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return CalcBalancedAccuracyMetric(approx, target, weight, begin, end, PositiveClass, Border);
}

TString TBalancedAccuracyMetric::GetDescription() const {
    return BuildDescription(ELossFunction::BalancedAccuracy, UseWeights, "%.3g", MakeBorderParam(Border));
}

void TBalancedAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TBalancedAccuracyMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcBalancedAccuracyMetric(error);
}

/* Balanced Error Rate */

namespace {
    struct TBalancedErrorRate: public TAdditiveMetric<TBalancedErrorRate> {
        explicit TBalancedErrorRate(double border)
                : Border(border)
        {
        }

        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int PositiveClass = 1;
        double Border = GetDefaultClassificationBorder();
    };
}

THolder<IMetric> MakeBinClassBalancedErrorRate(double border) {
    return MakeHolder<TBalancedErrorRate>(border);
}

TMetricHolder TBalancedErrorRate::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return CalcBalancedAccuracyMetric(approx, target, weight, begin, end, PositiveClass, Border);
}

TString TBalancedErrorRate::GetDescription() const {
    return BuildDescription(ELossFunction::BalancedErrorRate, UseWeights, "%.3g", MakeBorderParam(Border));
}

void TBalancedErrorRate::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

double TBalancedErrorRate::GetFinalError(const TMetricHolder& error) const {
    return 1 - CalcBalancedAccuracyMetric(error);
}

/* Kappa */

namespace {
    struct TKappaMetric: public TAdditiveMetric<TKappaMetric> {
        explicit TKappaMetric(int classCount = 2, double border = GetDefaultClassificationBorder())
            : Border(border)
            , ClassCount(classCount) {
            UseWeights.MakeIgnored();
        }
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        double Border = GetDefaultClassificationBorder();
        int ClassCount = 2;
    };
}

THolder<IMetric> MakeBinClassKappaMetric(double border) {
    return MakeHolder<TKappaMetric>(2, border);
}

THolder<IMetric> MakeMultiClassKappaMetric(int classCount) {
    return MakeHolder<TKappaMetric>(classCount);
}

TMetricHolder TKappaMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return CalcKappaMatrix(approx, target, begin, end, Border);
}

TString TKappaMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Kappa, "%.3g", MakeBorderParam(Border));
}

void TKappaMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TKappaMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Cohen);
}

/* WKappa */

namespace {
    struct TWKappaMatric: public TAdditiveMetric<TWKappaMatric> {
        explicit TWKappaMatric(int classCount = 2, double border = GetDefaultClassificationBorder())
            : Border(border)
            , ClassCount(classCount) {
            UseWeights.MakeIgnored();
        }

        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;

        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue *valueType, float *bestValue) const override;

    private:
        double Border = GetDefaultClassificationBorder();
        int ClassCount;
    };
}

THolder<IMetric> MakeBinClassWKappaMetric(double border) {
    return MakeHolder<TWKappaMatric>(2, border);
}

THolder<IMetric> MakeMultiClassWKappaMetric(int classCount) {
    return MakeHolder<TWKappaMatric>(classCount);
}

TMetricHolder TWKappaMatric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return CalcKappaMatrix(approx, target, begin, end, Border);
}

TString TWKappaMatric::GetDescription() const {
    return BuildDescription(ELossFunction::WKappa, "%.3g", MakeBorderParam(Border));
}

void TWKappaMatric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TWKappaMatric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Weighted);
}

/* F1 */

namespace {
    struct TF1Metric: public TAdditiveMetric<TF1Metric> {
        explicit TF1Metric(double border)
            : Border(border) {
        }

        explicit TF1Metric(int positiveClass)
            : PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        TVector<TString> GetStatDescriptions() const override;

    private:
        int PositiveClass = 1;
        bool IsMultiClass = false;
        double Border = GetDefaultClassificationBorder();
    };
}

THolder<IMetric> MakeBinClassF1Metric(double border) {
    return MakeHolder<TF1Metric>(border);
}

THolder<IMetric> MakeMultiClassF1Metric(int positiveClass) {
    CB_ENSURE(positiveClass >= 0, "Class id should not be negative");
    return MakeHolder<TF1Metric>(positiveClass);
}

TMetricHolder TF1Metric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(3); // Stats[0] == truePositive; Stats[1] == targetPositive; Stats[2] == approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass, Border,
                     &error.Stats[0], &error.Stats[1], &error.Stats[2]);

    return error;
}

double TF1Metric::GetFinalError(const TMetricHolder& error) const {
    double denominator = error.Stats[1] + error.Stats[2];
    return denominator > 0 ? 2 * error.Stats[0] / denominator : 0;
}

TVector<TString> TF1Metric::GetStatDescriptions() const {
    return {"TP", "TP+FN", "TP+FP"};
}

TString TF1Metric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::F1, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::F1, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

void TF1Metric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* TotalF1 */

namespace {
    struct TTotalF1Metric : public TAdditiveMetric<TTotalF1Metric> {
        explicit TTotalF1Metric(int classesCount)
            : ClassCount(classesCount) {
        }
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TVector<TString> GetStatDescriptions() const override;

    private:
        int ClassCount;
    };
}

THolder<IMetric> MakeTotalF1Metric(int classesCount) {
    return MakeHolder<TTotalF1Metric>(classesCount);
}

TMetricHolder TTotalF1Metric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    TVector<double> truePositive;
    TVector<double> targetPositive;
    TVector<double> approxPositive;
    GetTotalPositiveStats(approx, target, weight, begin, end,
                          &truePositive, &targetPositive, &approxPositive);

    int classesCount = truePositive.ysize();
    Y_VERIFY(classesCount == ClassCount);
    TMetricHolder error(3 * classesCount); // targetPositive[0], approxPositive[0], truePositive[0], targetPositive[1], ...

    for (int classIdx = 0; classIdx < classesCount; ++classIdx) {
        error.Stats[3 * classIdx] = targetPositive[classIdx];
        error.Stats[3 * classIdx + 1] = approxPositive[classIdx];
        error.Stats[3 * classIdx + 2] = truePositive[classIdx];
    }

    return error;
}

TString TTotalF1Metric::GetDescription() const {
    return BuildDescription(ELossFunction::TotalF1, UseWeights);
}

void TTotalF1Metric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TTotalF1Metric::GetFinalError(const TMetricHolder& error) const {
    double numerator = 0;
    double denom = 0;
    for (int classIdx = 0; classIdx < ClassCount; ++classIdx) {
        double denominator = error.Stats[3 * classIdx] + error.Stats[3 * classIdx + 1];
        numerator += denominator > 0 ? 2 * error.Stats[3 * classIdx + 2] / denominator * error.Stats[3 * classIdx] : 0;
        denom += error.Stats[3 * classIdx];
    }
    return numerator / (denom + 1e-38);
}

TVector<TString> TTotalF1Metric::GetStatDescriptions() const {
    TVector<TString> result;
    for (int classIdx = 0; classIdx < ClassCount; ++classIdx) {
        auto prefix = "Class=" + ToString(classIdx) + ",";
        result.push_back(prefix + "TP+FN");
        result.push_back(prefix + "TP+FP");
        result.push_back(prefix + "TP");
    }
    return result;
}

/* Confusion matrix */

static double& GetValue(TVector<double>& squareMatrix, int i, int j) {
    int columns = sqrt(squareMatrix.size());
    Y_ASSERT(columns * columns == squareMatrix.ysize());
    return squareMatrix[i * columns + j];
}

static double GetConstValue(const TVector<double>& squareMatrix, int i, int j) {
    int columns = sqrt(squareMatrix.size());
    Y_ASSERT(columns * columns == squareMatrix.ysize());
    return squareMatrix[i * columns + j];
}

static void BuildConfusionMatrix(
    const TVector<TVector<double>>& approx,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    TVector<double>* confusionMatrix
) {
    int classesCount = approx.ysize() == 1 ? 2 : approx.ysize();
    confusionMatrix->clear();
    confusionMatrix->resize(classesCount * classesCount);
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        int targetClass = static_cast<int>(target[i]);
        Y_ASSERT(targetClass >= 0 && targetClass < classesCount);
        float w = weight.empty() ? 1 : weight[i];
        GetValue(*confusionMatrix, approxClass, targetClass) += w;
    }
}


/* MCC */

namespace {
    struct TMCCMetric : public TAdditiveMetric<TMCCMetric> {
        explicit TMCCMetric(int classesCount)
            : ClassesCount(classesCount) {
        }
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        TVector<TString> GetStatDescriptions() const override;

    private:
        int ClassesCount;
    };
}

THolder<IMetric> MakeMCCMetric(int classesCount) {
    return MakeHolder<TMCCMetric>(classesCount);
}

TMetricHolder TMCCMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    TMetricHolder holder;
    BuildConfusionMatrix(approx, target, weight, begin, end, &holder.Stats);
    return holder;
}

double TMCCMetric::GetFinalError(const TMetricHolder& error) const {
    TVector<double> rowSum(ClassesCount, 0);
    TVector<double> columnSum(ClassesCount, 0);
    double totalSum = 0;
    for (int approxClass = 0; approxClass < ClassesCount; ++approxClass) {
        for (int tragetClass = 0; tragetClass < ClassesCount; ++tragetClass) {
            rowSum[approxClass] += GetConstValue(error.Stats, approxClass, tragetClass);
            columnSum[tragetClass] += GetConstValue(error.Stats, approxClass, tragetClass);
            totalSum += GetConstValue(error.Stats, approxClass, tragetClass);
        }
    }

    double numerator = 0;
    for (int classIdx = 0; classIdx < ClassesCount; ++classIdx) {
        numerator += GetConstValue(error.Stats, classIdx, classIdx) * totalSum - rowSum[classIdx] * columnSum[classIdx];
    }

    double sumSquareRowSums = 0;
    double sumSquareColumnSums = 0;
    for (int classIdx = 0; classIdx < ClassesCount; ++classIdx) {
        sumSquareRowSums += Sqr(rowSum[classIdx]);
        sumSquareColumnSums += Sqr(columnSum[classIdx]);
    }

    double denominator = sqrt((Sqr(totalSum) - sumSquareRowSums) * (Sqr(totalSum) - sumSquareColumnSums));
    return numerator / (denominator + FLT_EPSILON);
}

TString TMCCMetric::GetDescription() const {
    return BuildDescription(ELossFunction::MCC, UseWeights);
}

void TMCCMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Brier Score */

namespace {
    struct TBrierScoreMetric : public TAdditiveMetric<TBrierScoreMetric> {
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TBrierScoreMetric() {
            UseWeights.MakeIgnored();
        }
    };
}

THolder<IMetric> MakeBrierScoreMetric() {
    return MakeHolder<TBrierScoreMetric>();
}

TMetricHolder TBrierScoreMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return ComputeBrierScoreMetric(approx.front(), target, weight, begin, end);
}

TString TBrierScoreMetric::GetDescription() const {
    return ToString(ELossFunction::BrierScore);
}

void TBrierScoreMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

double TBrierScoreMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

/* Hinge loss */

namespace {
    struct THingeLossMetric : public TAdditiveMetric<THingeLossMetric> {
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
    };
}

THolder<IMetric> MakeHingeLossMetric() {
    return MakeHolder<THingeLossMetric>();
}

TMetricHolder THingeLossMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return ComputeHingeLossMetric(approx, target, weight, begin, end);
}

TString THingeLossMetric::GetDescription() const {
    return BuildDescription(ELossFunction::HingeLoss, UseWeights);
}

void THingeLossMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

double THingeLossMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

/* Zero one loss */

namespace {
    struct TZeroOneLossMetric : public TAdditiveMetric<TZeroOneLossMetric> {
        explicit TZeroOneLossMetric(double border, bool isMultiClass);
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
    private:
        double Border = GetDefaultClassificationBorder();
        bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeZeroOneLossMetric(double border, bool isMultiClass) {
    return MakeHolder<TZeroOneLossMetric>(border, isMultiClass);
}

TZeroOneLossMetric::TZeroOneLossMetric(double border, bool isMultiClass)
        : Border(border)
        , IsMultiClass(isMultiClass)
{
}

TMetricHolder TZeroOneLossMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(target.size() == approx[0].size());
    return GetAccuracy(approx, target, weight, begin, end, Border);
}

TString TZeroOneLossMetric::GetDescription() const {
    if (IsMultiClass) {
        return BuildDescription(ELossFunction::ZeroOneLoss, UseWeights, "%.3g", MakeBorderParam(Border));
    } else {
        return BuildDescription(ELossFunction::ZeroOneLoss, UseWeights);
    }
}

void TZeroOneLossMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

double TZeroOneLossMetric::GetFinalError(const TMetricHolder& error) const {
    return 1 - error.Stats[0] / error.Stats[1];
}

/* Hamming loss */

namespace {
    struct THammingLossMetric : public TAdditiveMetric<THammingLossMetric> {
        explicit THammingLossMetric(double border, bool isMultiClass);
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;

    private:
        double Border = GetDefaultClassificationBorder();
        bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeHammingLossMetric(double border, bool isMulticlass) {
    return MakeHolder<THammingLossMetric>(border, isMulticlass);
}

THammingLossMetric::THammingLossMetric(double border, bool isMultiClass)
        : Border(border)
        , IsMultiClass(isMultiClass)
{
}

TMetricHolder THammingLossMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(target.size() == approx[0].size());
    TMetricHolder error(2);
    const bool isMulticlass = approx.size() > 1;

    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        const float targetVal = isMulticlass ? target[i] : target[i] > Border;
        int targetClass = static_cast<int>(targetVal);

        float w = weight.empty() ? 1 : weight[i];
        error.Stats[0] += approxClass != targetClass ? w : 0.0;
        error.Stats[1] += w;
    }

    return error;
}

TString THammingLossMetric::GetDescription() const {
    if (IsMultiClass) {
        return BuildDescription(ELossFunction::HammingLoss, UseWeights, "%.3g", MakeBorderParam(Border));
    } else {
        return BuildDescription(ELossFunction::HammingLoss, UseWeights);
    }
}

void THammingLossMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

double THammingLossMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

TVector<TString> TMCCMetric::GetStatDescriptions() const {
    TVector<TString> result;
    for (int i = 0; i < ClassesCount; ++i) {
        for (int j = 0; j < ClassesCount; ++j) {
            result.push_back("ConfusionMatrix[" + ToString(i) + "][" + ToString(j) + "]");
        }
    }
    return result;
}

/* PairAccuracy */

namespace {
    struct TPairAccuracyMetric : public TAdditiveMetric<TPairAccuracyMetric> {
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

THolder<IMetric> MakePairAccuracyMetric() {
    return MakeHolder<TPairAccuracyMetric>();
}

TMetricHolder TPairAccuracyMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> /*target*/,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(approx.size() == 1, "Metric PairLogit supports only single-dimensional data");

    TMetricHolder error(2);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int begin = queriesInfo[queryIndex].Begin;
        for (int docId = 0; docId < queriesInfo[queryIndex].Competitors.ysize(); ++docId) {
            for (const auto& competitor : queriesInfo[queryIndex].Competitors[docId]) {
                const auto competitorWeight = UseWeights ? competitor.Weight : 1.0;
                if (approx[0][begin + docId] > approx[0][begin + competitor.Id]) {
                    error.Stats[0] += competitorWeight;
                }
                error.Stats[1] += competitorWeight;
            }
        }
    }
    return error;
}

EErrorType TPairAccuracyMetric::GetErrorType() const {
    return EErrorType::PairwiseError;
}

TString TPairAccuracyMetric::GetDescription() const {
    return BuildDescription(ELossFunction::PairAccuracy, UseWeights);
}

void TPairAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* PrecisionAtK */

namespace {
    struct TPrecisionAtKMetric: public TAdditiveMetric<TPrecisionAtKMetric> {
        explicit TPrecisionAtKMetric(int topSize, double border);
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    private:
        int TopSize;
        double Border;
    };
}

THolder<IMetric> MakePrecisionAtKMetric(int topSize, double border) {
    return MakeHolder<TPrecisionAtKMetric>(topSize, border);
}

TPrecisionAtKMetric::TPrecisionAtKMetric(int topSize, double border)
        : TopSize(topSize)
        , Border(border)
{
}

TMetricHolder TPrecisionAtKMetric::EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> /*weight*/,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int queryStartIndex,
        int queryEndIndex
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    TMetricHolder error(2);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int queryBegin = queriesInfo[queryIndex].Begin;
        int queryEnd = queriesInfo[queryIndex].End;

        TVector<double> approxCopy(approx[0].data() + queryBegin, approx[0].data() + queryEnd);
        TVector<float> targetCopy(target.begin() + queryBegin, target.begin() + queryEnd);

        error.Stats[0] += CalcPrecisionAtK(approxCopy, targetCopy, TopSize, Border);
        error.Stats[1]++;
    }
    return error;
}

EErrorType TPrecisionAtKMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TPrecisionAtKMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 1;
}

TString TPrecisionAtKMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != -1);
    return BuildDescription(ELossFunction::PrecisionAt, UseWeights, topSize, "%.3g", MakeBorderParam(Border));
}

void TPrecisionAtKMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* RecallAtK */

namespace {
    struct TRecallAtKMetric: public TAdditiveMetric<TRecallAtKMetric> {
        explicit TRecallAtKMetric(int topSize, double border);
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int TopSize;
        double Border;
    };
}

THolder<IMetric> MakeRecallAtKMetric(int topSize, double border) {
    return MakeHolder<TRecallAtKMetric>(topSize, border);
}

TRecallAtKMetric::TRecallAtKMetric(int topSize, double border)
        : TopSize(topSize)
        , Border(border)
{
}

TMetricHolder TRecallAtKMetric::EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> /*weight*/,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int queryStartIndex,
        int queryEndIndex
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    TMetricHolder error(2);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int queryBegin = queriesInfo[queryIndex].Begin;
        int queryEnd = queriesInfo[queryIndex].End;

        TVector<double> approxCopy(approx[0].data() + queryBegin, approx[0].data() + queryEnd);
        TVector<float> targetCopy(target.begin() + queryBegin, target.begin() + queryEnd);

        error.Stats[0] += CalcRecallAtK(approxCopy, targetCopy, TopSize, Border);
        error.Stats[1]++;
    }
    return error;
}

EErrorType TRecallAtKMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TRecallAtKMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 1;
}

TString TRecallAtKMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != -1);
    return BuildDescription(ELossFunction::RecallAt, UseWeights, topSize, "%.3g", MakeBorderParam(Border));
}

void TRecallAtKMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Mean Average Precision at k */

namespace {
    struct TMAPKMetric: public TAdditiveMetric<TMAPKMetric> {
        explicit TMAPKMetric(int topSize, double border);
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int TopSize;
        double Border;
    };
}

THolder<IMetric> MakeMAPKMetric(int topSize, double border) {
    return MakeHolder<TMAPKMetric>(topSize, border);
}

TMAPKMetric::TMAPKMetric(int topSize, double border)
        : TopSize(topSize)
        , Border(border)
{
}

TMetricHolder TMAPKMetric::EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> /*weight*/,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int queryStartIndex,
        int queryEndIndex
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    TMetricHolder error(2);

    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int queryBegin = queriesInfo[queryIndex].Begin;
        int queryEnd = queriesInfo[queryIndex].End;

        TVector<double> approxCopy(approx[0].data() + queryBegin, approx[0].data() + queryEnd);
        TVector<float> targetCopy(target.data() + queryBegin, target.data() + queryEnd);

        error.Stats[0] += CalcAveragePrecisionK(approxCopy, targetCopy, TopSize, Border);
        error.Stats[1]++;
    }
    return error;
}

TString TMAPKMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != -1);
    return BuildDescription(ELossFunction::MAP, UseWeights, topSize, "%.3g", MakeBorderParam(Border));
}

EErrorType TMAPKMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

// Mean average precision at K
double TMAPKMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TMAPKMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Custom */

namespace {
    class TCustomMetric: public IMetric {
    public:
        explicit TCustomMetric(const TCustomMetricDescriptor& descriptor);
        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor
        ) const override;
        TMetricHolder Eval(
            const TVector<TVector<double>>& /*approx*/,
            const TVector<TVector<double>>& /*approxDelta*/,
            bool /*isExpApprox*/,
            TConstArrayRef<float> /*target*/,
            TConstArrayRef<float> /*weight*/,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int /*begin*/,
            int /*end*/,
            NPar::TLocalExecutor& /*executor*/
        ) const override {
            CB_ENSURE(false, "Custom metrics do not support approx deltas and exponentiated approxes");
            return TMetricHolder();
        }
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        TVector<TString> GetStatDescriptions() const override;
        const TMap<TString, TString>& GetHints() const override;
        void AddHint(const TString& key, const TString& value) override;
        //we don't now anything about custom metrics
        bool IsAdditiveMetric() const final {
            return false;
        }
        // be conservative by default
        bool NeedTarget() const override {
            return true;
        }
    private:
        TCustomMetricDescriptor Descriptor;
        TMap<TString, TString> Hints;
    };
}

THolder<IMetric> MakeCustomMetric(const TCustomMetricDescriptor& descriptor) {
    return MakeHolder<TCustomMetric>(descriptor);
}

TCustomMetric::TCustomMetric(const TCustomMetricDescriptor& descriptor)
        : Descriptor(descriptor)
{
}

TMetricHolder TCustomMetric::Eval(
    const TVector<TVector<double>>& approx,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weightIn,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    auto weight = UseWeights ? weightIn : TConstArrayRef<float>{};
    TMetricHolder result = Descriptor.EvalFunc(approx, target, weight, begin, end, Descriptor.CustomData);
    CB_ENSURE(result.Stats.ysize() == 2, "Custom metric evaluate() returned incorrect value");
    return result;
}

TString TCustomMetric::GetDescription() const {
    TString description = Descriptor.GetDescriptionFunc(Descriptor.CustomData);
    return BuildDescription(description, UseWeights);
}

void TCustomMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    bool isMaxOptimal = Descriptor.IsMaxOptimalFunc(Descriptor.CustomData);
    *valueType = isMaxOptimal ? EMetricBestValue::Max : EMetricBestValue::Min;
}

EErrorType TCustomMetric::GetErrorType() const {
    return EErrorType::PerObjectError;
}

double TCustomMetric::GetFinalError(const TMetricHolder& error) const {
    return Descriptor.GetFinalErrorFunc(error, Descriptor.CustomData);
}

TVector<TString> TCustomMetric::GetStatDescriptions() const {
    return {"SumError", "SumWeight"};
}

const TMap<TString, TString>& TCustomMetric::GetHints() const {
    return Hints;
}

void TCustomMetric::AddHint(const TString& key, const TString& value) {
    Hints[key] = value;
}

/* UserDefinedPerObjectMetric */

namespace {
    class TUserDefinedPerObjectMetric : public TMetric {
    public:
        explicit TUserDefinedPerObjectMetric(const TMap<TString, TString>& params);
        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor
        ) const override;
        TMetricHolder Eval(
            const TVector<TVector<double>>& /*approx*/,
            const TVector<TVector<double>>& /*approxDelta*/,
            bool /*isExpApprox*/,
            TConstArrayRef<float> /*target*/,
            TConstArrayRef<float> /*weight*/,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int /*begin*/,
            int /*end*/,
            NPar::TLocalExecutor& /*executor*/
        ) const override {
            CB_ENSURE(
                false,
                "User-defined per object metrics do not support approx deltas and exponentiated approxes");
            return TMetricHolder();
        }
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const final {
            return true;
        }

    private:
        double Alpha;
    };
}

THolder<IMetric> MakeUserDefinedPerObjectMetric(const TMap<TString, TString>& params) {
    return MakeHolder<TUserDefinedPerObjectMetric>(params);
}

TUserDefinedPerObjectMetric::TUserDefinedPerObjectMetric(const TMap<TString, TString>& params)
        : Alpha(0.0)
{
    if (params.contains("alpha")) {
        Alpha = FromString<float>(params.at("alpha"));
    }
    UseWeights.MakeIgnored();
}

TMetricHolder TUserDefinedPerObjectMetric::Eval(
    const TVector<TVector<double>>& /*approx*/,
    TConstArrayRef<float> /*target*/,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int /*begin*/,
    int /*end*/,
    NPar::TLocalExecutor& /*executor*/
) const {
    CB_ENSURE(false, "Not implemented for TUserDefinedPerObjectMetric metric.");
    TMetricHolder metric(2);
    return metric;
}

TString TUserDefinedPerObjectMetric::GetDescription() const {
    return "UserDefinedPerObjectMetric";
}

void TUserDefinedPerObjectMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* UserDefinedQuerywiseMetric */

namespace {
    class TUserDefinedQuerywiseMetric : public TAdditiveMetric<TUserDefinedQuerywiseMetric> {
    public:
        explicit TUserDefinedQuerywiseMetric(const TMap<TString, TString>& params);
        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        double Alpha;
    };
}

THolder<IMetric> MakeUserDefinedQuerywiseMetric(const TMap<TString, TString>& params) {
    return MakeHolder<TUserDefinedQuerywiseMetric>(params);
}

TUserDefinedQuerywiseMetric::TUserDefinedQuerywiseMetric(const TMap<TString, TString>& params)
        : Alpha(0.0)
{
    if (params.contains("alpha")) {
        Alpha = FromString<float>(params.at("alpha"));
    }
    UseWeights.MakeIgnored();
}

TMetricHolder TUserDefinedQuerywiseMetric::EvalSingleThread(
    const TVector<TVector<double>>& /*approx*/,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> /*target*/,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int /*queryStartIndex*/,
    int /*queryEndIndex*/
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(false, "Not implemented for TUserDefinedQuerywiseMetric metric.");
    TMetricHolder metric(2);
    return metric;
}

EErrorType TUserDefinedQuerywiseMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TUserDefinedQuerywiseMetric::GetDescription() const {
    return "TUserDefinedQuerywiseMetric";
}

void TUserDefinedQuerywiseMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* AverageGain */

namespace {
    class TAverageGain : public TAdditiveMetric<TAverageGain> {
    public:
        explicit TAverageGain(float topSize)
            : TopSize(topSize) {
            CB_ENSURE(topSize > 0, "top size for AverageGain should be greater than 0");
            CB_ENSURE(topSize == (int)topSize, "top size for AverageGain should be an integer value");
        }

        TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    private:
        int TopSize;
    };
}

THolder<IMetric> MakeAverageGainMetric(float topSize) {
    return MakeHolder<TAverageGain>(topSize);
}

TMetricHolder TAverageGain::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(approx.size() == 1, "Metric AverageGain supports only single-dimensional data");

    TMetricHolder error(2);

    TVector<std::pair<double, ui32>> approxWithDoc;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        auto startIdx = queriesInfo[queryIndex].Begin;
        auto endIdx = queriesInfo[queryIndex].End;
        auto querySize = endIdx - startIdx;
        const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.0;

        double targetSum = 0;
        if ((int)querySize <= TopSize) {
            for (ui32 docId = startIdx; docId < endIdx; ++docId) {
                targetSum += target[docId];
            }
            error.Stats[0] += queryWeight * (targetSum / querySize);
        } else {
            approxWithDoc.yresize(querySize);
            for (ui32 i = 0; i < querySize; ++i) {
                ui32 docId = startIdx + i;
                approxWithDoc[i].first = approx[0][docId];
                approxWithDoc[i].second = docId;;
            }
            std::nth_element(approxWithDoc.begin(), approxWithDoc.begin() + TopSize, approxWithDoc.end(),
                             [&](std::pair<double, ui32> left, std::pair<double, ui64> right) -> bool {
                                 return CompareDocs(left.first, target[left.second], right.first, target[right.second]);
                             });
            for (int i = 0; i < TopSize; ++i) {
                targetSum += target[approxWithDoc[i].second];
            }
            error.Stats[0] += queryWeight * (targetSum / TopSize);
        }
        error.Stats[1] += queryWeight;
    }
    return error;
}

EErrorType TAverageGain::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TAverageGain::GetDescription() const {
    TMetricParam<int> topSize("top", TopSize, /*userDefined*/true);
    return BuildDescription(ELossFunction::AverageGain, UseWeights, topSize);
}

void TAverageGain::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Create */

static void CheckParameters(
    const TString& metricName,
    const TSet<TString>& validParam,
    const TMap<TString, TString>& inputParams) {
    TString warning = "";
    for (const auto& param : validParam) {
        warning += (warning.empty() ? "" : ", ");
        warning += param;
    }

    warning = (validParam.size() == 1 ? "Valid parameter is " : "Valid parameters are ") + warning + ".";

    for (const auto& param : inputParams) {
        CB_ENSURE(validParam.contains(param.first),
                  metricName + " metric shouldn't have " + param.first + " parameter. " + warning);
    }
}

int GetParameterTop(const TMap<TString, TString>& params, ELossFunction metric) {
    auto itTopSize = params.find("top");
    int topSize = -1;

    if (itTopSize != params.end()) {
        topSize = FromString<int>(itTopSize->second);
    }
    TString metricName;
    switch (metric) {
        case ELossFunction::PrecisionAt:
            metricName = "Precision at K";
            break;

        case ELossFunction::RecallAt:
            metricName = "Recall at K";
            break;

        case ELossFunction::MAP:
            metricName = "Mean Average Precision at K";
            break;

        default:
            Y_ASSERT(false);
    }
    return topSize;
}

static TVector<THolder<IMetric>> CreateMetric(ELossFunction metric, TMap<TString, TString> params, int approxDimension) {
    double border = GetDefaultClassificationBorder();
    if (params.contains("border")) {
        border = FromString<float>(params.at("border"));
    }

    TVector<THolder<IMetric>> result;
    TSet<TString> validParams;
    switch (metric) {
        case ELossFunction::Logloss:
            result.push_back(MakeCrossEntropyMetric(ELossFunction::Logloss, border));
            validParams = {"border"};
            break;

        case ELossFunction::CrossEntropy:
            result.push_back(MakeCrossEntropyMetric(ELossFunction::CrossEntropy));
            break;
        case ELossFunction::RMSE:
            result.push_back(MakeRMSEMetric());
            break;
        case ELossFunction::Lq:
            CB_ENSURE(params.contains("q"), "Metric " << ELossFunction::Lq << " requirese q as parameter");
            validParams={"q"};
            result.push_back(MakeLqMetric(FromString<float>(params.at("q"))));
            break;
        case ELossFunction::MAE:
            result.push_back(MakeQuantileMetric(ELossFunction::MAE));
            break;

        case ELossFunction::Quantile: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.push_back(MakeQuantileMetric(ELossFunction::Quantile, FromString<float>(it->second)));
            } else {
                result.push_back(MakeQuantileMetric(ELossFunction::Quantile));
            }
            validParams = {"alpha"};
            break;
        }

        case ELossFunction::LogLinQuantile: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.push_back(MakeLogLinQuantileMetric(FromString<float>(it->second)));
            } else {
                result.push_back(MakeLogLinQuantileMetric());
            }
            validParams = {"alpha"};
            break;
        }

        case ELossFunction::AverageGain:
        case ELossFunction::QueryAverage: {
            auto it = params.find("top");
            CB_ENSURE(it != params.end(), "AverageGain metric should have top parameter");
            result.emplace_back(new TAverageGain(FromString<float>(it->second)));
            validParams = {"top"};
            break;
        }

        case ELossFunction::MAPE:
            result.push_back(MakeMAPEMetric());
            break;

        case ELossFunction::Poisson:
            result.push_back(MakePoissonMetric());
            break;

        case ELossFunction::MedianAbsoluteError:
            result.push_back(MakeMedianAbsoluteErrorMetric());
            break;

        case ELossFunction::SMAPE:
            result.push_back(MakeSMAPEMetric());
            break;

        case ELossFunction::MSLE:
            result.push_back(MakeMSLEMetric());
            break;

        case ELossFunction::MultiClass:
            result.push_back(MakeMultiClassMetric());
            break;

        case ELossFunction::MultiClassOneVsAll:
            result.push_back(MakeMultiClassOneVsAllMetric());
            break;

        case ELossFunction::PairLogit:
            result.push_back(MakePairLogitMetric());
            validParams = {"max_pairs"};
            break;

        case ELossFunction::PairLogitPairwise:
            result.push_back(MakePairLogitMetric());
            validParams = {"max_pairs"};
            break;

        case ELossFunction::QueryRMSE:
            result.push_back(MakeQueryRMSEMetric());
            break;

        case ELossFunction::QuerySoftMax:
            result.emplace_back(new TQuerySoftMaxMetric());
            validParams = {"lambda"};
            break;

        case ELossFunction::YetiRank:
            result.push_back(MakePFoundMetric());
            validParams = {"decay", "permutations"};
            CB_ENSURE(!params.contains("permutations") || FromString<int>(params.at("permutations")) > 0, "Metric " << metric << " expects permutations > 0");
            break;

        case ELossFunction::YetiRankPairwise:
            result.push_back(MakePFoundMetric());
            validParams = {"decay", "permutations", "sampling_type"};
            CB_ENSURE(!params.contains("permutations") || FromString<int>(params.at("permutations")) > 0, "Metric " << metric << " expects permutations > 0");
            break;

        case ELossFunction::PFound: {
            auto itTopSize = params.find("top");
            auto itDecay = params.find("decay");
            int topSize = itTopSize != params.end() ? FromString<int>(itTopSize->second) : -1;
            double decay = itDecay != params.end() ? FromString<double>(itDecay->second) : 0.85;
            result.push_back(MakePFoundMetric(topSize, decay));
            validParams = {"top", "decay"};
            break;
        }

        case ELossFunction::LogLikelihoodOfPrediction:
            result.push_back(MakeLLPMetric());
            break;

        case ELossFunction::NDCG: {
            auto itTopSize = params.find("top");
            auto itType = params.find("type");
            int topSize = itTopSize != params.end() ? FromString<int>(itTopSize->second) : -1;

            ENdcgMetricType type = ENdcgMetricType::Base;

            if (itType != params.end()) {
                type = FromString<ENdcgMetricType>(itType->second);
            }

            result.emplace_back(new TNdcgMetric(topSize, type));
            validParams = {"top", "type"};
            break;
        }

        case ELossFunction::R2:
            result.push_back(MakeR2Metric());
            break;
        case ELossFunction::NumErrors: {
            CB_ENSURE(params.contains("greater_than"), "Metric " << ELossFunction::NumErrors << " requires greater_than as parameter");
            result.push_back(MakeNumErrorsMetric(FromString<double>(params.at("greater_than"))));
            validParams = {"greater_than"};
            break;
        }
        case ELossFunction::AUC: {
            if (approxDimension == 1) {
                result.push_back(MakeBinClassAucMetric(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.push_back(MakeMultiClassAucMetric(i));
                }
            }
            break;
        }

        case ELossFunction::Accuracy:
            result.push_back(MakeAccuracyMetric(border));
            validParams = {"border"};
            break;

        case ELossFunction::CtrFactor:
            result.push_back(MakeCtrFactorMetric(border));
            validParams = {"border"};
            break;

        case ELossFunction::Precision: {
            if (approxDimension == 1) {
                result.emplace_back(MakeBinClassPrecisionMetric(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(MakeMultiClassPrecisionMetric(i));
                }
            }
            break;
        }

        case ELossFunction::Recall: {
            if (approxDimension == 1) {
                result.emplace_back(MakeBinClassRecallMetric(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(MakeMultiClassRecallMetric(i));
                }
            }
            break;
        }

        case ELossFunction::BalancedAccuracy: {
            CB_ENSURE(approxDimension == 1, "Balanced accuracy is used only for binary classification problems.");
            validParams = {"border"};
            result.emplace_back(MakeBinClassBalancedAccuracyMetric(border));
            break;
        }

        case ELossFunction::BalancedErrorRate: {
            CB_ENSURE(approxDimension == 1, "Balanced Error Rate is used only for binary classification problems.");
            validParams = {"border"};
            result.emplace_back(MakeBinClassBalancedErrorRate(border));
            break;
        }

        case ELossFunction::Kappa: {
            if (approxDimension == 1) {
                validParams = {"border"};
                result.emplace_back(MakeBinClassKappaMetric(border));
            } else {
                result.emplace_back(MakeMultiClassKappaMetric(approxDimension));
            }
            break;
        }

        case ELossFunction::WKappa: {
            if (approxDimension == 1) {
                validParams = {"border"};
                result.emplace_back(MakeBinClassWKappaMetric(border));
            } else {
                result.emplace_back(MakeMultiClassWKappaMetric(approxDimension));
            }
            break;
        }

        case ELossFunction::F1: {
            if (approxDimension == 1) {
                result.emplace_back(MakeBinClassF1Metric(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(MakeMultiClassF1Metric(i));
                }
            }
            break;
        }

        case ELossFunction::TotalF1:
            result.push_back(MakeTotalF1Metric(approxDimension == 1 ? 2 : approxDimension));
            break;

        case ELossFunction::MCC:
            result.push_back(MakeMCCMetric(approxDimension == 1 ? 2 : approxDimension));
            break;

        case ELossFunction::BrierScore:
            CB_ENSURE(approxDimension == 1, "Brier Score is used only for binary classification problems.");
            result.push_back(MakeBrierScoreMetric());
            break;

        case ELossFunction::ZeroOneLoss:
            result.emplace_back(MakeZeroOneLossMetric(border, approxDimension > 1));
            validParams = {"border"};
            break;

        case ELossFunction::HammingLoss:
            result.push_back(MakeHammingLossMetric(border, approxDimension > 1));
            validParams = {"border"};
            break;

        case ELossFunction::HingeLoss:
            result.push_back(MakeHingeLossMetric());
            break;

        case ELossFunction::PairAccuracy:
            result.emplace_back(MakePairAccuracyMetric());
            break;

        case ELossFunction::PrecisionAt: {
            int topSize = GetParameterTop(params, ELossFunction::PrecisionAt);
            validParams = {"top", "border"};
            result.emplace_back(MakePrecisionAtKMetric(topSize, border));
            break;
        }

        case ELossFunction::RecallAt: {
            int topSize = GetParameterTop(params, ELossFunction::RecallAt);
            validParams = {"top", "border"};
            result.emplace_back(MakeRecallAtKMetric(topSize, border));
            break;
        }

        case ELossFunction::MAP: {
            int topSize = GetParameterTop(params, ELossFunction::MAP);
            validParams = {"top", "border"};
            result.emplace_back(MakeMAPKMetric(topSize, border));
            break;
        }

        case ELossFunction::UserPerObjMetric: {
            result.emplace_back(MakeUserDefinedPerObjectMetric(params));
            validParams = {"alpha"};
            break;
        }

        case ELossFunction::UserQuerywiseMetric: {
            result.emplace_back(MakeUserDefinedQuerywiseMetric(params));
            validParams = {"alpha"};
            break;
        }
        case ELossFunction::QueryCrossEntropy: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.push_back(MakeQueryCrossEntropyMetric(FromString<float>(it->second)));
            } else {
                result.push_back(MakeQueryCrossEntropyMetric());
            }
            validParams = {"alpha"};
            break;
        }
        default:
            CB_ENSURE(false, "Unsupported loss_function: " << metric);
            return TVector<THolder<IMetric>>();
    }

    validParams.insert("hints");
    if (result && !result[0]->UseWeights.IsIgnored()) {
        validParams.insert("use_weights");
    }

    if (ShouldSkipCalcOnTrainByDefault(metric)) {
        for (THolder<IMetric>& metric : result) {
            metric->AddHint("skip_train", "true");
        }
    }

    if (params.contains("hints")) { // TODO(smirnovpavel): hints shouldn't be added for each metric
        TMap<TString, TString> hints = ParseHintsDescription(params.at("hints"));
        for (const auto& hint : hints) {
            for (THolder<IMetric>& metric : result) {
                metric->AddHint(hint.first, hint.second);
            }
        }
    }

    if (params.contains("use_weights")) {
        const bool useWeights = FromString<bool>(params.at("use_weights"));
        for (THolder<IMetric>& metric : result) {
            metric->UseWeights = useWeights;
        }
    }

    CheckParameters(ToString(metric), validParams, params);

    return result;
}

static TVector<THolder<IMetric>> CreateMetricFromDescription(const TString& description, int approxDimension) {
    ELossFunction metric = ParseLossType(description);
    TMap<TString, TString> params = ParseLossParams(description);
    return CreateMetric(metric, params, approxDimension);
}

TVector<THolder<IMetric>> CreateMetricsFromDescription(const TVector<TString>& description, int approxDim) {
    TVector<THolder<IMetric>> metrics;
    for (const auto& metricDescription : description) {
        auto metricsBatch = CreateMetricFromDescription(metricDescription, approxDim);
        for (ui32 i = 0; i < metricsBatch.size(); ++i) {
            metrics.push_back(std::move(metricsBatch[i]));
        }
    }
    return metrics;
}

TVector<THolder<IMetric>> CreateMetricFromDescription(const NCatboostOptions::TLossDescription& description, int approxDimension) {
    auto metric = description.GetLossFunction();
    return CreateMetric(metric, description.GetLossParams(), approxDimension);
}

TVector<THolder<IMetric>> CreateMetrics(
    TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
    int approxDim) {

    TVector<THolder<IMetric>> metrics;
    for (const auto& metricDescription : metricDescriptions) {
        auto metricsBatch = CreateMetricFromDescription(metricDescription, approxDim);
        for (ui32 i = 0; i < metricsBatch.size(); ++i) {
            metrics.push_back(std::move(metricsBatch[i]));
        }
    }
    return metrics;
}


TVector<THolder<IMetric>> CreateMetrics(
        const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
        const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        int approxDimension
) {
    TVector<THolder<IMetric>> errors;
    THashSet<TString> usedDescriptions;

    if (evalMetricOptions->EvalMetric.IsSet()) {
        if (evalMetricOptions->EvalMetric->GetLossFunction() == ELossFunction::Custom) {
            errors.emplace_back(MakeCustomMetric(*evalMetricDescriptor));
        } else {
            TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(evalMetricOptions->EvalMetric, approxDimension);
            CB_ENSURE(createdMetrics.size() == 1, "Eval metric should have a single value. Metric " <<
                ToString(evalMetricOptions->EvalMetric->GetLossFunction()) <<
                " provides a value for each class, thus it cannot be used as " <<
                "a single value to select best iteration or to detect overfitting. " <<
                "If you just want to look on the values of this metric use custom_metric parameter.");
            errors.push_back(std::move(createdMetrics.front()));
        }
        usedDescriptions.insert(errors.back()->GetDescription());
    }

    if (lossFunctionOption->GetLossFunction() != ELossFunction::Custom) {
        TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(lossFunctionOption, approxDimension);
        for (auto& metric : createdMetrics) {
            if (!usedDescriptions.contains(metric->GetDescription())) {
                usedDescriptions.insert(metric->GetDescription());
                errors.push_back(std::move(metric));
            }
        }
    }

    for (const auto& description : evalMetricOptions->CustomMetrics.Get()) {
        TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(description, approxDimension);
        for (auto& metric : createdMetrics) {
            if (!usedDescriptions.contains(metric->GetDescription())) {
                usedDescriptions.insert(metric->GetDescription());
                errors.push_back(std::move(metric));
            }
        }
    }

    return errors;
}

TVector<TString> GetMetricsDescription(const TVector<const IMetric*>& metrics) {
    TVector<TString> result;
    result.reserve(metrics.size());
    for (const auto& metric : metrics) {
        result.push_back(metric->GetDescription());
    }
    return result;
}

TVector<TString> GetMetricsDescription(const TVector<THolder<IMetric>>& metrics) {
    return GetMetricsDescription(GetConstPointers(metrics));
}

TVector<bool> GetSkipMetricOnTrain(const TVector<const IMetric*>& metrics) {
    TVector<bool> result;
    result.reserve(metrics.size());
    for (const auto& metric : metrics) {
        const TMap<TString, TString>& hints = metric->GetHints();
        result.push_back(hints.contains("skip_train") && hints.at("skip_train") == "true");
    }
    return result;
}

TVector<bool> GetSkipMetricOnTrain(const TVector<THolder<IMetric>>& metrics) {
    return GetSkipMetricOnTrain(GetConstPointers(metrics));
}

TVector<bool> GetSkipMetricOnTest(bool testHasTarget, const TVector<const IMetric*>& metrics) {
    TVector<bool> result;
    result.reserve(metrics.size());
    for (const auto& metric : metrics) {
        result.push_back(!testHasTarget && metric->NeedTarget());
    }
    return result;
}


TMetricHolder EvalErrors(
        const TVector<TVector<double>>& approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        const THolder<IMetric>& error,
        NPar::TLocalExecutor* localExecutor
) {
    if (error->GetErrorType() == EErrorType::PerObjectError) {
        int begin = 0, end = target.size();
        Y_VERIFY(approx[0].ysize() == end - begin);
        return error->Eval(approx, target, weight, queriesInfo, begin, end, *localExecutor);
    } else {
        Y_VERIFY(error->GetErrorType() == EErrorType::QuerywiseError || error->GetErrorType() == EErrorType::PairwiseError);
        int queryStartIndex = 0, queryEndIndex = queriesInfo.size();
        return error->Eval(approx, target, weight, queriesInfo, queryStartIndex, queryEndIndex, *localExecutor);
    }
}


TMetricHolder EvalErrors(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        const THolder<IMetric>& error,
        NPar::TLocalExecutor* localExecutor
) {
    if (error->GetErrorType() == EErrorType::PerObjectError) {
        int begin = 0, end = target.size();
        Y_VERIFY(end <= approx[0].ysize());
        return error->Eval(approx, approxDelta, isExpApprox, target, weight, queriesInfo, begin, end, *localExecutor);
    } else {
        Y_VERIFY(error->GetErrorType() == EErrorType::QuerywiseError || error->GetErrorType() == EErrorType::PairwiseError);
        int queryStartIndex = 0, queryEndIndex = queriesInfo.size();
        return error->Eval(approx, approxDelta, isExpApprox, target, weight, queriesInfo, queryStartIndex, queryEndIndex, *localExecutor);
    }
}


static inline double BestQueryShift(const double* cursor,
                                    const float* targets,
                                    const float* weights,
                                    int size) {
    double bestShift = 0;
    double left = -20;
    double right = 20;

    for (int i = 0; i < 30; ++i) {
        double der = 0;
        if (weights) {
            for (int doc = 0; doc < size; ++doc) {
                const double expApprox = exp(cursor[doc] + bestShift);
                const double p = (std::isfinite(expApprox) ? (expApprox / (1.0 + expApprox)) : 1.0);
                der += weights[doc] * (targets[doc] - p);
            }
        } else {
            for (int doc = 0; doc < size; ++doc) {
                const double expApprox = exp(cursor[doc] + bestShift);
                const double p = (std::isfinite(expApprox) ? (expApprox / (1.0 + expApprox)) : 1.0);
                der += (targets[doc] - p);
            }
        }

        if (der > 0) {
            left = bestShift;
        } else {
            right = bestShift;
        }

        bestShift = (left + right) / 2;
    }
    return bestShift;
}

static inline bool IsSingleClassQuery(const float* targets, int querySize) {
    for (int i = 1; i < querySize; ++i) {
        if (Abs(targets[i] - targets[0]) > 1e-20) {
            return false;
        }
    }
    return true;
}

namespace {
    struct TQueryCrossEntropyMetric : public TAdditiveMetric<TQueryCrossEntropyMetric> {
        explicit TQueryCrossEntropyMetric(double alpha);
        TMetricHolder EvalSingleThread(
                const TVector<TVector<double>>& approx,
                const TVector<TVector<double>>& approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const;
        EErrorType GetErrorType() const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        void AddSingleQuery(const double* approxes,
                            const float* target,
                            const float* weight,
                            int querySize,
                            TMetricHolder* metricHolder) const;
    private:
        double Alpha;
    };
}

THolder<IMetric> MakeQueryCrossEntropyMetric(double alpha) {
    return MakeHolder<TQueryCrossEntropyMetric>(alpha);
}

void TQueryCrossEntropyMetric::AddSingleQuery(const double* approxes, const float* targets, const float* weights, int querySize,
                                              TMetricHolder* metricHolder) const {
    const double bestShift = BestQueryShift(approxes, targets, weights, querySize);

    double sum = 0;
    double weight = 0;

    const bool isSingleClassQuery = IsSingleClassQuery(targets, querySize);
    for (int i = 0; i < querySize; ++i) {
        const double approx = approxes[i];
        const double target = targets[i];
        const double w = weights ? weights[i] : 1.0;

        const double expApprox = exp(approx);
        const double shiftedExpApprox = exp(approx + bestShift);

        {
            const double logExpValPlusOne = std::isfinite(expApprox + 1) ? log(1 + expApprox) : approx;
            const double llp = -w * (target * approx - logExpValPlusOne);
            sum += (1.0 - Alpha) * llp;
        }

        if (!isSingleClassQuery) {
            const double shiftedApprox = approx + bestShift;
            const double logExpValPlusOne = std::isfinite(shiftedExpApprox + 1) ? log(1 + shiftedExpApprox) : shiftedApprox;
            const double llmax = -w * (target * shiftedApprox - logExpValPlusOne);
            sum += Alpha * llmax;
        }
        weight += w;
    }

    metricHolder->Stats[0] += sum;
    metricHolder->Stats[1] += weight;
}


TMetricHolder TQueryCrossEntropyMetric::EvalSingleThread(const TVector<TVector<double>>& approx,
                                                         const TVector<TVector<double>>& approxDelta,
                                                         bool isExpApprox,
                                                         TConstArrayRef<float> target,
                                                         TConstArrayRef<float> weight,
                                                         TConstArrayRef<TQueryInfo> queriesInfo,
                                                         int queryStartIndex,
                                                         int queryEndIndex) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    TMetricHolder result(2);
    for (int qid = queryStartIndex; qid < queryEndIndex; ++qid) {
        auto& qidInfo = queriesInfo[qid];
        AddSingleQuery(
                approx[0].data() + qidInfo.Begin,
                target.data() + qidInfo.Begin,
                weight.empty() ? nullptr : weight.data() + qidInfo.Begin,
                qidInfo.End - qidInfo.Begin,
                &result);
    }
    return result;
}

EErrorType TQueryCrossEntropyMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}


TString TQueryCrossEntropyMetric::GetDescription() const {
    const TMetricParam<double> alpha("alpha", Alpha, /*userDefined*/true);
    return BuildDescription(ELossFunction::QueryCrossEntropy, UseWeights, "%.3g", alpha);
}

TQueryCrossEntropyMetric::TQueryCrossEntropyMetric(double alpha)
        : Alpha(alpha) {
}

void TQueryCrossEntropyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

inline void CheckMetric(const ELossFunction metric, const ELossFunction modelLoss) {
    if (metric == ELossFunction::Custom || modelLoss == ELossFunction::Custom) {
        return;
    }

    if (IsMultiDimensionalError(metric) && !IsSingleDimensionalError(metric)) {
        CB_ENSURE(IsMultiDimensionalError(modelLoss),
            "Cannot use strict multiclassification and not multiclassification metrics together: "
            << "If you din't train multiclassification, use binary classification, regression or ranking metrics instead.");
    }

    if (IsMultiDimensionalError(modelLoss) && !IsSingleDimensionalError(modelLoss)) {
        CB_ENSURE(IsMultiDimensionalError(metric),
            "Cannot use strict multiclassification and not multiclassification metrics together: "
            << "If you trained multiclassification, use multiclassification metrics.");
    }

    if (IsForCrossEntropyOptimization(modelLoss)) {
        CB_ENSURE(IsForCrossEntropyOptimization(metric) || IsForOrderOptimization(metric),
                  "Cannot calc metric which requires absolute values for logits.");
    }

    if (IsForOrderOptimization(modelLoss)) {
        CB_ENSURE(IsForOrderOptimization(metric),
                  "Cannot calc metric which requires logits or absolute values with order-optimization loss together.");
    }

    if (IsForAbsoluteValueOptimization(modelLoss)) {
        CB_ENSURE(IsForAbsoluteValueOptimization(metric) || IsForOrderOptimization(metric),
                  "Cannot calc metric which requires logits for absolute values.");
    }
}

void CheckMetrics(const TVector<THolder<IMetric>>& metrics, const ELossFunction modelLoss) {
    CB_ENSURE(!metrics.empty(), "No metrics specified for evaluation");
    for (int i = 0; i < metrics.ysize(); ++i) {
        ELossFunction metric;
        try {
            metric = ParseLossType(metrics[i]->GetDescription());
        } catch (...) {
            metric = ELossFunction::Custom;
        }
        CheckMetric(metric, modelLoss);
    }
}

void CheckPreprocessedTarget(
    TConstArrayRef<float> target,
    const NCatboostOptions::TLossDescription& lossDesciption,
    bool isLearnData,
    bool allowConstLabel
) {
    ELossFunction lossFunction = lossDesciption.GetLossFunction();
    if (isLearnData && (lossFunction == ELossFunction::Logloss)) {
        auto border = NCatboostOptions::GetLogLossBorder(lossDesciption);
        auto targetBounds = CalcMinMax(target);
        CB_ENSURE(targetBounds.Min <= border, "All train targets are greater than border " << border);
        CB_ENSURE(targetBounds.Max > border,
                  "All train targets are smaller than or equal to border " << border);
    }
    if (isLearnData && (lossFunction != ELossFunction::PairLogit)) {
        auto targetBounds = CalcMinMax(target);
        CB_ENSURE((targetBounds.Min != targetBounds.Max) || allowConstLabel, "All train targets are equal");
    }
    if (lossFunction == ELossFunction::CrossEntropy) {
        auto targetBounds = CalcMinMax(target);
        CB_ENSURE(targetBounds.Min >= 0, "Min target less than 0: " + ToString(targetBounds.Min));
        CB_ENSURE(targetBounds.Max <= 1, "Max target greater than 1: " + ToString(targetBounds.Max));
    }

    if (lossFunction == ELossFunction::QuerySoftMax) {
        float minTarget = *MinElement(target.begin(), target.end());
        CB_ENSURE(minTarget >= 0, "Min target less than 0: " + ToString(minTarget));
    }

    if (IsMultiClassMetric(lossFunction) && !IsBinaryClassMetric(lossFunction)) {
        CB_ENSURE(AllOf(target, [](float x) { return int(x) == x && x >= 0; }),
                  "metric/loss-function " << lossFunction << " is a Multiclassification metric, "
                  " each target label should be a nonnegative integer");
    }
}
