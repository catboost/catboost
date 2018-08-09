#include "metric.h"
#include "auc.h"
#include "balanced_accuracy.h"
#include "brier_score.h"
#include "classification_utils.h"
#include "dcg.h"
#include "doc_comparator.h"
#include "hinge_loss.h"
#include "kappa.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/loss_description.h>

#include <util/generic/ymath.h>
#include <util/generic/string.h>
#include <util/generic/maybe.h>
#include <util/string/iterator.h>
#include <util/string/cast.h>
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

/* CrossEntropy */

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
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
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

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder holder(2);
    const double* approxPtr = approxVec.data();
    const float* targetPtr = target.data();
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        const double approxExp = exp(approxPtr[i]);
        //this check should not be bottleneck
        const float prob = LossFunction == ELossFunction::Logloss ? targetPtr[i] > Border : targetPtr[i];
        holder.Stats[0] += w * ((IsFinite(approxExp) ? log(1 + approxExp) : approxPtr[i]) - prob * approxPtr[i]);
        holder.Stats[1] += w;
    }
    return holder;
}

TString TCrossEntropyMetric::GetDescription() const {
    return BuildDescription(LossFunction, UseWeights, MakeBorderParam(Border));

}

void TCrossEntropyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* CtrFactor */

TMetricHolder TCtrFactorMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric CtrFactor supports only single-dimensional data");

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

TMetricHolder TRMSEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric RMSE supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Stats[0] += Sqr(approxVec[k] - target[k]) * w;
        error.Stats[1] += w;
    }
    return error;
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

/* Quantile */

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
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric quantile supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        double val = target[i] - approxVec[i];
        double multiplier = (val > 0) ? Alpha : -(1 - Alpha);
        error.Stats[0] += (multiplier * val) * w;
        error.Stats[1] += w;
    }
    if (LossFunction == ELossFunction::MAE) {
        error.Stats[0] *= 2;
    }
    return error;
}

TString TQuantileMetric::GetDescription() const {
    if (LossFunction == ELossFunction::Quantile) {
        const TMetricParam<double> alpha("alpha", Alpha, /*userDefined*/true);
        return BuildDescription(LossFunction, UseWeights, "%.3lf", alpha);
    } else {
        return BuildDescription(LossFunction, UseWeights);
    }
}

void TQuantileMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* LogLinQuantile */

TLogLinQuantileMetric::TLogLinQuantileMetric(double alpha)
        : Alpha(alpha)
{
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6, "Alpha parameter for log-linear quantile metric should be in interval (0, 1)");
}

TMetricHolder TLogLinQuantileMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric log-linear quantile supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        double val = target[i] - exp(approxVec[i]);
        double multiplier = (val > 0) ? Alpha : -(1 - Alpha);
        error.Stats[0] += (multiplier * val) * w;
        error.Stats[1] += w;
    }

    return error;
}

TString TLogLinQuantileMetric::GetDescription() const {
    const TMetricParam<double> alpha("alpha", Alpha, /*userDefined*/true);
    return BuildDescription(ELossFunction::LogLinQuantile, UseWeights, "%.3lf", alpha);
}

void TLogLinQuantileMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* MAPE */

TMetricHolder TMAPEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric MAPE quantile supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Stats[0] += Abs(1 - approxVec[k] / target[k]) * w;
        error.Stats[1] += w;
    }

    return error;
}

TString TMAPEMetric::GetDescription() const {
    return BuildDescription(ELossFunction::MAPE, UseWeights);
}

void TMAPEMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Poisson */

TMetricHolder TPoissonMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    // Error function:
    // Sum_d[approx(d) - target(d) * log(approx(d))]
    // approx(d) == exp(Sum(tree_value))

    Y_ASSERT(approx.size() == 1);

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        error.Stats[0] += (exp(approxVec[i]) - target[i] * approxVec[i]) * w;
        error.Stats[1] += w;
    }

    return error;
}

TString TPoissonMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Poisson, UseWeights);
}

void TPoissonMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* Mean squared logarithmic error */

TMetricHolder TMSLEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric Mean squared logarithmic error supports only single-dimensional data");
    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

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

TMetricHolder TMedianAbsoluteErrorMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /*executor*/
) const {
    CB_ENSURE(approx.size() == 1, "Metric Median absolute error supports only single-dimensional data");
    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder error(2);
    TVector<double> values;
    values.reserve(end - begin);
    for (int i = begin; i < end; ++i) {
        values.push_back(fabs(approxVec[i] - target[i]));
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

TMetricHolder TSMAPEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Symmetric mean absolute percentage error supports only single-dimensional data");
    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

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

/* MultiClass */

TMetricHolder TMultiClassMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());
    int approxDimension = approx.ysize();

    TMetricHolder error(2);

    for (int k = begin; k < end; ++k) {
        double maxApprox = std::numeric_limits<double>::min();
        int maxApproxIndex = 0;
        for (int dim = 1; dim < approxDimension; ++dim) {
            if (approx[dim][k] > maxApprox) {
                maxApprox = approx[dim][k];
                maxApproxIndex = dim;
            }
        }

        double sumExpApprox = 0;
        for (int dim = 0; dim < approxDimension; ++dim) {
            sumExpApprox += exp(approx[dim][k] - maxApprox);
        }

        int targetClass = static_cast<int>(target[k]);
        Y_ASSERT(targetClass >= 0 && targetClass < approx.ysize());
        double targetClassApprox = approx[targetClass][k];

        float w = weight.empty() ? 1 : weight[k];
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

TMetricHolder TMultiClassOneVsAllMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());
    int approxDimension = approx.ysize();

    TMetricHolder error(2);
    for (int k = begin; k < end; ++k) {
        double sumDimErrors = 0;
        for (int dim = 0; dim < approxDimension; ++dim) {
            double expApprox = exp(approx[dim][k]);
            sumDimErrors += IsFinite(expApprox) ? -log(1 + expApprox) : -approx[dim][k];
        }

        int targetClass = static_cast<int>(target[k]);
        Y_ASSERT(targetClass >= 0 && targetClass < approx.ysize());
        sumDimErrors += approx[targetClass][k];

        float w = weight.empty() ? 1 : weight[k];
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

TMetricHolder TPairLogitMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& /*target*/,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric PairLogit supports only single-dimensional data");

    TMetricHolder error(2);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int begin = queriesInfo[queryIndex].Begin;
        int end = queriesInfo[queryIndex].End;

        double maxQueryApprox = approx[0][begin];
        for (int docId = begin; docId < end; ++docId) {
            maxQueryApprox = Max(maxQueryApprox, approx[0][docId]);
        }
        TVector<double> approxExpShifted(end - begin);
        for (int docId = begin; docId < end; ++docId) {
            approxExpShifted[docId - begin] = exp(approx[0][docId] - maxQueryApprox);
        }

        for (int docId = 0; docId < queriesInfo[queryIndex].Competitors.ysize(); ++docId) {
            for (const auto& competitor : queriesInfo[queryIndex].Competitors[docId]) {
                const auto weight = UseWeights ? competitor.Weight : 1.0;
                error.Stats[0] += -weight * log(approxExpShifted[docId] / (approxExpShifted[docId] + approxExpShifted[competitor.Id]));
                error.Stats[1] += weight;
            }
        }
    }
    return error;
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

TMetricHolder TQueryRMSEMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric QueryRMSE supports only single-dimensional data");

    TMetricHolder error(2);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int begin = queriesInfo[queryIndex].Begin;
        int end = queriesInfo[queryIndex].End;
        double queryAvrg = CalcQueryAvrg(begin, end - begin, approx[0], target, weight);
        for (int docId = begin; docId < end; ++docId) {
            float w = weight.empty() ? 1 : weight[docId];
            error.Stats[0] += (Sqr(target[docId] - approx[0][docId] - queryAvrg)) * w;
            error.Stats[1] += w;
        }
    }
    return error;
}

double TQueryRMSEMetric::CalcQueryAvrg(
    int start,
    int count,
    const TVector<double>& approxes,
    const TVector<float>& targets,
    const TVector<float>& weights
) const {
    double qsum = 0;
    double qcount = 0;
    for (int docId = start; docId < start + count; ++docId) {
        double w = weights.empty() ? 1 : weights[docId];
        qsum += (targets[docId] - approxes[docId]) * w;
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

TPFoundMetric::TPFoundMetric(int topSize, double decay)
        : TopSize(topSize)
        , Decay(decay) {
}

TMetricHolder TPFoundMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    TPFoundCalcer calcer(TopSize, Decay);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int queryBegin = queriesInfo[queryIndex].Begin;
        int queryEnd = queriesInfo[queryIndex].End;
        const ui32* subgroupIdData = nullptr;
        const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.0;
        if (!queriesInfo[queryIndex].SubgroupId.empty()) {
            subgroupIdData = queriesInfo[queryIndex].SubgroupId.data();
        }
        calcer.AddQuery(target.data() + queryBegin, approx[0].data() + queryBegin, queryWeight, subgroupIdData, queryEnd - queryBegin);
    }
    return calcer.GetMetric();
}

EErrorType TPFoundMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TPFoundMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != -1);
    const TMetricParam<double> decay("decay", Decay, Decay != 0.85);
    return BuildDescription(ELossFunction::PFound, UseWeights, topSize, decay);
}

double TPFoundMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TPFoundMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* NDCG@N */

TNDCGMetric::TNDCGMetric(int topSize)
    : TopSize(topSize) {
}

TMetricHolder TNDCGMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    TMetricHolder error(2);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int queryBegin = queriesInfo[queryIndex].Begin;
        int queryEnd = queriesInfo[queryIndex].End;
        int querySize = queryEnd - queryBegin;
        const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.0;
        size_t sampleSize = (TopSize < 0 || querySize < TopSize) ? querySize : static_cast<size_t>(TopSize);

        TVector<double> approxCopy(approx[0].data() + queryBegin, approx[0].data() + queryBegin + sampleSize);
        TVector<double> targetCopy(target.data() + queryBegin, target.data() + queryBegin + sampleSize);
        TVector<NMetrics::TSample> samples = NMetrics::TSample::FromVectors(targetCopy, approxCopy);
        error.Stats[0] += queryWeight * CalcNDCG(samples);
        error.Stats[1] += queryWeight;
    }
    return error;
}

TString TNDCGMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != -1);
    return BuildDescription(ELossFunction::NDCG, UseWeights, topSize);
}

EErrorType TNDCGMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TNDCGMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TNDCGMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}


/* QuerySoftMax */

TMetricHolder TQuerySoftMaxMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric QuerySoftMax supports only single-dimensional data");

    TMetricHolder error(2);
    TVector<double> softmax;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int begin = queriesInfo[queryIndex].Begin;
        int end = queriesInfo[queryIndex].End;
        error.Add(EvalSingleQuery(begin, end - begin, approx[0], target, weight, &softmax));
    }
    return error;
}

EErrorType TQuerySoftMaxMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TMetricHolder TQuerySoftMaxMetric::EvalSingleQuery(
    int start,
    int count,
    const TVector<double>& approxes,
    const TVector<float>& targets,
    const TVector<float>& weights,
    TVector<double>* softmax
) const {
    double maxApprox = -std::numeric_limits<double>::max();
    double sumExpApprox = 0;
    double sumWeightedTargets = 0;
    for (int dim = 0; dim < count; ++dim) {
        if (weights.empty() || weights[start + dim] > 0) {
            maxApprox = std::max(maxApprox, approxes[start + dim]);
            if (targets[start + dim] > 0) {
                if (!weights.empty()) {
                    sumWeightedTargets += weights[start + dim] * targets[start + dim];
                } else {
                    sumWeightedTargets += targets[start + dim];
                }
            }
        }
    }

    TMetricHolder error(2);
    if (sumWeightedTargets > 0) {
        if (softmax->size() < static_cast<size_t>(count)) {
            softmax->resize(static_cast<size_t>(count));
        }
        for (int dim = 0; dim < count; ++dim) {
            if (weights.empty() || weights[start + dim] > 0) {
                double expApprox = exp(approxes[start + dim] - maxApprox);
                if (!weights.empty()) {
                    expApprox *= weights[start + dim];
                }
                (*softmax)[dim] = expApprox;
                sumExpApprox += expApprox;
            } else {
                (*softmax)[dim] = 0.0;
            }
        }
        if (!weights.empty()) {
            for (int dim = 0; dim < count; ++dim) {
                if (weights[start + dim] > 0 && targets[start + dim] > 0) {
                    error.Stats[0] -= weights[start + dim] * targets[start + dim] * log((*softmax)[dim] / sumExpApprox);
                }
            }
        } else {
            for (int dim = 0; dim < count; ++dim) {
                if (targets[start + dim] > 0) {
                    error.Stats[0] -= targets[start + dim] * log((*softmax)[dim] / sumExpApprox);
                }
            }
        }
        error.Stats[1] = sumWeightedTargets;
    }
    return error;
}

TString TQuerySoftMaxMetric::GetDescription() const {
    return BuildDescription(ELossFunction::QuerySoftMax, UseWeights);
}

void TQuerySoftMaxMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* R2 */

TMetricHolder TR2Metric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Metric R2 supports only single-dimensional data");

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

THolder<TAUCMetric> TAUCMetric::CreateBinClassMetric(double border) {
    return new TAUCMetric(border);
}

THolder<TAUCMetric> TAUCMetric::CreateMultiClassMetric(int positiveClass) {
    CB_ENSURE(positiveClass >= 0, "Class id should not be negative");

    auto metric = new TAUCMetric();
    metric->PositiveClass = positiveClass;
    metric->IsMultiClass = true;
    return metric;
}

TMetricHolder TAUCMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weightIn,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    const auto& approxVec = approx.ysize() == 1 ? approx.front() : approx[PositiveClass];
    Y_ASSERT(approxVec.size() == target.size());
    const auto& weight = UseWeights ? weightIn : TVector<float>{};

    TVector<double> approxCopy(approxVec.begin() + begin, approxVec.begin() + end);
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
        return BuildDescription(ELossFunction::AUC, UseWeights, MakeBorderParam(Border));
    }
}

void TAUCMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Accuracy */

TMetricHolder TAccuracyMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());
    return GetAccuracy(approx, target, weight, begin, end, Border);
}

TString TAccuracyMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Accuracy, UseWeights, MakeBorderParam(Border));
}

void TAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Precision */

THolder<TPrecisionMetric> TPrecisionMetric::CreateBinClassMetric(double border) {
    return new TPrecisionMetric(border);
}

THolder<TPrecisionMetric> TPrecisionMetric::CreateMultiClassMetric(int positiveClass) {
    CB_ENSURE(positiveClass >= 0, "Class id should not be negative");

    auto metric = new TPrecisionMetric();
    metric->PositiveClass = positiveClass;
    metric->IsMultiClass = true;
    return metric;
}

TMetricHolder TPrecisionMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

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
        return BuildDescription(ELossFunction::Precision, UseWeights, MakeBorderParam(Border));
    }
}

void TPrecisionMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TPrecisionMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

/* Recall */

THolder<TRecallMetric> TRecallMetric::CreateBinClassMetric(double border) {
    return new TRecallMetric(border);
}

THolder<TRecallMetric> TRecallMetric::CreateMultiClassMetric(int positiveClass) {
    CB_ENSURE(positiveClass >= 0, "Class id should not be negative");

    auto metric = new TRecallMetric();
    metric->PositiveClass = positiveClass;
    metric->IsMultiClass = true;
    return metric;
}

TMetricHolder TRecallMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

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
        return BuildDescription(ELossFunction::Recall, UseWeights, MakeBorderParam(Border));
    }
}

double TRecallMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] > 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TRecallMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Balanced Accuracy */

THolder<TBalancedAccuracyMetric> TBalancedAccuracyMetric::CreateBinClassMetric(double border) {
    return new TBalancedAccuracyMetric(border);
}

TMetricHolder TBalancedAccuracyMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    return CalcBalancedAccuracyMetric(approx, target, weight, begin, end, PositiveClass, Border);
}

TString TBalancedAccuracyMetric::GetDescription() const {
    return BuildDescription(ELossFunction::BalancedAccuracy, UseWeights, MakeBorderParam(Border));
}

void TBalancedAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TBalancedAccuracyMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcBalancedAccuracyMetric(error);
}

/* Balanced Error Rate */

THolder<TBalancedErrorRate> TBalancedErrorRate::CreateBinClassMetric(double border) {
    return new TBalancedErrorRate(border);
}

TMetricHolder TBalancedErrorRate::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    return CalcBalancedAccuracyMetric(approx, target, weight, begin, end, PositiveClass, Border);
}

TString TBalancedErrorRate::GetDescription() const {
    return BuildDescription(ELossFunction::BalancedErrorRate, UseWeights, MakeBorderParam(Border));
}

void TBalancedErrorRate::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

double TBalancedErrorRate::GetFinalError(const TMetricHolder& error) const {
    return 1 - CalcBalancedAccuracyMetric(error);
}

/* Kappa */

THolder<TKappaMetric> TKappaMetric::CreateBinClassMetric(double border) {
    return new TKappaMetric(2, border);
}

THolder<TKappaMetric> TKappaMetric::CreateMultiClassMetric(int classCount) {
    return new TKappaMetric(classCount);
}

TMetricHolder TKappaMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    return CalcKappaMatrix(approx, target, begin, end, Border);
}

TString TKappaMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Kappa, MakeBorderParam(Border));
}

void TKappaMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TKappaMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Cohen);
}

/* WKappa */

THolder<TWKappaMatric> TWKappaMatric::CreateBinClassMetric(double border) {
    return new TWKappaMatric(2, border);
}

THolder<TWKappaMatric> TWKappaMatric::CreateMultiClassMetric(int classCount) {
    return new TWKappaMatric(classCount);
}

TMetricHolder TWKappaMatric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    return CalcKappaMatrix(approx, target, begin, end, Border);
}

TString TWKappaMatric::GetDescription() const {
    return BuildDescription(ELossFunction::WKappa, MakeBorderParam(Border));
}

void TWKappaMatric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TWKappaMatric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Weighted);
}

/* F1 */

THolder<TF1Metric> TF1Metric::CreateF1Multiclass(int positiveClass) {
    CB_ENSURE(positiveClass >= 0, "Class id should not be negative");
    THolder<TF1Metric> result = new TF1Metric;
    result->PositiveClass = positiveClass;
    result->IsMultiClass = true;
    return result;
}

THolder<TF1Metric> TF1Metric::CreateF1BinClass(double border) {
    THolder<TF1Metric> result = new TF1Metric;
    result->Border = border;
    result->IsMultiClass = false;
    return result;
}

TMetricHolder TF1Metric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

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
        return BuildDescription(ELossFunction::F1, UseWeights, MakeBorderParam(Border));
    }
}

void TF1Metric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* TotalF1 */

TMetricHolder TTotalF1Metric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
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
    const TVector<float>& target,
    const TVector<float>& weight,
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

TMetricHolder TMCCMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
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

TMetricHolder TBrierScoreMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(target.size() == weight.size(), "BrierScore metric requires weights");
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

TMetricHolder THingeLossMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
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

TZeroOneLossMetric::TZeroOneLossMetric(double border, bool isMultiClass)
        : Border(border),
          IsMultiClass(isMultiClass)
{
}

TMetricHolder TZeroOneLossMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());
    return GetAccuracy(approx, target, weight, begin, end, Border);
}

TString TZeroOneLossMetric::GetDescription() const {
    if (IsMultiClass) {
        return BuildDescription(ELossFunction::ZeroOneLoss, UseWeights, MakeBorderParam(Border));
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

THammingLossMetric::THammingLossMetric(double border, bool isMultiClass)
        : Border(border),
          IsMultiClass(isMultiClass)
{
}

TMetricHolder THammingLossMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());
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
        return BuildDescription(ELossFunction::HammingLoss, UseWeights, MakeBorderParam(Border));
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

TMetricHolder TPairAccuracyMetric::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& /*target*/,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
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

/* Custom */

TCustomMetric::TCustomMetric(const TCustomMetricDescriptor& descriptor)
        : Descriptor(descriptor)
{
}

TMetricHolder TCustomMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weightIn,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    const auto& weight = UseWeights ? weightIn : TVector<float>{};
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

TUserDefinedPerObjectMetric::TUserDefinedPerObjectMetric(const TMap<TString, TString>& params)
        : Alpha(0.0)
{
    if (params.has("alpha")) {
        Alpha = FromString<float>(params.at("alpha"));
    }
    UseWeights.MakeIgnored();
}

TMetricHolder TUserDefinedPerObjectMetric::Eval(
    const TVector<TVector<double>>& /*approx*/,
    const TVector<float>& /*target*/,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& /*queriesInfo*/,
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

TUserDefinedQuerywiseMetric::TUserDefinedQuerywiseMetric(const TMap<TString, TString>& params)
        : Alpha(0.0)
{
    if (params.has("alpha")) {
        Alpha = FromString<float>(params.at("alpha"));
    }
    UseWeights.MakeIgnored();
}

TMetricHolder TUserDefinedQuerywiseMetric::EvalSingleThread(
    const TVector<TVector<double>>& /*approx*/,
    const TVector<float>& /*target*/,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int /*queryStartIndex*/,
    int /*queryEndIndex*/
) const {
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

/* QueryAverage */

TMetricHolder TQueryAverage::EvalSingleThread(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& /*weight*/,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric QueryAverage supports only single-dimensional data");

    TMetricHolder error(2);

    TVector<std::pair<double, int>> approxWithDoc;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        auto startIdx = queriesInfo[queryIndex].Begin;
        auto endIdx = queriesInfo[queryIndex].End;
        auto querySize = endIdx - startIdx;
        const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.0;

        double targetSum = 0;
        if ((int)querySize <= TopSize) {
            for (int docId = startIdx; docId < endIdx; ++docId) {
                targetSum += target[docId];
            }
            error.Stats[0] += queryWeight * (targetSum / querySize);
        } else {
            approxWithDoc.yresize(querySize);
            for (int i = 0; i < querySize; ++i) {
                int docId = startIdx + i;
                approxWithDoc[i].first = approx[0][docId];
                approxWithDoc[i].second = docId;;
            }
            std::nth_element(approxWithDoc.begin(), approxWithDoc.begin() + TopSize, approxWithDoc.end(),
                             [&](std::pair<double, int> left, std::pair<double, int> right) -> bool {
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

EErrorType TQueryAverage::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TQueryAverage::GetDescription() const {
    TMetricParam<int> topSize("top", TopSize, /*userDefined*/true);
    return BuildDescription(ELossFunction::QueryAverage, UseWeights, topSize);
}

void TQueryAverage::GetBestValue(EMetricBestValue* valueType, float*) const {
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
        CB_ENSURE(validParam.has(param.first),
                  metricName + " metric shouldn't have " + param.first + " parameter. " + warning);
    }
}

static TVector<THolder<IMetric>> CreateMetric(ELossFunction metric, TMap<TString, TString> params, int approxDimension) {
    double border = GetDefaultClassificationBorder();
    if (params.has("border")) {
        border = FromString<float>(params.at("border"));
    }

    TVector<THolder<IMetric>> result;
    TSet<TString> validParams;
    switch (metric) {
        case ELossFunction::Logloss:
            result.emplace_back(new TCrossEntropyMetric(ELossFunction::Logloss, border));
            validParams = {"border"};
            break;

        case ELossFunction::CrossEntropy:
            result.emplace_back(new TCrossEntropyMetric(ELossFunction::CrossEntropy));
            break;
        case ELossFunction::RMSE:
            result.emplace_back(new TRMSEMetric());
            break;

        case ELossFunction::MAE:
            result.emplace_back(new TQuantileMetric(ELossFunction::MAE));
            break;

        case ELossFunction::Quantile: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.emplace_back(new TQuantileMetric(ELossFunction::Quantile, FromString<float>(it->second)));
            } else {
                result.emplace_back(new TQuantileMetric(ELossFunction::Quantile));
            }
            validParams = {"alpha"};
            break;
        }

        case ELossFunction::LogLinQuantile: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.emplace_back(new TLogLinQuantileMetric(FromString<float>(it->second)));
            } else {
                result.emplace_back(new TLogLinQuantileMetric());
            }
            validParams = {"alpha"};
            break;
        }

        case ELossFunction::QueryAverage: {
            auto it = params.find("top");
            CB_ENSURE(it != params.end(), "QueryAverage metric should have top parameter");
            result.emplace_back(new TQueryAverage(FromString<float>(it->second)));
            validParams = {"top"};
            break;
        }

        case ELossFunction::MAPE:
            result.emplace_back(new TMAPEMetric());
            break;

        case ELossFunction::Poisson:
            result.emplace_back(new TPoissonMetric());
            break;

        case ELossFunction::MedianAbsoluteError:
            result.emplace_back(new TMedianAbsoluteErrorMetric());
            break;

        case ELossFunction::SMAPE:
            result.emplace_back(new TSMAPEMetric());
            break;

        case ELossFunction::MSLE:
            result.emplace_back(new TMSLEMetric());
            break;

        case ELossFunction::MultiClass:
            result.emplace_back(new TMultiClassMetric());
            break;

        case ELossFunction::MultiClassOneVsAll:
            result.emplace_back(new TMultiClassOneVsAllMetric());
            break;

        case ELossFunction::PairLogit:
            result.emplace_back(new TPairLogitMetric());
            validParams = {"max_pairs"};
            break;

        case ELossFunction::PairLogitPairwise:
            result.emplace_back(new TPairLogitMetric());
            validParams = {"max_pairs"};
            break;

        case ELossFunction::QueryRMSE:
            result.emplace_back(new TQueryRMSEMetric());
            break;

        case ELossFunction::QuerySoftMax:
            result.emplace_back(new TQuerySoftMaxMetric());
            validParams = {"lambda"};
            break;

        case ELossFunction::YetiRank:
            result.emplace_back(new TPFoundMetric());
            validParams = {"decay", "permutations"};
            CB_ENSURE(!params.has("permutations") || FromString<int>(params.at("permutations")) > 0, "Metric " << metric << " expects permutations > 0");
            break;

        case ELossFunction::YetiRankPairwise:
            result.emplace_back(new TPFoundMetric());
            validParams = {"decay", "permutations"};
            CB_ENSURE(!params.has("permutations") || FromString<int>(params.at("permutations")) > 0, "Metric " << metric << " expects permutations > 0");
            break;

        case ELossFunction::PFound: {
            auto itTopSize = params.find("top");
            auto itDecay = params.find("decay");
            int topSize = itTopSize != params.end() ? FromString<int>(itTopSize->second) : -1;
            double decay = itDecay != params.end() ? FromString<double>(itDecay->second) : 0.85;
            result.emplace_back(new TPFoundMetric(topSize, decay));
            validParams = {"top", "decay"};
            break;
        }

        case ELossFunction::NDCG: {
            auto itTopSize = params.find("top");
            int topSize = -1;
            if (itTopSize != params.end()) {
                topSize = FromString<int>(itTopSize->second);
            }

            result.emplace_back(new TNDCGMetric(topSize));
            validParams = {"top"};
            break;
        }

        case ELossFunction::R2:
            result.emplace_back(new TR2Metric());
            break;

        case ELossFunction::AUC: {
            if (approxDimension == 1) {
                result.emplace_back(TAUCMetric::CreateBinClassMetric(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(TAUCMetric::CreateMultiClassMetric(i));
                }
            }
            break;
        }

        case ELossFunction::Accuracy:
            result.emplace_back(new TAccuracyMetric(border));
            validParams = {"border"};
            break;

        case ELossFunction::CtrFactor:
            result.emplace_back(new TCtrFactorMetric(border));
            validParams = {"border"};
            break;

        case ELossFunction::Precision: {
            if (approxDimension == 1) {
                result.emplace_back(TPrecisionMetric::CreateBinClassMetric(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(TPrecisionMetric::CreateMultiClassMetric(i));
                }
            }
            break;
        }

        case ELossFunction::Recall: {
            if (approxDimension == 1) {
                result.emplace_back(TRecallMetric::CreateBinClassMetric(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(TRecallMetric::CreateMultiClassMetric(i));
                }
            }
            break;
        }

        case ELossFunction::BalancedAccuracy: {
            CB_ENSURE(approxDimension == 1, "Balanced accuracy is used only for binary classification problems.");
            validParams = {"border"};
            result.emplace_back(TBalancedAccuracyMetric::CreateBinClassMetric(border));
            break;
        }

        case ELossFunction::BalancedErrorRate: {
            CB_ENSURE(approxDimension == 1, "Balanced Error Rate is used only for binary classification problems.");
            validParams = {"border"};
            result.emplace_back(TBalancedErrorRate::CreateBinClassMetric(border));
            break;
        }

        case ELossFunction::Kappa: {
            if (approxDimension == 1) {
                validParams = {"border"};
                result.emplace_back(TKappaMetric::CreateBinClassMetric(border));
            } else {
                result.emplace_back(TKappaMetric::CreateMultiClassMetric(approxDimension));
            }
            break;
        }

        case ELossFunction::WKappa: {
            if (approxDimension == 1) {
                validParams = {"border"};
                result.emplace_back(TWKappaMatric::CreateBinClassMetric(border));
            } else {
                result.emplace_back(TWKappaMatric::CreateMultiClassMetric(approxDimension));
            }
            break;
        }

        case ELossFunction::F1: {
            if (approxDimension == 1) {
                result.emplace_back(TF1Metric::CreateF1BinClass(border));
                validParams = {"border"};
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(TF1Metric::CreateF1Multiclass(i));
                }
            }
            break;
        }

        case ELossFunction::TotalF1:
            result.emplace_back(new TTotalF1Metric(approxDimension == 1 ? 2 : approxDimension));
            break;

        case ELossFunction::MCC:
            result.emplace_back(new TMCCMetric(approxDimension == 1 ? 2 : approxDimension));
            break;

        case ELossFunction::BrierScore:
            CB_ENSURE(approxDimension == 1, "Brier Score is used only for binary classification problems.");
            result.emplace_back(new TBrierScoreMetric());
            break;

        case ELossFunction::ZeroOneLoss:
            result.emplace_back(new TZeroOneLossMetric(border, approxDimension > 1));
            validParams = {"border"};
            break;

        case ELossFunction::HammingLoss:
            result.emplace_back(new THammingLossMetric(border, approxDimension > 1));
            validParams = {"border"};
            break;

        case ELossFunction::HingeLoss:
            result.emplace_back(new THingeLossMetric());
            break;

        case ELossFunction::PairAccuracy:
            result.emplace_back(new TPairAccuracyMetric());
            break;

        case ELossFunction::UserPerObjMetric: {
            result.emplace_back(new TUserDefinedPerObjectMetric(params));
            validParams = {"alpha"};
            break;
        }

        case ELossFunction::UserQuerywiseMetric: {
            result.emplace_back(new TUserDefinedQuerywiseMetric(params));
            validParams = {"alpha"};
            break;
        }
        case ELossFunction::QueryCrossEntropy: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.emplace_back(new TQueryCrossEntropyMetric(FromString<float>(it->second)));
            } else {
                result.emplace_back(new TQueryCrossEntropyMetric());
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

    if (params.has("hints")) { // TODO(smirnovpavel): hints shouldn't be added for each metric
        TMap<TString, TString> hints = ParseHintsDescription(params.at("hints"));
        for (const auto& hint : hints) {
            for (THolder<IMetric>& metric : result) {
                metric->AddHint(hint.first, hint.second);
            }
        }
    }

    if (params.has("use_weights")) {
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
        const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
        const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        int approxDimension
) {
    TVector<THolder<IMetric>> errors;

    if (evalMetricOptions->EvalMetric.IsSet()) {
        if (evalMetricOptions->EvalMetric->GetLossFunction() == ELossFunction::Custom) {
            errors.emplace_back(new TCustomMetric(*evalMetricDescriptor));
        } else {
            TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(evalMetricOptions->EvalMetric, approxDimension);
            CB_ENSURE(createdMetrics.size() == 1, "Eval metric should have a single value. Metric " <<
                ToString(evalMetricOptions->EvalMetric->GetLossFunction()) <<
                " provides a value for each class, thus it cannot be used as " <<
                "a single value to select best iteration or to detect overfitting. " <<
                "If you just want to look on the values of this metric use custom_metric parameter.");
            errors.push_back(std::move(createdMetrics.front()));
        }
    }

    if (lossFunctionOption->GetLossFunction() != ELossFunction::Custom) {
        TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(lossFunctionOption, approxDimension);
        for (auto& metric : createdMetrics) {
            errors.push_back(std::move(metric));
        }
    }

    for (const auto& description : evalMetricOptions->CustomMetrics.Get()) {
        TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(description, approxDimension);
        for (auto& metric : createdMetrics) {
            errors.push_back(std::move(metric));
        }
    }
    return errors;
}

TVector<TString> GetMetricsDescription(const TVector<const IMetric*>& metrics) {
    TVector<TString> result;
    for (const auto& metric : metrics) {
        result.push_back(metric->GetDescription());
    }
    return result;
}

TVector<bool> GetSkipMetricOnTrain(const TVector<const IMetric*>& metrics) {
    TVector<bool> result;
    for (const auto& metric : metrics) {
        const TMap<TString, TString>& hints = metric->GetHints();
        result.push_back(hints.has("skip_train") && hints.at("skip_train") == "true");
    }
    return result;
}

double EvalErrors(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        const THolder<IMetric>& error,
        NPar::TLocalExecutor* localExecutor
) {
    TMetricHolder metric;
    if (error->GetErrorType() == EErrorType::PerObjectError) {
        int begin = 0, end = target.ysize();
        Y_VERIFY(approx[0].ysize() == end - begin);
        metric = error->Eval(approx, target, weight, queriesInfo, begin, end, *localExecutor);
    } else {
        Y_VERIFY(error->GetErrorType() == EErrorType::QuerywiseError || error->GetErrorType() == EErrorType::PairwiseError);
        int queryStartIndex = 0, queryEndIndex = queriesInfo.ysize();
        metric = error->Eval(approx, target, weight, queriesInfo, queryStartIndex, queryEndIndex, *localExecutor);
    }
    return error->GetFinalError(metric);
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
                                                         const TVector<float>& target,
                                                         const TVector<float>& weight,
                                                         const TVector<TQueryInfo>& queriesInfo,
                                                         int queryStartIndex,
                                                         int queryEndIndex) const {
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
    return BuildDescription(ELossFunction::QueryCrossEntropy, UseWeights, alpha);
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
