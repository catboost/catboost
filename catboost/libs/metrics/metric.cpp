#include "metric.h"
#include "auc.h"
#include "doc_comparator.h"

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

static inline TString AddBorderIfNotDefault(const TString& description, double border) {
    if (border != GetDefaultClassificationBorder()) {
        return TStringBuilder() << description <<  ":border=" << border;
    } else {
        return description;
    }
}

EErrorType TMetric::GetErrorType() const {
    return EErrorType::PerObjectError;
}

double TMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Error / (error.Weight + 1e-38);
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

    TMetricHolder holder;
    const double* approxPtr = approxVec.data();
    const float* targetPtr = target.data();
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        const double approxExp = exp(approxPtr[i]);
        //this check should not be bottleneck
        const float prob = LossFunction == ELossFunction::Logloss ? targetPtr[i] > Border : targetPtr[i];
        holder.Error += w * (log(1 + approxExp) - prob * approxPtr[i]);
        holder.Weight += w;
    }
    return holder;
}

TString TCrossEntropyMetric::GetDescription() const {
    return AddBorderIfNotDefault(ToString(LossFunction), Border);
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

    TMetricHolder holder;
    const double* approxPtr = approxVec.data();
    const float* targetPtr = target.data();
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        const float targetVal = targetPtr[i] > Border;
        holder.Error += w * targetVal;

        const double approxExp = exp(approxPtr[i]);
        holder.Weight += w * approxExp / (approxExp + 1);
    }
    return holder;
}

TString TCtrFactorMetric::GetDescription() const {
    return ToString(ELossFunction::CtrFactor);
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

    TMetricHolder error;
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Error += Sqr(approxVec[k] - target[k]) * w;
        error.Weight += w;
    }
    return error;
}

double TRMSEMetric::GetFinalError(const TMetricHolder& error) const {
    return sqrt(error.Error / (error.Weight + 1e-38));
}

TString TRMSEMetric::GetDescription() const {
    return ToString(ELossFunction::RMSE);
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

    TMetricHolder error;
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        double val = target[i] - approxVec[i];
        double multiplier = (val > 0) ? Alpha : -(1 - Alpha);
        error.Error += (multiplier * val) * w;
        error.Weight += w;
    }
    if (LossFunction == ELossFunction::MAE) {
        error.Error *= 2;
    }
    return error;
}

TString TQuantileMetric::GetDescription() const {
    auto metricName = ToString(LossFunction);
    if (LossFunction == ELossFunction::Quantile) {
        return Sprintf("%s:alpha=%.3lf", metricName.c_str(), Alpha);
    } else {
        return metricName;
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

    TMetricHolder error;
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        double val = target[i] - exp(approxVec[i]);
        double multiplier = (val > 0) ? Alpha : -(1 - Alpha);
        error.Error += (multiplier * val) * w;
        error.Weight += w;
    }

    return error;
}

TString TLogLinQuantileMetric::GetDescription() const {
    auto metricName = ToString(ELossFunction::LogLinQuantile);
    return Sprintf("%s:alpha=%.3lf", metricName.c_str(), Alpha);
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

    TMetricHolder error;
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Error += Abs(1 - approxVec[k] / target[k]) * w;
        error.Weight += w;
    }

    return error;
}

TString TMAPEMetric::GetDescription() const {
    return ToString(ELossFunction::MAPE);
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

    TMetricHolder error;
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        error.Error += (exp(approxVec[i]) - target[i] * approxVec[i]) * w;
        error.Weight += w;
    }

    return error;
}

TString TPoissonMetric::GetDescription() const {
    return ToString(ELossFunction::Poisson);
}

void TPoissonMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
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

    TMetricHolder error;

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
        double targetClassApprox = approx[targetClass][k];

        float w = weight.empty() ? 1 : weight[k];
        error.Error += (targetClassApprox - maxApprox - log(sumExpApprox)) * w;
        error.Weight += w;
    }

    return error;
}

TString TMultiClassMetric::GetDescription() const {
    return ToString(ELossFunction::MultiClass);
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

    TMetricHolder error;
    for (int k = begin; k < end; ++k) {
        double sumDimErrors = 0;
        for (int dim = 0; dim < approxDimension; ++dim) {
            double expApprox = exp(approx[dim][k]);
            sumDimErrors += -log(1 + expApprox);
        }

        int targetClass = static_cast<int>(target[k]);
        sumDimErrors += approx[targetClass][k];

        float w = weight.empty() ? 1 : weight[k];
        error.Error += sumDimErrors / approxDimension * w;
        error.Weight += w;
    }
    return error;
}

TString TMultiClassOneVsAllMetric::GetDescription() const {
    return ToString(ELossFunction::MultiClassOneVsAll);
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

    TMetricHolder error;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int begin = queriesInfo[queryIndex].Begin;
        int end = queriesInfo[queryIndex].End;

        TVector<double> approxExpShifted(end - begin);
        for (int docId = begin; docId < end; ++docId) {
            approxExpShifted[docId - begin] = exp(approx[0][docId]);
        }

        for (int docId = 0; docId < queriesInfo[queryIndex].Competitors.ysize(); ++docId) {
            for (const auto& competitor : queriesInfo[queryIndex].Competitors[docId]) {
                error.Error += -competitor.Weight * log(approxExpShifted[docId] / (approxExpShifted[docId] + approxExpShifted[competitor.Id]));
                error.Weight += competitor.Weight;
            }
        }
    }
    return error;
}

EErrorType TPairLogitMetric::GetErrorType() const {
    return EErrorType::PairwiseError;
}

TString TPairLogitMetric::GetDescription() const {
    return ToString(ELossFunction::PairLogit);
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

    TMetricHolder error;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int begin = queriesInfo[queryIndex].Begin;
        int end = queriesInfo[queryIndex].End;
        double queryAvrg = CalcQueryAvrg(begin, end - begin, approx[0], target, weight);
        for (int docId = begin; docId < end; ++docId) {
            float w = weight.empty() ? 1 : weight[docId];
            error.Error += (Sqr(target[docId] - approx[0][docId] - queryAvrg)) * w;
            error.Weight += w;
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
    return ToString(ELossFunction::QueryRMSE);
}

double TQueryRMSEMetric::GetFinalError(const TMetricHolder& error) const {
    return sqrt(error.Error / (error.Weight + 1e-38));
}

void TQueryRMSEMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* PFound */

TPFoundMetric::TPFoundMetric(int topSize, double decay)
        : TopSize(topSize)
        , Decay(decay)
    {
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
        if (!queriesInfo[queryIndex].SubgroupId.empty()) {
            subgroupIdData = queriesInfo[queryIndex].SubgroupId.data();
        }
        calcer.AddQuery(target.data() + queryBegin, approx[0].data() + queryBegin, subgroupIdData, queryEnd - queryBegin);
    }
    return calcer.GetMetric();
}

EErrorType TPFoundMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TPFoundMetric::GetDescription() const {
    auto metricName = ToString(ELossFunction::PFound);
    TString topInfo = (TopSize == -1 ? "" : "top=" + ToString(TopSize));
    TString decayInfo = (Decay == 0.85 ? "" : "decay=" + ToString(Decay));
    if (topInfo != "" && decayInfo != "") {
        metricName += ":" + topInfo + ";" + decayInfo;
    } else if (topInfo != "" || decayInfo != "") {
        metricName += ":" + topInfo + decayInfo;
    }
    return metricName;
}

double TPFoundMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Weight > 0 ? error.Error / error.Weight : 0;
}

void TPFoundMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
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

    TMetricHolder error;
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

    TMetricHolder error;
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
                    error.Error -= weights[start + dim] * targets[start + dim] * log((*softmax)[dim] / sumExpApprox);
                }
            }
        } else {
            for (int dim = 0; dim < count; ++dim) {
                if (targets[start + dim] > 0) {
                    error.Error -= targets[start + dim] * log((*softmax)[dim] / sumExpApprox);
                }
            }
        }
        error.Weight = sumWeightedTargets;
    }
    return error;
}

TString TQuerySoftMaxMetric::GetDescription() const {
    return ToString(ELossFunction::QuerySoftMax);
}

void TQuerySoftMaxMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Min;
}

/* R2 */

TMetricHolder TR2Metric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    CB_ENSURE(approx.size() == 1, "Metric R2 supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    double avrgTarget = Accumulate(approxVec.begin() + begin, approxVec.begin() + end, 0.0);
    Y_ASSERT(begin < end);
    avrgTarget /= end - begin;

    double mse = 0;
    double targetVariance = 0;

    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        mse += Sqr(approxVec[k] - target[k]) * w;
        targetVariance += Sqr(target[k] - avrgTarget) * w;
    }
    TMetricHolder error;
    error.Error = 1 - mse / targetVariance;
    error.Weight = 1;
    return error;
}

double TR2Metric::GetFinalError(const TMetricHolder& error) const {
    return error.Error;
}

TString TR2Metric::GetDescription() const {
    return ToString(ELossFunction::R2);
}

void TR2Metric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Classification helpers */

static int GetApproxClass(const TVector<TVector<double>>& approx, int docIdx) {
    if (approx.ysize() == 1) {
        return approx[0][docIdx] > 0.0;
    } else {
        double maxApprox = approx[0][docIdx];
        int maxApproxIndex = 0;

        for (int dim = 1; dim < approx.ysize(); ++dim) {
            if (approx[dim][docIdx] > maxApprox) {
                maxApprox = approx[dim][docIdx];
                maxApproxIndex = dim;
            }
        }
        return maxApproxIndex;
    }
}

static void GetPositiveStats(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    int begin,
    int end,
    int positiveClass,
    double border,
    double* truePositive,
    double* targetPositive,
    double* approxPositive
) {
    *truePositive = 0;
    *targetPositive = 0;
    *approxPositive = 0;
    const bool isMulticlass = approx.size() > 1;
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        const float targetVal = isMulticlass ? target[i] : target[i] > border;
        int targetClass = static_cast<int>(targetVal);
        float w = weight.empty() ? 1 : weight[i];

        if (targetClass == positiveClass) {
            *targetPositive += w;
            if (approxClass == positiveClass) {
                *truePositive += w;
            }
        }
        if (approxClass == positiveClass) {
            *approxPositive += w;
        }
    }
}

static void GetTotalPositiveStats(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    int begin,
    int end,
    TVector<double>* truePositive,
    TVector<double>* targetPositive,
    TVector<double>* approxPositive
) {
    int classesCount = approx.ysize() == 1 ? 2 : approx.ysize();
    truePositive->assign(classesCount, 0);
    targetPositive->assign(classesCount, 0);
    approxPositive->assign(classesCount, 0);
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        int targetClass = static_cast<int>(target[i]);
        float w = weight.empty() ? 1 : weight[i];

        if (approxClass == targetClass) {
            truePositive->at(targetClass) += w;
        }
        targetPositive->at(targetClass) += w;
        approxPositive->at(approxClass) += w;
    }
}

/* AUC */

TAUCMetric::TAUCMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TMetricHolder TAUCMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    const auto& approxVec = approx.ysize() == 1 ? approx.front() : approx[PositiveClass];
    Y_ASSERT(approxVec.size() == target.size());

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

    TMetricHolder error;
    error.Error = CalcAUC(&samples);
    error.Weight = 1.0;
    return error;
}

TString TAUCMetric::GetDescription() const {
    if (IsMultiClass) {
        return Sprintf("%s:class=%d", ToString(ELossFunction::AUC).c_str(), PositiveClass);
    } else {
        return AddBorderIfNotDefault(ToString(ELossFunction::AUC), Border);
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
    const bool isMulticlass = approx.size() > 1;

    TMetricHolder error;
    for (int k = begin; k < end; ++k) {
        int approxClass = GetApproxClass(approx, k);
        const float targetVal = isMulticlass ? target[k] : target[k] > Border;
        int targetClass = static_cast<int>(targetVal);

        float w = weight.empty() ? 1 : weight[k];
        error.Error += approxClass == targetClass ? w : 0.0;
        error.Weight += w;
    }
    return error;
}

// TODO(annaveronika): write border in description if differs from default.
TString TAccuracyMetric::GetDescription() const {
    return ToString(ELossFunction::Accuracy);
}

void TAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Precision */

TPrecisionMetric::TPrecisionMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TMetricHolder TPrecisionMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;

    GetPositiveStats(approx, target, weight, begin, end, PositiveClass, Border,
        &truePositive, &targetPositive, &approxPositive);

    TMetricHolder error;
    error.Error = approxPositive > 0 ? truePositive / approxPositive : 0;
    error.Weight = 1;
    return error;
}

TString TPrecisionMetric::GetDescription() const {
    if (IsMultiClass) {
        return Sprintf("%s:class=%d", ToString(ELossFunction::Precision).c_str(), PositiveClass);
    } else {
        return AddBorderIfNotDefault(ToString(ELossFunction::Precision), Border);
    }
}

void TPrecisionMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Recall */

TRecallMetric::TRecallMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TMetricHolder TRecallMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(
        approx,
        target,
        weight,
        begin,
        end,
        PositiveClass,
        Border,
        &truePositive,
        &targetPositive,
        &approxPositive
    );

    TMetricHolder error;
    error.Error = targetPositive > 0 ? truePositive / targetPositive : 0;
    error.Weight = 1;
    return error;
}

TString TRecallMetric::GetDescription() const {
    if (IsMultiClass) {
        return Sprintf("%s:class=%d", ToString(ELossFunction::Recall).c_str(), PositiveClass);
    } else {
        return AddBorderIfNotDefault(ToString(ELossFunction::Recall), Border);
    }
}

void TRecallMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
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

TMetricHolder TF1Metric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass, Border,
                     &truePositive, &targetPositive, &approxPositive);

    TMetricHolder error;
    double denominator = targetPositive + approxPositive;
    error.Error = denominator > 0 ? 2 * truePositive / denominator : 0;
    error.Weight = 1;
    return error;
}
TString TF1Metric::GetDescription() const {
    if (IsMultiClass) {
        return Sprintf("%s:class=%d", ToString(ELossFunction::F1).c_str(), PositiveClass);
    } else {
        return AddBorderIfNotDefault(ToString(ELossFunction::F1), Border);
    }
}

void TF1Metric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* TotalF1 */

TMetricHolder TTotalF1Metric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    TVector<double> truePositive;
    TVector<double> targetPositive;
    TVector<double> approxPositive;
    GetTotalPositiveStats(approx, target, weight, begin, end,
                          &truePositive, &targetPositive, &approxPositive);

    int classesCount = truePositive.ysize();
    TMetricHolder error;
    for (int classIdx = 0; classIdx < classesCount; ++classIdx) {
        double denominator = targetPositive[classIdx] + approxPositive[classIdx];
        error.Error += denominator > 0 ? 2 * truePositive[classIdx] / denominator * targetPositive[classIdx] : 0;
        error.Weight += targetPositive[classIdx];
    }
    return error;
}

TString TTotalF1Metric::GetDescription() const {
    return ToString(ELossFunction::TotalF1);
}

void TTotalF1Metric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Confusion matrix */

static TVector<TVector<double>> GetConfusionMatrix(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    int begin,
    int end
) {
    int classesCount = approx.ysize() == 1 ? 2 : approx.ysize();
    TVector<TVector<double>> confusionMatrix(classesCount, TVector<double>(classesCount, 0));
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        int targetClass = static_cast<int>(target[i]);
        float w = weight.empty() ? 1 : weight[i];
        confusionMatrix[approxClass][targetClass] += w;
    }
    return confusionMatrix;
}


/* MCC */

TMetricHolder TMCCMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    TVector<TVector<double>> confusionMatrix = GetConfusionMatrix(approx, target, weight, begin, end);
    int classesCount = confusionMatrix.ysize();

    TVector<double> rowSum(classesCount, 0);
    TVector<double> columnSum(classesCount, 0);
    double totalSum = 0;
    for (int approxClass = 0; approxClass < classesCount; ++approxClass) {
        for (int tragetClass = 0; tragetClass < classesCount; ++tragetClass) {
            rowSum[approxClass] += confusionMatrix[approxClass][tragetClass];
            columnSum[tragetClass] += confusionMatrix[approxClass][tragetClass];
            totalSum += confusionMatrix[approxClass][tragetClass];
        }
    }

    double numerator = 0;
    for (int classIdx = 0; classIdx < classesCount; ++classIdx) {
        numerator += confusionMatrix[classIdx][classIdx] * totalSum - rowSum[classIdx] * columnSum[classIdx];
    }

    double sumSquareRowSums = 0;
    double sumSquareColumnSums = 0;
    for (int classIdx = 0; classIdx < classesCount; ++classIdx) {
        sumSquareRowSums += Sqr(rowSum[classIdx]);
        sumSquareColumnSums += Sqr(columnSum[classIdx]);
    }

    double denominator = sqrt((Sqr(totalSum) - sumSquareRowSums) * (Sqr(totalSum) - sumSquareColumnSums));

    TMetricHolder error;
    error.Error = numerator / (denominator + FLT_EPSILON);
    error.Weight = 1;
    return error;
}

TString TMCCMetric::GetDescription() const {
    return ToString(ELossFunction::MCC);
}

void TMCCMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
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

    TMetricHolder error;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        int begin = queriesInfo[queryIndex].Begin;
        for (int docId = 0; docId < queriesInfo[queryIndex].Competitors.ysize(); ++docId) {
            for (const auto& competitor : queriesInfo[queryIndex].Competitors[docId]) {
                if (approx[0][begin + docId] > approx[0][begin + competitor.Id]) {
                    error.Error += competitor.Weight;
                }
                error.Weight += competitor.Weight;
            }
        }
    }
    return error;
}

EErrorType TPairAccuracyMetric::GetErrorType() const {
    return EErrorType::PairwiseError;
}

TString TPairAccuracyMetric::GetDescription() const {
    return ToString(ELossFunction::PairAccuracy);
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
    const TVector<float>& weight,
    const TVector<TQueryInfo>& /*queriesInfo*/,
    int begin,
    int end,
    NPar::TLocalExecutor& /* executor */
) const {
    return Descriptor.EvalFunc(approx, target, weight, begin, end, Descriptor.CustomData);
}

TString TCustomMetric::GetDescription() const {
    return Descriptor.GetDescriptionFunc(Descriptor.CustomData);
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

/* UserDefinedPerObjectMetric */

TUserDefinedPerObjectMetric::TUserDefinedPerObjectMetric(const TMap<TString, TString>& params)
    : Alpha(0.0)
{
    if (params.has("alpha")) {
        Alpha = FromString<float>(params.at("alpha"));
    }
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
    TMetricHolder metric;
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
    TMetricHolder metric;
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
    const TVector<float>& weight,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approx.size() == 1, "Metric QueryAverage supports only single-dimensional data");
    Y_UNUSED(weight);

    TMetricHolder error;

    TVector<std::pair<double, int>> approxWithDoc;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        auto startIdx = queriesInfo[queryIndex].Begin;
        auto endIdx = queriesInfo[queryIndex].End;
        auto querySize = endIdx - startIdx;

        double targetSum = 0;
        if ((int)querySize <= TopSize) {
            for (int docId = startIdx; docId < endIdx; ++docId) {
                targetSum += target[docId];
            }
            error.Error += targetSum / querySize;
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
            error.Error += targetSum / TopSize;
        }
        error.Weight += 1;
    }
    return error;
}

EErrorType TQueryAverage::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TQueryAverage::GetDescription() const {
    auto metricName = ToString(ELossFunction::QueryAverage);
    return Sprintf("%s:top=%d", metricName.c_str(), TopSize);
}

void TQueryAverage::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Create */

static TVector<THolder<IMetric>> CreateMetric(ELossFunction metric, const TMap<TString, TString>& params, int approxDimension) {
    TSet<ELossFunction> metricsWithParams = {
        ELossFunction::Quantile,
        ELossFunction::LogLinQuantile,
        ELossFunction::QueryAverage,
        ELossFunction::Logloss,
        ELossFunction::AUC,
        ELossFunction::Precision,
        ELossFunction::Accuracy,
        ELossFunction::F1,
        ELossFunction::TotalF1,
        ELossFunction::PFound,
        ELossFunction::CtrFactor,
        ELossFunction::YetiRank
    };
    if (!metricsWithParams.has(metric)) {
        CB_ENSURE(params.empty(), "Metric " + ToString(metric) + " does not have any params");
    }

    double border = GetDefaultClassificationBorder();
    if (params.has("border")) {
        border = FromString<float>(params.at("border"));
    }

    TVector<THolder<IMetric>> result;
    switch (metric) {
        case ELossFunction::Logloss:
            result.emplace_back(new TCrossEntropyMetric(ELossFunction::Logloss, border));
            return result;

        case ELossFunction::CrossEntropy:
            result.emplace_back(new TCrossEntropyMetric(ELossFunction::CrossEntropy));
            return result;

        case ELossFunction::RMSE:
            result.emplace_back(new TRMSEMetric());
            return result;

        case ELossFunction::MAE:
            result.emplace_back(new TQuantileMetric(ELossFunction::MAE));
            return result;

        case ELossFunction::Quantile: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.emplace_back(new TQuantileMetric(ELossFunction::Quantile, FromString<float>(it->second)));
            } else {
                result.emplace_back(new TQuantileMetric(ELossFunction::Quantile));
            }
            return result;
        }

        case ELossFunction::LogLinQuantile: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.emplace_back(new TLogLinQuantileMetric(FromString<float>(it->second)));
            } else {
                result.emplace_back(new TLogLinQuantileMetric());
            }
            return result;
        }

        case ELossFunction::QueryAverage: {
            auto it = params.find("top");
            CB_ENSURE(it != params.end(), "QueryAverage metric should have top parameter");
            result.emplace_back(new TQueryAverage(FromString<float>(it->second)));
            return result;
        }

        case ELossFunction::MAPE:
            result.emplace_back(new TMAPEMetric());
            return result;

        case ELossFunction::Poisson:
            result.emplace_back(new TPoissonMetric());
            return result;

        case ELossFunction::MultiClass:
            result.emplace_back(new TMultiClassMetric());
            return result;

        case ELossFunction::MultiClassOneVsAll:
            result.emplace_back(new TMultiClassOneVsAllMetric());
            return result;

        case ELossFunction::PairLogit:
            result.emplace_back(new TPairLogitMetric());
            return result;

        case ELossFunction::QueryRMSE:
            result.emplace_back(new TQueryRMSEMetric());
            return result;

        case ELossFunction::QuerySoftMax:
            result.emplace_back(new TQuerySoftMaxMetric());
            return result;

        case ELossFunction::YetiRank:
            result.emplace_back(new TPFoundMetric());
            return result;

        case ELossFunction::PFound: {
            auto itTopSize = params.find("top");
            auto itDecay = params.find("decay");
            int topSize = itTopSize != params.end() ? FromString<int>(itTopSize->second) : -1;
            double decay = itDecay != params.end() ? FromString<double>(itDecay->second) : 0.85;
            result.emplace_back(new TPFoundMetric(topSize, decay));
            return result;
        }

        case ELossFunction::R2:
            result.emplace_back(new TR2Metric());
            return result;

        case ELossFunction::AUC: {
            if (approxDimension == 1) {
                result.emplace_back(new TAUCMetric(border));
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TAUCMetric(i));
                }
            }
            return result;
        }

        case ELossFunction::Accuracy:
            result.emplace_back(new TAccuracyMetric(border));
            return result;

        case ELossFunction::CtrFactor:
            result.emplace_back(new TCtrFactorMetric(border));
            return result;

        case ELossFunction::Precision: {
            if (approxDimension == 1) {
                result.emplace_back(new TPrecisionMetric(border));
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TPrecisionMetric(i));
                }
            }
            return result;
        }

        case ELossFunction::Recall: {
            if (approxDimension == 1) {
                result.emplace_back(new TRecallMetric(border));
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TRecallMetric(i));
                }
            }
            return result;
        }

        case ELossFunction::F1: {
            if (approxDimension == 1) {
                result.emplace_back(TF1Metric::CreateF1BinClass(border));
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(TF1Metric::CreateF1Multiclass(i));
                }
            }
            return result;
        }

        case ELossFunction::TotalF1:
            result.emplace_back(new TTotalF1Metric());
            return result;

        case ELossFunction::MCC:
            result.emplace_back(new TMCCMetric());
            return result;

        case ELossFunction::PairAccuracy:
            result.emplace_back(new TPairAccuracyMetric());
            return result;

        case ELossFunction::UserPerObjMetric: {
            result.emplace_back(new TUserDefinedPerObjectMetric(params));
            return result;
        }

        case ELossFunction::UserQuerywiseMetric: {
            result.emplace_back(new TUserDefinedQuerywiseMetric(params));
            return result;
        }

        default:
            Y_ASSERT(false);
            return TVector<THolder<IMetric>>();
    }
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

static TVector<THolder<IMetric>> CreateMetricFromDescription(const NCatboostOptions::TLossDescription& description, int approxDimension) {
    auto metric = description.GetLossFunction();
    return CreateMetric(metric, description.GetLossParams(), approxDimension);
}

TVector<THolder<IMetric>> CreateMetrics(
    const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
    const NCatboostOptions::TCpuOnlyOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    int approxDimension
) {
    TVector<THolder<IMetric>> errors;

    if (evalMetricOptions->EvalMetric.IsSet()) {
        if (evalMetricOptions->EvalMetric->GetLossFunction() == ELossFunction::Custom) {
            errors.emplace_back(new TCustomMetric(*evalMetricDescriptor));
        } else {
            TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(evalMetricOptions->EvalMetric, approxDimension);
            for (auto& metric : createdMetrics) {
                errors.push_back(std::move(metric));
            }
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

TVector<TString> GetMetricsDescription(const TVector<THolder<IMetric>>& metrics) {
     TVector<TString> result;
     for (const auto& metric : metrics) {
         result.push_back(metric->GetDescription());
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
        auto avrgApprox = approx;
        int begin = 0, end = target.ysize();
        Y_VERIFY(approx[0].ysize() == end - begin);
        metric = error->Eval(avrgApprox, target, weight, queriesInfo, begin, end, *localExecutor);
    } else {
        Y_VERIFY(error->GetErrorType() == EErrorType::QuerywiseError || error->GetErrorType() == EErrorType::PairwiseError);
        auto avrgApprox = approx;
        int queryStartIndex = 0, queryEndIndex = queriesInfo.ysize();
        metric = error->Eval(avrgApprox, target, weight, queriesInfo, queryStartIndex, queryEndIndex, *localExecutor);
    }
    return error->GetFinalError(metric);
}
