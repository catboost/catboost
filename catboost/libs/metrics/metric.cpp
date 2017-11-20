#include "metric.h"
#include "auc.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/ymath.h>
#include <util/generic/string.h>
#include <util/generic/maybe.h>
#include <util/string/iterator.h>
#include <util/string/cast.h>
#include <util/string/printf.h>
#include <util/system/yassert.h>

#include <limits>

/* TMetric */

TMetricHolder TMetric::EvalPairwise(const TVector<TVector<double>>& /*approx*/,
                                   const TVector<TPair>& /*pairs*/,
                                   int /*begin*/, int /*end*/) const {
    CB_ENSURE(false, "This eval is only for Pairwise");
}

TMetricHolder TMetric::EvalQuerywise(const TVector<TVector<double>>& /*approx*/,
                                    const TVector<float>& /*target*/,
                                    const TVector<float>& /*weight*/,
                                    const TVector<ui32>& /*queriesId*/,
                                    const yhash<ui32, ui32>& /*queriesSize*/,
                                    int /*begin*/, int /*end*/) const {
    CB_ENSURE(false, "This eval is only for Querywise");
}

EErrorType TMetric::GetErrorType() const {
    return EErrorType::PerObjectError;
}

double TMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Error / (error.Weight + 1e-38);
}

/* TPairMetric */

TMetricHolder TPairwiseMetric::Eval(const TVector<TVector<double>>& /*approx*/,
                                   const TVector<float>& /*target*/,
                                   const TVector<float>& /*weight*/,
                                   int /*begin*/, int /*end*/,
                                   NPar::TLocalExecutor& /*executor*/) const {
    CB_ENSURE(false, "This eval is not for Pairwise");
}

TMetricHolder TPairwiseMetric::EvalQuerywise(const TVector<TVector<double>>& /*approx*/,
                                            const TVector<float>& /*target*/,
                                            const TVector<float>& /*weight*/,
                                            const TVector<ui32>& /*queriesId*/,
                                            const yhash<ui32, ui32>& /*queriesSize*/,
                                            int /*begin*/, int /*end*/) const {
    CB_ENSURE(false, "This eval is not for Pairwise");
}

EErrorType TPairwiseMetric::GetErrorType() const {
    return EErrorType::PairwiseError;
}

double TPairwiseMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Error / (error.Weight + 1e-38);
}

/* TQueryMetric */

TMetricHolder TQuerywiseMetric::Eval(const TVector<TVector<double>>& /*approx*/,
                                    const TVector<float>& /*target*/,
                                    const TVector<float>& /*weight*/,
                                    int /*begin*/, int /*end*/,
                                    NPar::TLocalExecutor& /*executor*/) const {
    CB_ENSURE(false, "This eval is not for Querywise");
}

TMetricHolder TQuerywiseMetric::EvalPairwise(const TVector<TVector<double>>& /*approx*/,
                                   const TVector<TPair>& /*pairs*/,
                                   int /*begin*/, int /*end*/) const {
    CB_ENSURE(false, "This eval is not for Querywise");
}

EErrorType TQuerywiseMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TQuerywiseMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Error / (error.Weight + 1e-38);
}

/* CrossEntropy */

TCrossEntropyMetric::TCrossEntropyMetric(ELossFunction lossFunction)
    : LossFunction(lossFunction)
{
    Y_ASSERT(lossFunction == ELossFunction::Logloss || lossFunction == ELossFunction::CrossEntropy);
}

TMetricHolder TCrossEntropyMetric::Eval(const TVector<TVector<double>>& approx,
                                       const TVector<float>& target,
                                       const TVector<float>& weight,
                                       int begin, int end,
                                       NPar::TLocalExecutor& executor) const {
    // p * log(1/(1+exp(-f))) + (1-p) * log(1 - 1/(1+exp(-f))) =
    // p * log(exp(f) / (exp(f) + 1)) + (1-p) * log(exp(-f)/(1+exp(-f))) =
    // p * log(exp(f) / (exp(f) + 1)) + (1-p) * log(1/(exp(f) + 1)) =
    // p * (log(val) - log(val + 1)) + (1-p) * (-log(val + 1)) =
    // p*log(val) - p*log(val+1) - log(val+1) + p*log(val+1) =
    // p*log(val) - log(val+1)

    CB_ENSURE(approx.size() == 1, "Metric logloss supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    NPar::TLocalExecutor::TBlockParams blockParams(begin, end);
    blockParams.SetBlockCount(executor.GetThreadCount() + 1);
    TVector<TMetricHolder> errorHolders(blockParams.GetBlockCount());

    executor.ExecRange([&](int blockId) {
        TMetricHolder holder;
        const double* approxPtr = approxVec.data();
        const float* targetPtr = target.data();

        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [approxPtr, targetPtr, &holder, &weight](int i) {
            float w = weight.empty() ? 1 : weight[i];
            const double approxExp = exp(approxPtr[i]);
            const float prob = targetPtr[i];
            holder.Error += w * (log(1 + approxExp) - prob * approxPtr[i]);
            holder.Weight += w;
        })(blockId);

        errorHolders[blockId] = holder;
    },
                       0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    TMetricHolder error;
    for (const auto& eh : errorHolders) {
        error.Add(eh);
    }

    return error;
}

TString TCrossEntropyMetric::GetDescription() const {
    return ToString(LossFunction);
}

bool TCrossEntropyMetric::IsMaxOptimal() const {
    return false;
}

/* RMSE */

TMetricHolder TRMSEMetric::Eval(const TVector<TVector<double>>& approx,
                               const TVector<float>& target,
                               const TVector<float>& weight,
                               int begin, int end,
                               NPar::TLocalExecutor& /* executor */) const {
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

bool TRMSEMetric::IsMaxOptimal() const {
    return false;
}

/* Quantile */

TQuantileMetric::TQuantileMetric(ELossFunction lossFunction, double alpha)
    : LossFunction(lossFunction)
    , Alpha(alpha)
{
    Y_ASSERT(lossFunction == ELossFunction::Quantile || lossFunction == ELossFunction::MAE);
    CB_ENSURE(lossFunction == ELossFunction::Quantile || alpha == 0.5, "Alpha parameter should not be used for MAE loss");
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6,
              "Alpha parameter for quantile metric should be in interval [0, 1]");
}

TMetricHolder TQuantileMetric::Eval(const TVector<TVector<double>>& approx,
                                   const TVector<float>& target,
                                   const TVector<float>& weight,
                                   int begin, int end,
                                   NPar::TLocalExecutor& /* executor */) const {
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

bool TQuantileMetric::IsMaxOptimal() const {
    return false;
}

/* LogLinQuantile */

TLogLinQuantileMetric::TLogLinQuantileMetric(double alpha)
    : Alpha(alpha)
{
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6,
              "Alpha parameter for log-linear quantile metric should be in interval (0, 1)");
}

TMetricHolder TLogLinQuantileMetric::Eval(const TVector<TVector<double>>& approx,
                                         const TVector<float>& target,
                                         const TVector<float>& weight,
                                         int begin, int end,
                                         NPar::TLocalExecutor& /* executor */) const {
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

bool TLogLinQuantileMetric::IsMaxOptimal() const {
    return false;
}

/* MAPE */

TMetricHolder TMAPEMetric::Eval(const TVector<TVector<double>>& approx,
                               const TVector<float>& target,
                               const TVector<float>& weight,
                               int begin, int end,
                               NPar::TLocalExecutor& /* executor */) const {
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

bool TMAPEMetric::IsMaxOptimal() const {
    return false;
}

/* Poisson */

TMetricHolder TPoissonMetric::Eval(const TVector<TVector<double>>& approx,
                                  const TVector<float>& target,
                                  const TVector<float>& weight,
                                  int begin, int end,
                                  NPar::TLocalExecutor& /* executor */) const {
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

bool TPoissonMetric::IsMaxOptimal() const {
    return false;
}

/* MultiClass */

TMetricHolder TMultiClassMetric::Eval(const TVector<TVector<double>>& approx,
                                     const TVector<float>& target,
                                     const TVector<float>& weight,
                                     int begin, int end,
                                     NPar::TLocalExecutor& /* executor */) const {
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

bool TMultiClassMetric::IsMaxOptimal() const {
    return true;
}

/* MultiClassOneVsAll */

TMetricHolder TMultiClassOneVsAllMetric::Eval(const TVector<TVector<double>>& approx,
                                             const TVector<float>& target,
                                             const TVector<float>& weight,
                                             int begin, int end,
                                             NPar::TLocalExecutor& /* executor */) const {
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

bool TMultiClassOneVsAllMetric::IsMaxOptimal() const {
    return true;
}

/* PairLogit */

TMetricHolder TPairLogitMetric::EvalPairwise(const TVector<TVector<double>>& approx,
                                            const TVector<TPair>& pairs,
                                            int begin, int end) const {
    CB_ENSURE(approx.size() == 1, "Metric PairLogit supports only single-dimensional data");

    TVector<double> approxExpShifted(end - begin);
    for (int docId = begin; docId < end; ++docId) {
        approxExpShifted[docId - begin] = exp(approx[0][docId]);
    }

    TMetricHolder error;
    for (const auto& pair : pairs) {
        if (pair.WinnerId < begin || pair.WinnerId >= end ||
            pair.LoserId < begin || pair.LoserId >= end) {
            continue;
        }

        float w = pair.Weight;
        double expWinner = approxExpShifted[pair.WinnerId - begin];
        double expLoser = approxExpShifted[pair.LoserId - begin];
        error.Error += -log(expWinner / (expWinner + expLoser));
        error.Weight += w;
    }
    return error;
}

TString TPairLogitMetric::GetDescription() const {
    return ToString(ELossFunction::PairLogit);
}

bool TPairLogitMetric::IsMaxOptimal() const {
    return false;
}

/* QueryRMSE */

TMetricHolder TQueryRMSEMetric::EvalQuerywise(const TVector<TVector<double>>& approx,
                                         const TVector<float>& target,
                                         const TVector<float>& weight,
                                         const TVector<ui32>& queriesId,
                                         const yhash<ui32, ui32>& queriesSize,
                                         int begin, int end) const {
    CB_ENSURE(approx.size() == 1, "Metric QueryRMSE supports only single-dimensional data");

    int offset = 0;
    TMetricHolder error;
    while (begin + offset < end) {
        ui32 querySize = queriesSize.find(queriesId[begin + offset])->second;
        double queryAvrg = CalcQueryAvrg(begin + offset, querySize, approx[0], target, weight);
        double queryAvrgSqr = Sqr(queryAvrg);

        for (ui32 docId = begin + offset; docId < begin + offset + querySize; ++docId) {
            float w = weight.empty() ? 1 : weight[docId];
            error.Error += (Sqr(target[docId] - approx[0][docId]) - queryAvrgSqr) * w;
            error.Weight += w;
        }
        offset += querySize;
    }

    return error;
}

double TQueryRMSEMetric::CalcQueryAvrg(int start, int count,
                     const TVector<double>& approxes,
                     const TVector<float>& targets,
                     const TVector<float>& weights) const {
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

TString TQueryRMSEMetric::GetDescription() const {
    return ToString(ELossFunction::QueryRMSE);
}

bool TQueryRMSEMetric::IsMaxOptimal() const {
    return false;
}

/* R2 */

TMetricHolder TR2Metric::Eval(const TVector<TVector<double>>& approx,
                             const TVector<float>& target,
                             const TVector<float>& weight,
                             int begin, int end,
                             NPar::TLocalExecutor& /* executor */) const {
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

bool TR2Metric::IsMaxOptimal() const {
    return true;
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

static void GetPositiveStats(const TVector<TVector<double>>& approx,
                             const TVector<float>& target,
                             const TVector<float>& weight,
                             int begin, int end,
                             int positiveClass,
                             double* truePositive,
                             double* targetPositive,
                             double* approxPositive) {
    *truePositive = 0;
    *targetPositive = 0;
    *approxPositive = 0;
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        int targetClass = static_cast<int>(target[i]);
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

static void GetTotalPositiveStats(const TVector<TVector<double>>& approx,
                                  const TVector<float>& target,
                                  const TVector<float>& weight,
                                  int begin, int end,
                                  TVector<double>* truePositive,
                                  TVector<double>* targetPositive,
                                  TVector<double>* approxPositive) {
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

TMetricHolder TAUCMetric::Eval(const TVector<TVector<double>>& approx,
                                 const TVector<float>& target,
                                 const TVector<float>& weight,
                                 int begin, int end,
                                 NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    const auto& approxVec = approx.ysize() == 1 ? approx.front() : approx[PositiveClass];
    Y_ASSERT(approxVec.size() == target.size());

    TVector<double> approxCopy(approxVec.begin() + begin, approxVec.begin() + end);
    TVector<double> targetCopy(target.begin() + begin, target.begin() + end);

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
        return ToString(ELossFunction::AUC);
    }
}

bool TAUCMetric::IsMaxOptimal() const {
    return true;
}

/* Accuracy */

TMetricHolder TAccuracyMetric::Eval(const TVector<TVector<double>>& approx,
                                   const TVector<float>& target,
                                   const TVector<float>& weight,
                                   int begin, int end,
                                   NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());

    TMetricHolder error;
    for (int k = begin; k < end; ++k) {
        int approxClass = GetApproxClass(approx, k);
        int targetClass = static_cast<int>(target[k]);

        float w = weight.empty() ? 1 : weight[k];
        error.Error += approxClass == targetClass ? w : 0.0;
        error.Weight += w;
    }
    return error;
}

TString TAccuracyMetric::GetDescription() const {
    return ToString(ELossFunction::Accuracy);
}

bool TAccuracyMetric::IsMaxOptimal() const {
    return true;
}

/* Precision */

TPrecisionMetric::TPrecisionMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TMetricHolder TPrecisionMetric::Eval(const TVector<TVector<double>>& approx,
                                    const TVector<float>& target,
                                    const TVector<float>& weight,
                                    int begin, int end,
                                    NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass,
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
        return ToString(ELossFunction::Precision);
    }
}
bool TPrecisionMetric::IsMaxOptimal() const {
    return true;
}

/* Recall */

TRecallMetric::TRecallMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TMetricHolder TRecallMetric::Eval(const TVector<TVector<double>>& approx,
                                 const TVector<float>& target,
                                 const TVector<float>& weight,
                                 int begin, int end,
                                 NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass,
                     &truePositive, &targetPositive, &approxPositive);

    TMetricHolder error;
    error.Error = targetPositive > 0 ? truePositive / targetPositive : 0;
    error.Weight = 1;
    return error;
}
TString TRecallMetric::GetDescription() const {
    if (IsMultiClass) {
        return Sprintf("%s:class=%d", ToString(ELossFunction::Recall).c_str(), PositiveClass);
    } else {
        return ToString(ELossFunction::Recall);
    }
}
bool TRecallMetric::IsMaxOptimal() const {
    return true;
}

/* F1 */

TF1Metric::TF1Metric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TMetricHolder TF1Metric::Eval(const TVector<TVector<double>>& approx,
                             const TVector<float>& target,
                             const TVector<float>& weight,
                             int begin, int end,
                             NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass,
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
        return ToString(ELossFunction::F1);
    }
}

bool TF1Metric::IsMaxOptimal() const {
    return true;
}

/* TotalF1 */

TMetricHolder TTotalF1Metric::Eval(const TVector<TVector<double>>& approx,
                                  const TVector<float>& target,
                                  const TVector<float>& weight,
                                  int begin, int end,
                                  NPar::TLocalExecutor& /* executor */) const {
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

bool TTotalF1Metric::IsMaxOptimal() const {
    return true;
}

/* Confusion matrix */

static TVector<TVector<double>> GetConfusionMatrix(const TVector<TVector<double>>& approx,
                                                   const TVector<float>& target,
                                                   const TVector<float>& weight,
                                                   int begin, int end) {
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

TMetricHolder TMCCMetric::Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& /* executor */) const {
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

bool TMCCMetric::IsMaxOptimal() const {
    return true;
}

/* PairAccuracy */

TMetricHolder TPairAccuracyMetric::EvalPairwise(const TVector<TVector<double>>& approx,
                                               const TVector<TPair>& pairs,
                                               int begin, int end) const {
    CB_ENSURE(approx.size() == 1, "Metric PairLogit supports only single-dimensional data");

    TMetricHolder error;
    for (const auto& pair : pairs) {
        if (pair.WinnerId < begin || pair.WinnerId >= end ||
            pair.LoserId < begin || pair.LoserId >= end) {
            continue;
        }

        float w = pair.Weight;
        if (approx[0][pair.WinnerId] > approx[0][pair.LoserId]) {
            error.Error += w;
        }
        error.Weight += w;
    }

    return error;
}

TString TPairAccuracyMetric::GetDescription() const {
    return ToString(ELossFunction::PairAccuracy);
}

bool TPairAccuracyMetric::IsMaxOptimal() const {
    return true;
}

/* Custom */

TCustomMetric::TCustomMetric(const TCustomMetricDescriptor& descriptor)
    : Descriptor(descriptor)
{
}

TMetricHolder TCustomMetric::Eval(const TVector<TVector<double>>& approx,
                                 const TVector<float>& target,
                                 const TVector<float>& weight,
                                 int begin, int end,
                                 NPar::TLocalExecutor& /* executor */) const {
    return Descriptor.EvalFunc(approx, target, weight, begin, end, Descriptor.CustomData);
}

TMetricHolder TCustomMetric::EvalPairwise(const TVector<TVector<double>>& /*approx*/,
                                         const TVector<TPair>& /*pairs*/,
                                         int /*begin*/, int /*end*/) const {
    CB_ENSURE(false, "This eval is only for PairLogit");
}

TMetricHolder TCustomMetric::EvalQuerywise(const TVector<TVector<double>>& /*approx*/,
                                          const TVector<float>& /*target*/,
                                          const TVector<float>& /*weight*/,
                                          const TVector<ui32>& /*queriesId*/,
                                          const yhash<ui32, ui32>& /*queriesSize*/,
                                          int /*begin*/, int /*end*/) const {
    CB_ENSURE(false, "This eval is only for QueryRMSE");
}

TString TCustomMetric::GetDescription() const {
    return Descriptor.GetDescriptionFunc(Descriptor.CustomData);
}

bool TCustomMetric::IsMaxOptimal() const {
    return Descriptor.IsMaxOptimalFunc(Descriptor.CustomData);
}

EErrorType TCustomMetric::GetErrorType() const {
    return EErrorType::PerObjectError;
}

double TCustomMetric::GetFinalError(const TMetricHolder& error) const {
    return Descriptor.GetFinalErrorFunc(error, Descriptor.CustomData);
}

/* Create */

TVector<THolder<IMetric>> CreateMetric(ELossFunction metric, const yhash<TString, float>& params, int approxDimension) {
    if (metric != ELossFunction::Quantile && metric != ELossFunction::LogLinQuantile) {
        CB_ENSURE(params.empty(), "Metric " + ToString(metric) + " does not have any params");
    }
    TVector<THolder<IMetric>> result;
    switch (metric) {
        case ELossFunction::Logloss:
            result.emplace_back(new TCrossEntropyMetric(ELossFunction::Logloss));
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
                result.emplace_back(new TQuantileMetric(ELossFunction::Quantile, it->second));
            } else {
                result.emplace_back(new TQuantileMetric(ELossFunction::Quantile));
            }
            return result;
        }

        case ELossFunction::LogLinQuantile: {
            auto it = params.find("alpha");
            if (it != params.end()) {
                result.emplace_back(new TLogLinQuantileMetric(it->second));
            } else {
                result.emplace_back(new TLogLinQuantileMetric());
            }
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

        case ELossFunction::R2:
            result.emplace_back(new TR2Metric());
            return result;

        case ELossFunction::AUC: {
            if (approxDimension == 1) {
                result.emplace_back(new TAUCMetric());
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TAUCMetric(i));
                }
            }
            return result;
        }

        case ELossFunction::Accuracy:
            result.emplace_back(new TAccuracyMetric());
            return result;

        case ELossFunction::Precision: {
            if (approxDimension == 1) {
                result.emplace_back(new TPrecisionMetric());
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TPrecisionMetric(i));
                }
            }
            return result;
        }

        case ELossFunction::Recall: {
            if (approxDimension == 1) {
                result.emplace_back(new TRecallMetric());
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TRecallMetric(i));
                }
            }
            return result;
        }

        case ELossFunction::F1: {
            if (approxDimension == 1) {
                result.emplace_back(new TF1Metric());
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TF1Metric(i));
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

        default:
            Y_ASSERT(false);
            return TVector<THolder<IMetric>>();
    }
}

TVector<THolder<IMetric>> CreateMetricFromDescription(const TString& description, int approxDimension) {
    ELossFunction metric = GetLossType(description);
    auto params = GetLossParams(description);
    return CreateMetric(metric, params, approxDimension);
}

TVector<THolder<IMetric>> CreateMetrics(const TMaybe<TString>& evalMetric, const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                                        const TVector<TString>& customLoss, int approxDimension) {
    TVector<THolder<IMetric>> errors;
    if (GetLossType(*evalMetric) == ELossFunction::Custom) {
        errors.emplace_back(new TCustomMetric(*evalMetricDescriptor));
    } else {
        TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(*evalMetric, approxDimension);
        for (auto& metric : createdMetrics) {
            errors.push_back(std::move(metric));
        }
    }

    for (const TString& description : customLoss) {
        TVector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(description, approxDimension);
        for (auto& metric : createdMetrics) {
            errors.push_back(std::move(metric));
        }
    }
    return errors;
}

ELossFunction GetLossType(const TString& lossDescription) {
    TVector<TString> tokens = StringSplitter(lossDescription).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(!tokens.empty(), "custom loss is missing in desctiption: " << lossDescription);
    ELossFunction customLoss;
    CB_ENSURE(TryFromString<ELossFunction>(tokens[0], customLoss), tokens[0] + " loss is not supported");
    return customLoss;
}

yhash<TString, float> GetLossParams(const TString& lossDescription) {
    const char* errorMessage = "Invalid metric description, it should be in the form "
                               "\"metric_name:param1=value1;...;paramN=valueN\"";

    TVector<TString> tokens = StringSplitter(lossDescription).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(!tokens.empty(), "Metric description should not be empty");
    CB_ENSURE(tokens.size() <= 2, errorMessage);

    yhash<TString, float> params;
    if (tokens.size() == 2) {
        TVector<TString> paramsTokens = StringSplitter(tokens[1]).Split(';').ToList<TString>();

        for (const auto& token : paramsTokens) {
            TVector<TString> keyValue = StringSplitter(token).SplitLimited('=', 2).ToList<TString>();
            CB_ENSURE(keyValue.size() == 2, errorMessage);
            params[keyValue[0]] = FromString<float>(keyValue[1]);
        }
    }
    return params;
}

