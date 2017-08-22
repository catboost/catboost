#include "metric.h"
#include <catboost/libs/helpers/exception.h>

#include <catboost/libs/metrics/auc.h>

#include <util/generic/ymath.h>
#include <util/generic/string.h>
#include <util/string/iterator.h>
#include <util/string/cast.h>
#include <util/string/printf.h>
#include <util/system/yassert.h>

#include <limits>

double TMetric::GetFinalError(const TErrorHolder& error) const {
    return error.Error / (error.Weight + 1e-38);
}

/* Logloss */

TLoglossMetric::TLoglossMetric(ELossFunction lossFunction)
    : LossFunction(lossFunction)
{
    Y_ASSERT(lossFunction == ELossFunction::Logloss || lossFunction == ELossFunction::CrossEntropy);
}

TErrorHolder TLoglossMetric::Eval(const yvector<yvector<double>>& approx,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
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
    blockParams.WaitCompletion().SetBlockCount(executor.GetThreadCount() + 1);
    yvector<TErrorHolder> errorHolders(blockParams.GetBlockCount());

    executor.ExecRange([&](int blockId) {
        TErrorHolder holder;
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

    TErrorHolder error;
    for (const auto& eh : errorHolders) {
        error.Add(eh);
    }

    return error;
}

TString TLoglossMetric::GetDescription() const {
    return ToString(LossFunction);
}

bool TLoglossMetric::IsMaxOptimal() const {
    return false;
}

/* RMSE */

TErrorHolder TRMSEMetric::Eval(const yvector<yvector<double>>& approx,
                               const yvector<float>& target,
                               const yvector<float>& weight,
                               int begin, int end,
                               NPar::TLocalExecutor& /* executor */) const {
    CB_ENSURE(approx.size() == 1, "Metric RMSE supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TErrorHolder error;
    for (int k = begin; k < end; ++k) {
        float w = weight.empty() ? 1 : weight[k];
        error.Error += Sqr(approxVec[k] - target[k]) * w;
        error.Weight += w;
    }
    return error;
}

double TRMSEMetric::GetFinalError(const TErrorHolder& error) const {
    return sqrt(error.Error / (error.Weight + 1e-38));
}

TString TRMSEMetric::GetDescription() const {
    return ToString(ELossFunction::RMSE);
}

bool TRMSEMetric::IsMaxOptimal() const {
    return false;
}

/* R2 */

TErrorHolder TR2Metric::Eval(const yvector<yvector<double>>& approx,
                             const yvector<float>& target,
                             const yvector<float>& weight,
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
    TErrorHolder error;
    error.Error = 1 - mse / targetVariance;
    error.Weight = 1;
    return error;
}

double TR2Metric::GetFinalError(const TErrorHolder& error) const {
    return error.Error;
}

TString TR2Metric::GetDescription() const {
    return ToString(ELossFunction::R2);
}

bool TR2Metric::IsMaxOptimal() const {
    return true;
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

TErrorHolder TQuantileMetric::Eval(const yvector<yvector<double>>& approx,
                                   const yvector<float>& target,
                                   const yvector<float>& weight,
                                   int begin, int end,
                                   NPar::TLocalExecutor& /* executor */) const {
    CB_ENSURE(approx.size() == 1, "Metric quantile supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TErrorHolder error;
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

/* Log-linear quantile */

TLogLinearQuantileMetric::TLogLinearQuantileMetric(double alpha)
    : Alpha(alpha)
{
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6,
              "Alpha parameter for log-linear quantile metric should be in interval (0, 1)");
}

TErrorHolder TLogLinearQuantileMetric::Eval(const yvector<yvector<double>>& approx,
                                            const yvector<float>& target,
                                            const yvector<float>& weight,
                                            int begin, int end,
                                            NPar::TLocalExecutor& /* executor */) const {
    CB_ENSURE(approx.size() == 1, "Metric log-linear quantile supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TErrorHolder error;
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        double val = target[i] - exp(approxVec[i]);
        double multiplier = (val > 0) ? Alpha : -(1 - Alpha);
        error.Error += (multiplier * val) * w;
        error.Weight += w;
    }

    return error;
}

TString TLogLinearQuantileMetric::GetDescription() const {
    auto metricName = ToString(ELossFunction::LogLinQuantile);
    return Sprintf("%s:alpha=%.3lf", metricName.c_str(), Alpha);
}

bool TLogLinearQuantileMetric::IsMaxOptimal() const {
    return false;
}

/* Poisson */

TErrorHolder TPoissonMetric::Eval(const yvector<yvector<double>>& approx,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
                                  int begin, int end,
                                  NPar::TLocalExecutor& /* executor */) const {
    // Error function:
    // Sum_d[approx(d) - target(d) * log(approx(d))]
    // approx(d) == exp(Sum(tree_value))

    Y_ASSERT(approx.size() == 1);

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TErrorHolder error;
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

/* MAPE */

TErrorHolder TMAPEMetric::Eval(const yvector<yvector<double>>& approx,
                               const yvector<float>& target,
                               const yvector<float>& weight,
                               int begin, int end,
                               NPar::TLocalExecutor& /* executor */) const {
    CB_ENSURE(approx.size() == 1, "Metric MAPE quantile supports only single-dimensional data");

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TErrorHolder error;
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

/* Classification helpers */

static int GetApproxClass(const yvector<yvector<double>>& approx, int docIdx) {
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

static void GetPositiveStats(const yvector<yvector<double>>& approx,
                             const yvector<float>& target,
                             const yvector<float>& weight,
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

static void GetTotalPositiveStats(const yvector<yvector<double>>& approx,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
                                  int begin, int end,
                                  yvector<double>* truePositive,
                                  yvector<double>* targetPositive,
                                  yvector<double>* approxPositive) {
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

/* ROC-AUC */

TRocAUCMetric::TRocAUCMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TErrorHolder TRocAUCMetric::Eval(const yvector<yvector<double>>& approx,
                                 const yvector<float>& target,
                                 const yvector<float>& weight,
                                 int begin, int end,
                                 NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);
    const auto& approxVec = approx.ysize() == 1 ? approx.front() : approx[PositiveClass];
    Y_ASSERT(approxVec.size() == target.size());

    yvector<double> approxCopy(approxVec.begin() + begin, approxVec.begin() + end);
    yvector<double> targetCopy(target.begin() + begin, target.begin() + end);

    if (approx.ysize() > 1) {
        int positiveClass = PositiveClass;
        ForEach(targetCopy.begin(), targetCopy.end(), [positiveClass](double& x) {
            x = (x == static_cast<double>(positiveClass));
        });
    }

    yvector<NMetrics::TSample> samples;
    if (weight.empty()) {
        samples = NMetrics::TSample::FromVectors(targetCopy, approxCopy);
    } else {
        yvector<double> weightCopy(weight.begin() + begin, weight.begin() + end);
        samples = NMetrics::TSample::FromVectors(targetCopy, approxCopy, weightCopy);
    }

    TErrorHolder error;
    error.Error = CalcAUC(&samples);
    error.Weight = 1.0;
    return error;
}

TString TRocAUCMetric::GetDescription() const {
    if (IsMultiClass) {
        return Sprintf("%s:class=%d", ToString(ELossFunction::AUC).c_str(), PositiveClass);
    } else {
        return ToString(ELossFunction::AUC);
    }
}

bool TRocAUCMetric::IsMaxOptimal() const {
    return true;
}

/* Recall */

TRecallMetric::TRecallMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TErrorHolder TRecallMetric::Eval(const yvector<yvector<double>>& approx,
                                 const yvector<float>& target,
                                 const yvector<float>& weight, int begin, int end,
                                 NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass,
                     &truePositive, &targetPositive, &approxPositive);

    TErrorHolder error;
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

/* Precision */

TPrecisionMetric::TPrecisionMetric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TErrorHolder TPrecisionMetric::Eval(const yvector<yvector<double>>& approx,
                                    const yvector<float>& target,
                                    const yvector<float>& weight, int begin, int end,
                                    NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass,
                     &truePositive, &targetPositive, &approxPositive);

    TErrorHolder error;
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

/* F1 */

TF1Metric::TF1Metric(int positiveClass)
    : PositiveClass(positiveClass)
    , IsMultiClass(true)
{
    CB_ENSURE(PositiveClass >= 0, "Class id should not be negative");
}

TErrorHolder TF1Metric::Eval(const yvector<yvector<double>>& approx,
                             const yvector<float>& target,
                             const yvector<float>& weight, int begin, int end,
                             NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT((approx.size() > 1) == IsMultiClass);

    double truePositive;
    double targetPositive;
    double approxPositive;
    GetPositiveStats(approx, target, weight, begin, end, PositiveClass,
                     &truePositive, &targetPositive, &approxPositive);

    TErrorHolder error;
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

TErrorHolder TTotalF1Metric::Eval(const yvector<yvector<double>>& approx,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
                                  int begin, int end,
                                  NPar::TLocalExecutor& /* executor */) const {
    yvector<double> truePositive;
    yvector<double> targetPositive;
    yvector<double> approxPositive;
    GetTotalPositiveStats(approx, target, weight, begin, end,
                          &truePositive, &targetPositive, &approxPositive);

    int classesCount = truePositive.ysize();
    TErrorHolder error;
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

/* Accuracy */

TErrorHolder TAccuracyMetric::Eval(const yvector<yvector<double>>& approx,
                                   const yvector<float>& target,
                                   const yvector<float>& weight,
                                   int begin, int end,
                                   NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());

    TErrorHolder error;
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

/* Multiclass */

TErrorHolder TMulticlassLoglossMetric::Eval(const yvector<yvector<double>>& approx,
                                            const yvector<float>& target,
                                            const yvector<float>& weight,
                                            int begin, int end,
                                            NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());
    int approxDimension = approx.ysize();

    TErrorHolder error;

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

TString TMulticlassLoglossMetric::GetDescription() const {
    return ToString(ELossFunction::MultiClass);
}

bool TMulticlassLoglossMetric::IsMaxOptimal() const {
    return true;
}

/* MulticlassOneVsAll */

TErrorHolder TMulticlassOneVsAllLoglossMetric::Eval(const yvector<yvector<double>>& approx,
                                                    const yvector<float>& target,
                                                    const yvector<float>& weight,
                                                    int begin, int end,
                                                    NPar::TLocalExecutor& /* executor */) const {
    Y_ASSERT(target.ysize() == approx[0].ysize());
    int approxDimension = approx.ysize();

    TErrorHolder error;
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

TString TMulticlassOneVsAllLoglossMetric::GetDescription() const {
    return ToString(ELossFunction::MultiClassOneVsAll);
}

bool TMulticlassOneVsAllLoglossMetric::IsMaxOptimal() const {
    return true;
}

/* Custom */

TCustomMetric::TCustomMetric(const TCustomMetricDescriptor& descriptor)
    : Descriptor(descriptor)
{
}

TErrorHolder TCustomMetric::Eval(const yvector<yvector<double>>& approx,
                                 const yvector<float>& target,
                                 const yvector<float>& weight,
                                 int begin, int end,
                                 NPar::TLocalExecutor& /* executor */) const {
    return Descriptor.EvalFunc(approx, target, weight, begin, end, Descriptor.CustomData);
}

TString TCustomMetric::GetDescription() const {
    return Descriptor.GetDescriptionFunc(Descriptor.CustomData);
}

bool TCustomMetric::IsMaxOptimal() const {
    return Descriptor.IsMaxOptimalFunc(Descriptor.CustomData);
}

double TCustomMetric::GetFinalError(const TErrorHolder& error) const {
    return Descriptor.GetFinalErrorFunc(error, Descriptor.CustomData);
}

yvector<THolder<IMetric>> CreateMetric(ELossFunction metric, const yhash<TString, float>& params, int approxDimension) {
    if (metric != ELossFunction::Quantile && metric != ELossFunction::LogLinQuantile) {
        CB_ENSURE(params.empty(), "Metric " + ToString(metric) + " does not have any params");
    }
    yvector<THolder<IMetric>> result;
    switch (metric) {
        case ELossFunction::R2:
            result.emplace_back(new TR2Metric());
            return result;

        case ELossFunction::Logloss:
            result.emplace_back(new TLoglossMetric(ELossFunction::Logloss));
            return result;
        case ELossFunction::CrossEntropy:
            result.emplace_back(new TLoglossMetric(ELossFunction::CrossEntropy));
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
                result.emplace_back(new TLogLinearQuantileMetric(it->second));
            } else {
                result.emplace_back(new TLogLinearQuantileMetric());
            }
            return result;
        }
        case ELossFunction::Poisson:
            result.emplace_back(new TPoissonMetric());
            return result;
        case ELossFunction::MAPE:
            result.emplace_back(new TMAPEMetric());
            return result;
        case ELossFunction::Accuracy:
            result.emplace_back(new TAccuracyMetric());
            return result;
        case ELossFunction::MultiClass:
            result.emplace_back(new TMulticlassLoglossMetric());
            return result;
        case ELossFunction::MultiClassOneVsAll:
            result.emplace_back(new TMulticlassOneVsAllLoglossMetric());
            return result;
        case ELossFunction::AUC: {
            if (approxDimension == 1) {
                result.emplace_back(new TRocAUCMetric());
            } else {
                for (int i = 0; i < approxDimension; ++i) {
                    result.emplace_back(new TRocAUCMetric(i));
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
        default:
            Y_ASSERT(false);
            return yvector<THolder<IMetric>>();
    }
}

yvector<THolder<IMetric>> CreateMetricFromDescription(const TString& description, int approxDimension) {
    ELossFunction metric = GetLossType(description);
    auto params = GetLossParams(description);
    return CreateMetric(metric, params, approxDimension);
}

yvector<THolder<IMetric>> CreateMetrics(const TFitParams& params, int approxDimension) {
    yvector<THolder<IMetric>> errors;
    if (GetLossType(*params.EvalMetric) == ELossFunction::Custom) {
        errors.emplace_back(new TCustomMetric(*params.EvalMetricDescriptor));
    } else {
        yvector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(*params.EvalMetric, approxDimension);
        for (auto& metric : createdMetrics) {
            errors.push_back(std::move(metric));
        }
    }

    for (const TString& description : params.CustomLoss) {
        yvector<THolder<IMetric>> createdMetrics = CreateMetricFromDescription(description, approxDimension);
        for (auto& metric : createdMetrics) {
            errors.push_back(std::move(metric));
        }
    }
    return errors;
}
