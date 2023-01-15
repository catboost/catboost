#include "caching_metric.h"
#include "metric.h"
#include "description_utils.h"
#include "classification_utils.h"
#include "enums.h"
#include "kappa.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <util/generic/string.h>
#include <util/generic/set.h>

using NCB::AppendTemporaryMetricsVector;
using NCB::AsVector;

constexpr int BinaryClassesCount = 2;
static const TString ConfusionMatrixCacheKey = "Confusion Matrix";

/* Caching metric */

namespace {
    class ICacheHolder {
    public:
        virtual ~ICacheHolder() = default;
    };

    template <typename TValue, typename... TKeys>
    class TCacheHolder : public ICacheHolder {
    public:
        template <typename TValueMaker>
        TValue& Get(TValueMaker maker, const TKeys & ...keys) {
            auto key = std::make_tuple(keys...);
            if (!Cache.contains(key)) {
                Cache.insert({key, maker()});
            }

            return Cache.at(key);
        }

    private:
        TMap<std::tuple<TKeys...>, TValue> Cache;
    };

    class TCache {
    public:
        template <typename TValueMaker, typename... TKeys>
        auto Get(const TString& cacheKey, TValueMaker maker, TKeys... keys) -> const decltype(maker()) & {
            if (!Cache.contains(cacheKey)) {
                Cache.insert({cacheKey, MakeHolder<TCacheHolder<decltype(maker()), TKeys...>>()});
            }

            auto cache = dynamic_cast<TCacheHolder<decltype(maker()), TKeys...> *>(Cache.at(cacheKey).Get());
            CB_ENSURE(cache, "Cache is typed differently");
            return cache->Get(maker, keys...);
        }

    private:
        TMap<TString, THolder<ICacheHolder>> Cache;
    };

    struct TCachingMetric: public TMetric {
        explicit TCachingMetric(ELossFunction lossFunction, const TLossParams& params)
            : TMetric(lossFunction, params)
            {}
        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor
        ) const override {
            return Eval(To2DConstArrayRef<double>(approx), /*approxDelta*/{}, /*isExpApprox*/false, target, weight, queriesInfo, begin, end, executor);
        }
        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor
        ) const override {
            const auto evalMetric = [&](int from, int to) {
                return Eval(approx, approxDelta, isExpApprox, target, weight, queriesInfo, from, to, Nothing());
            };

            if (IsAdditiveMetric()) {
                return ParallelEvalMetric(evalMetric, GetMinBlockSize(end - begin), begin, end, executor);
            } else {
                return evalMetric(begin, end);
            }
        }
        virtual TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const = 0;
    };
}
/* Confusion matrix */

static TMetricHolder BuildConfusionMatrix(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    double targetBorder,
    double predictionBorder
) {
    const bool isMultiClass = approx.size() > 1;
    const int classesCount = isMultiClass ? approx.size() : BinaryClassesCount;
    const double predictionLogitBorder = NCB::Logit(predictionBorder);

    const auto buildImpl = [&](auto useWeights, auto isMultiClass) {
        TMetricHolder confusionMatrix(classesCount * classesCount);
        for (int idx = begin; idx < end; ++idx) {
            int approxClass = GetApproxClass(approx, idx, predictionLogitBorder);
            int targetClass = isMultiClass ? static_cast<int>(target[idx]) : target[idx] > targetBorder;
            float w = useWeights ? weight[idx] : 1;

            Y_ASSERT(targetClass >= 0 && targetClass < classesCount);

            confusionMatrix.Stats[approxClass * classesCount + targetClass] += w;
        }
        return confusionMatrix;
    };

    return DispatchGenericLambda(buildImpl, !weight.empty(), isMultiClass);
}

/* MCC caching metric */

namespace {
    struct TMCCCachingMetric final : public TCachingMetric {
        explicit TMCCCachingMetric(const TLossParams& params,
                                   int classesCount)
            : TCachingMetric(ELossFunction::MCC, params)
            , ClassesCount(classesCount) {
        }
        explicit TMCCCachingMetric(const TLossParams& params,
                                   double predictionBorder)
            : TCachingMetric(ELossFunction::MCC, params)
            , PredictionBorder(predictionBorder)
        {
        }
        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const override;
        double GetFinalError(const TMetricHolder &error) const override;
        void GetBestValue(EMetricBestValue *valueType, float *bestPossibleValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int ClassesCount = BinaryClassesCount;
        const double PredictionBorder = GetDefaultPredictionBorder();
    };
}

THolder<IMetric> MakeMCCMetric(const TLossParams& params, int classesCount) {
    return MakeHolder<TMCCCachingMetric>(params, classesCount);
}

TMetricHolder TMCCCachingMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(approxDelta.empty());

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights),
            TargetBorder, PredictionBorder);

    return confusionMatrix;
}

double TMCCCachingMetric::GetFinalError(const TMetricHolder &error) const {
    TVector<double> rowSum(ClassesCount);
    TVector<double> columnSum(ClassesCount);
    double totalSum = 0;

    const auto getStats = [&](int i, int j) { return error.Stats[i * ClassesCount + j]; };
    for (auto i : xrange(ClassesCount)) {
        for (auto j : xrange(ClassesCount)) {
            rowSum[i] += getStats(i, j);
            columnSum[j] += getStats(i, j);
            totalSum += getStats(i, j);
        }
    }

    double numerator = 0;
    for (auto i : xrange(ClassesCount)) {
        numerator += getStats(i, i) * totalSum - rowSum[i] * columnSum[i];
    }

    double sumSquareRowSums = 0;
    double sumSquareColumnSums = 0;
    for (auto i : xrange(ClassesCount)) {
        sumSquareRowSums += Sqr(rowSum[i]);
        sumSquareColumnSums += Sqr(columnSum[i]);
    }

    double denominator = sqrt((Sqr(totalSum) - sumSquareRowSums) * (Sqr(totalSum) - sumSquareColumnSums));
    return denominator != 0 ? numerator / denominator : 0.0;
}

void TMCCCachingMetric::GetBestValue(EMetricBestValue *valueType, float *) const {
    *valueType = EMetricBestValue::Max;
}

/* Zero one loss caching metric */

namespace {
    struct TZeroOneLossCachingMetric final: public TCachingMetric {
        explicit TZeroOneLossCachingMetric(const TLossParams& params,
                                           int classesCount)
            : TCachingMetric(ELossFunction::ZeroOneLoss, params)
            , ClassesCount(classesCount) {
        }
        explicit TZeroOneLossCachingMetric(const TLossParams& params,
                                           double predictionBorder)
            : TCachingMetric(ELossFunction::ZeroOneLoss, params)
            , PredictionBorder(predictionBorder) {
        }
        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const override;
        double GetFinalError(const TMetricHolder &error) const override;
        void GetBestValue(EMetricBestValue *valueType, float *bestPossibleValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int ClassesCount = BinaryClassesCount;
        const double PredictionBorder = GetDefaultPredictionBorder();
    };
}

TMetricHolder TZeroOneLossCachingMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(approxDelta.empty());

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights),
            TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(2);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[0] += getStats(i, i);
        for (auto j : xrange(ClassesCount)) {
            error.Stats[1] += getStats(i, j);
        }
    }
    return error;
}

double TZeroOneLossCachingMetric::GetFinalError(const TMetricHolder &error) const {
    return 1 - error.Stats[0] / error.Stats[1];
}

void TZeroOneLossCachingMetric::GetBestValue(EMetricBestValue *valueType, float *) const {
    *valueType = EMetricBestValue::Min;
}

/* Accuracy caching metric */

namespace {
    struct TAccuracyCachingMetric final: public TCachingMetric {
        explicit TAccuracyCachingMetric(const TLossParams& params,
                                        double predictionBorder)
            : TCachingMetric(ELossFunction::Accuracy, params)
            , PredictionBorder(predictionBorder) {
        }
        explicit TAccuracyCachingMetric(const TLossParams& params,
                                        int classesCount)
            : TCachingMetric(ELossFunction::Accuracy, params)
            , ClassesCount(classesCount) {
        }

        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const double PredictionBorder = GetDefaultPredictionBorder();
        const int ClassesCount = BinaryClassesCount;
    };
}

TMetricHolder TAccuracyCachingMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(approxDelta.empty());

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights),
            TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(2);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[0] += getStats(i, i);
        for (auto j : xrange(ClassesCount)) {
            error.Stats[1] += getStats(i, j);
        }
    }
    return error;
}

void TAccuracyCachingMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Recall caching metric */

namespace {
    struct TRecallCachingMetric final: public TCachingMetric {
        explicit TRecallCachingMetric(const TLossParams& params,
                                      double predictionBorder)
            : TCachingMetric(ELossFunction::Recall, params)
            , ClassesCount(BinaryClassesCount)
            , PredictionBorder(predictionBorder)
            , IsMultiClass(false) {
        }

        explicit TRecallCachingMetric(const TLossParams& params,
                                      int classesCount, int positiveClass)
            : TCachingMetric(ELossFunction::Recall, params)
            , ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const override;

        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int ClassesCount = BinaryClassesCount;
        const int PositiveClass = 1;
        const double PredictionBorder = GetDefaultPredictionBorder();
        const bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassRecallMetric(const TLossParams& params,
                                          double predictionBorder) {
    return MakeHolder<TRecallCachingMetric>(params, predictionBorder);
}

THolder<IMetric> MakeMultiClassRecallMetric(const TLossParams& params,
                                            int classesCount, int positiveClass) {
    return MakeHolder<TRecallCachingMetric>(params, classesCount, positiveClass);
}

TMetricHolder TRecallCachingMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(approxDelta.empty());

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(2);
    error.Stats[0] = getStats(PositiveClass, PositiveClass);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[1] += getStats(i, PositiveClass);
    }
    return error;
}

TString TRecallCachingMetric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::Recall, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::Recall, UseWeights, "%.3g", MakeTargetBorderParam(TargetBorder),
                                MakePredictionBorderParam(PredictionBorder));
    }
}

double TRecallCachingMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1;
}

void TRecallCachingMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Precision caching metric */

namespace {
    struct TPrecisionCachingMetric final: public TCachingMetric {
        explicit TPrecisionCachingMetric(const TLossParams& params,
                                         double predictionBorder)
            : TCachingMetric(ELossFunction::Precision, params)
            , ClassesCount(BinaryClassesCount)
            , PredictionBorder(predictionBorder)
            , IsMultiClass(false) {
        }

        explicit TPrecisionCachingMetric(const TLossParams& params,
                                         int classesCount, int positiveClass)
            : TCachingMetric(ELossFunction::Precision, params)
            , ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int ClassesCount = BinaryClassesCount;
        const int PositiveClass = 1;
        const double PredictionBorder = GetDefaultPredictionBorder();
        const bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassPrecisionMetric(const TLossParams& params,
                                             double predictionBorder) {
    return MakeHolder<TPrecisionCachingMetric>(params, predictionBorder);
}

THolder<IMetric> MakeMultiClassPrecisionMetric(const TLossParams& params,
                                               int classesCount, int positiveClass) {
    return MakeHolder<TPrecisionCachingMetric>(params, classesCount, positiveClass);
}

TMetricHolder TPrecisionCachingMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(approxDelta.empty());

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(2);
    error.Stats[0] = getStats(PositiveClass, PositiveClass);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[1] += getStats(PositiveClass, i);
    }

    return error;
}

TString TPrecisionCachingMetric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::Precision, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::Precision, UseWeights, "%.3g", MakeTargetBorderParam(TargetBorder),
                                MakePredictionBorderParam(PredictionBorder));
    }
}

double TPrecisionCachingMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1;
}

void TPrecisionCachingMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* F1 caching metric */

namespace {
    struct TF1CachingMetric final: public TCachingMetric {
        explicit TF1CachingMetric(const TLossParams& params,
                                  double predictionBorder)
            : TCachingMetric(ELossFunction::F1, params)
            , ClassesCount(BinaryClassesCount)
            , PredictionBorder(predictionBorder)
            , IsMultiClass(false) {
        }

        explicit TF1CachingMetric(const TLossParams& params,
                                  int classesCount, int positiveClass)
            : TCachingMetric(ELossFunction::F1, params)
            , ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        TString GetDescription() const override;
        TVector<TString> GetStatDescriptions() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int ClassesCount = BinaryClassesCount;
        const int PositiveClass = 1;
        const double PredictionBorder = GetDefaultPredictionBorder();
        const bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassF1Metric(const TLossParams& params,
                                      double predictionBorder) {
    return MakeHolder<TF1CachingMetric>(params, predictionBorder);
}

THolder<IMetric> MakeMultiClassF1Metric(const TLossParams& params,
                                        int classesCount, int positiveClass) {
    return MakeHolder<TF1CachingMetric>(params, classesCount, positiveClass);
}

TMetricHolder TF1CachingMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(approxDelta.empty());

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(3);
    error.Stats[0] = getStats(PositiveClass, PositiveClass);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[1] += getStats(PositiveClass, i);
        error.Stats[2] += getStats(i, PositiveClass);
    }
    return error;
}

TVector<TString> TF1CachingMetric::GetStatDescriptions() const {
    return {"TP", "TP+FP", "TP+FN"};
}

TString TF1CachingMetric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::F1, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::F1, UseWeights, "%.3g", MakeTargetBorderParam(TargetBorder),
                                MakePredictionBorderParam(PredictionBorder));
    }
}

double TF1CachingMetric::GetFinalError(const TMetricHolder& error) const {
    double precision = error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1.0;
    double recall = error.Stats[2] != 0 ? error.Stats[0] / error.Stats[2] : 1.0;
    return precision + recall != 0 ? 2 * precision * recall / (precision + recall) : 0.0;
}

void TF1CachingMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* TotalF1 caching metric */

namespace {
    struct TTotalF1CachingMetric final: public TCachingMetric {
        static constexpr int StatsCardinality = 3;

        explicit TTotalF1CachingMetric(const TLossParams& params,
                                       double predictionBorder, EF1AverageType averageType)
            : TCachingMetric(ELossFunction::TotalF1, params)
            , ClassesCount(BinaryClassesCount)
            , PredictionBorder(predictionBorder)
            , AverageType(averageType) {
        }

        explicit TTotalF1CachingMetric(const TLossParams& params,
                                       int classesCount, EF1AverageType averageType)
            : TCachingMetric(ELossFunction::TotalF1, params)
            , ClassesCount(classesCount)
            , AverageType(averageType) {
        }

        TMetricHolder Eval(
            const TConstArrayRef<TConstArrayRef<double>> approx,
            const TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        TVector<TString> GetStatDescriptions() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int ClassesCount = BinaryClassesCount;
        const double PredictionBorder = GetDefaultPredictionBorder();
        const EF1AverageType AverageType;
    };
}

THolder<IMetric> MakeTotalF1Metric(const TLossParams& params,
                                   int classesCount, EF1AverageType averageType) {
    return MakeHolder<TTotalF1CachingMetric>(params, classesCount, averageType);
}

TMetricHolder TTotalF1CachingMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(!isExpApprox);
    Y_ASSERT(approxDelta.empty());

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() || 1 ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TVector<double> classTruePositive(ClassesCount);
    TVector<double> classSize(ClassesCount);
    TVector<double> classPredictedSize(ClassesCount);

    for (auto i : xrange(ClassesCount)) {
        classTruePositive[i] = getStats(i, i);
        for (auto j : xrange(ClassesCount)) {
            classSize[i] += getStats(j, i);
            classPredictedSize[i] += getStats(i, j);
        }
    }

    TMetricHolder error(StatsCardinality * ClassesCount);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[StatsCardinality * i + 0] = classSize[i];
        error.Stats[StatsCardinality * i + 1] = classPredictedSize[i];
        error.Stats[StatsCardinality * i + 2] = classTruePositive[i];
    }
    return error;
}

TVector<TString> TTotalF1CachingMetric::GetStatDescriptions() const {
    TVector<TString> description;
    for (auto classIdx : xrange(ClassesCount)) {
        auto prefix = "Class=" + ToString(classIdx) + ",";
        description.push_back(prefix + "TP+FN");
        description.push_back(prefix + "TP+FP");
        description.push_back(prefix + "TP");
    }
    return description;
}

double TTotalF1CachingMetric::GetFinalError(const TMetricHolder& error) const {
    double numerator = 0.0;
    double denumerator = 0.0;

    for (auto i : xrange(ClassesCount)) {
        double classSize = error.Stats[StatsCardinality * i + 0];
        double classPredictedSize = error.Stats[StatsCardinality * i + 1];
        double truePositive = error.Stats[StatsCardinality * i + 2];

        double f1 = classSize + classPredictedSize != 0 ? 2 * truePositive / (classSize + classPredictedSize) : 0.0;

        if (AverageType == EF1AverageType::Weighted) {
            numerator += f1 * classSize;
            denumerator += classSize;
        } else if (AverageType == EF1AverageType::Micro) {
            numerator += 2 * truePositive;
            denumerator += classSize + classPredictedSize;
        } else if (AverageType == EF1AverageType::Macro) {
            numerator += f1;
            denumerator += 1;
        }
    }

    return numerator / denumerator;
}

void TTotalF1CachingMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Kappa */

namespace {
    struct TKappaMetric final: public TCachingMetric {
        explicit TKappaMetric(const TLossParams& params,
                              int classCount = 2, double predictionBorder = GetDefaultPredictionBorder())
            : TCachingMetric(ELossFunction::Kappa, params)
            , TargetBorder(GetDefaultTargetBorder())
            , PredictionBorder(predictionBorder)
            , ClassCount(classCount) {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);

        TMetricHolder Eval(
                const TConstArrayRef<TConstArrayRef<double>> approx,
                const TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end,
                TMaybe<TCache*> cache
        ) const override;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        const double TargetBorder = GetDefaultTargetBorder();
        const double PredictionBorder = GetDefaultPredictionBorder();
        const int ClassCount;
    };
}

// static.
TVector<THolder<IMetric>> TKappaMetric::Create(const TMetricConfig& config) {
    if (config.approxDimension == 1) {
        config.validParams->insert("border");
        return AsVector(MakeHolder<TKappaMetric>(config.params, /*classCount=*/2, config.GetPredictionBorderOrDefault()));
    } else {
        return AsVector(MakeHolder<TKappaMetric>(config.params, config.approxDimension));
    }
}

TMetricHolder TKappaMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    return cache.Empty() || 1 ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);
}

TString TKappaMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Kappa, "%.3g", MakeTargetBorderParam(TargetBorder),
                            MakePredictionBorderParam(PredictionBorder));
}

void TKappaMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TKappaMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Cohen);
}

/* WKappa */

namespace {
    struct TWKappaMetric final: public TCachingMetric {
        explicit TWKappaMetric(const TLossParams& params,
                               int classCount = 2, double predictionBorder = GetDefaultPredictionBorder())
            : TCachingMetric(ELossFunction::WKappa, params)
            , TargetBorder(GetDefaultTargetBorder())
            , PredictionBorder(predictionBorder)
            , ClassCount(classCount) {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);

        TMetricHolder Eval(
                const TConstArrayRef<TConstArrayRef<double>> approx,
                const TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end,
                TMaybe<TCache*> cache
        ) const override;

        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue *valueType, float *bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        const double TargetBorder = GetDefaultTargetBorder();
        const double PredictionBorder = GetDefaultPredictionBorder();
        const int ClassCount;
    };
}

// static.
TVector<THolder<IMetric>> TWKappaMetric::Create(const TMetricConfig& config) {
    if (config.approxDimension == 1) {
        config.validParams->insert("border");
        return AsVector(MakeHolder<TWKappaMetric>(config.params, /*classCount=*/2, config.GetPredictionBorderOrDefault()));
    } else {
        return AsVector(MakeHolder<TWKappaMetric>(config.params, config.approxDimension));
    }
}

TMetricHolder TWKappaMetric::Eval(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    const auto MakeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    return cache.Empty() || 1 ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);
}

TString TWKappaMetric::GetDescription() const {
    return BuildDescription(ELossFunction::WKappa, "%.3g", MakeTargetBorderParam(TargetBorder),
                            MakePredictionBorderParam(PredictionBorder));
}

void TWKappaMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

double TWKappaMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Weighted);
}

TVector<TMetricHolder> EvalErrorsWithCaching(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    TConstArrayRef<const IMetric*> metrics,
    NPar::TLocalExecutor* localExecutor
) {
    const auto threadCount = localExecutor->GetThreadCount() + 1;
    const auto objectCount = approx.front().size();
    const auto queryCount = queriesInfo.size();

    const auto calcCaching = [&](auto metric, auto from, auto to, auto *cache) {
        CB_ENSURE(!metric->NeedTarget() || target.size() == 1, "Metric [" + metric->GetDescription() + "] requires "
                  << (target.size() > 1 ? "one-dimensional" : "") <<  "target");
        return metric->Eval(To2DConstArrayRef<double>(approx), To2DConstArrayRef<double>(approxDelta), isExpApprox, metric->NeedTarget() ? target[0] : TConstArrayRef<float>(),
                            weight, queriesInfo, from, to, cache);
    };
    const auto calcNonCaching = [&](auto metric, auto from, auto to) {
        CB_ENSURE(!metric->NeedTarget() || target.size() == 1, "Metric [" + metric->GetDescription() + "] requires "
                  << (target.size() > 1 ? "one-dimensional" : "") <<  "target");
        return metric->Eval(To2DConstArrayRef<double>(approx), To2DConstArrayRef<double>(approxDelta), isExpApprox, metric->NeedTarget() ? target[0] : TConstArrayRef<float>(),
                            weight, queriesInfo, from, to, *localExecutor);
    };
    const auto calcMultiRegression = [&](auto metric, auto from, auto to) {
        CB_ENSURE(!metric->NeedTarget() || target.size() > 0, "Metric [" + metric->GetDescription() + "] requires target");
        CB_ENSURE(!isExpApprox, "Metric [" << metric->GetDescription() << "] does not support exponentiated approxes");
        return metric->Eval(approx, approxDelta, target,
                            weight, from, to, *localExecutor);
    };

    TVector<TMetricHolder> errors;
    errors.reserve(metrics.size());

    NPar::TLocalExecutor::TExecRangeParams objectwiseBlockParams(0, objectCount);
    if (!target.empty()) {
        const auto objectwiseEffectiveBlockCount = Min(threadCount, int(ceil(double(objectCount) / GetMinBlockSize(objectCount))));
        objectwiseBlockParams.SetBlockCount(objectwiseEffectiveBlockCount);
    }

    NPar::TLocalExecutor::TExecRangeParams querywiseBlockParams(0, queryCount);
    if (!queriesInfo.empty()) {
        const auto querywiseEffectiveBlockCount = Min(threadCount, int(ceil(double(queryCount) / GetMinBlockSize(objectCount))));
        querywiseBlockParams.SetBlockCount(querywiseEffectiveBlockCount);
    }

    TCache nonAdditiveCache;
    TVector<TCache> objectwiseAdditiveCache(objectwiseBlockParams.GetBlockCount());
    TVector<TCache> querywiseAdditiveCache(querywiseBlockParams.GetBlockCount());
    TVector<TMetricHolder> objectwiseBlockResults(objectwiseBlockParams.GetBlockCount());
    TVector<TMetricHolder> querywiseBlockResults(querywiseBlockParams.GetBlockCount());

    for (auto i : xrange(metrics.size())) {
        auto metric = metrics[i];
        auto cachingMetric = dynamic_cast<const TCachingMetric*>(metrics[i]);
        auto multiMetric = dynamic_cast<const TMultiRegressionMetric*>(metrics[i]);
        Y_ASSERT(cachingMetric == nullptr || multiMetric == nullptr);

        const bool isObjectwise = metric->GetErrorType() == EErrorType::PerObjectError;
        if (cachingMetric && metric->IsAdditiveMetric()) {
            const auto blockSize = isObjectwise ? objectwiseBlockParams.GetBlockSize() : querywiseBlockParams.GetBlockSize();
            const auto blockCount = isObjectwise ? objectwiseBlockParams.GetBlockCount() : querywiseBlockParams.GetBlockCount();

            auto &results = isObjectwise ? objectwiseBlockResults : querywiseBlockResults;
            auto &cache = isObjectwise ? objectwiseAdditiveCache : querywiseAdditiveCache;
            const auto end = isObjectwise ? objectCount : queryCount;

            NPar::ParallelFor(*localExecutor, 0, blockCount, [&](auto blockId) {
                const auto from = blockId * blockSize;
                const auto to = Min<int>((blockId + 1) * blockSize, end);
                results[blockId] = calcCaching(cachingMetric, from, to, &cache[blockId]);
            });

            TMetricHolder error;
            for (const auto &blockResult : results) {
                error.Add(blockResult);
            }
            errors.push_back(error);
        } else {
            const auto end = isObjectwise ? objectCount : queryCount;
            if (cachingMetric) {
                errors.push_back(calcCaching(cachingMetric, 0, end, &nonAdditiveCache));
            } else if (multiMetric) {
                errors.push_back(calcMultiRegression(multiMetric, 0, end));
            } else {
                errors.push_back(calcNonCaching(metric, 0, end));
            }
        }
    }

    return errors;
}

template <typename TMetricType>
static TVector<THolder<IMetric>> CreateMetric(int approxDimension, const TMetricConfig& config) {
    TVector<THolder<IMetric>> result;
    if (approxDimension == 1) {
        result.emplace_back(MakeHolder<TMetricType>(config.params, config.GetPredictionBorderOrDefault()));
    } else {
        result.emplace_back(MakeHolder<TMetricType>(config.params, approxDimension));
    }
    return result;
}

template <typename TMetricType>
static TVector<THolder<IMetric>> CreateMetricClasswise(int approxDimension, const TMetricConfig& config) {
    TVector<THolder<IMetric>> result;
    if (approxDimension == 1) {
        result.emplace_back(MakeHolder<TMetricType>(config.params, config.GetPredictionBorderOrDefault()));
    } else {
        for (int i : xrange(approxDimension)) {
            result.emplace_back(MakeHolder<TMetricType>(config.params, approxDimension, i));
        }
    }
    return result;
}

TVector<THolder<IMetric>> CreateCachingMetrics(const TMetricConfig& config) {
    *config.validParams = TSet<TString>{};
    TVector<THolder<IMetric>> result;

    switch(config.metric) {
        case ELossFunction::F1: {
            return CreateMetricClasswise<TF1CachingMetric>(config.approxDimension, config);
        }
        case ELossFunction::TotalF1: {
            config.validParams->insert("average");
            EF1AverageType averageType = EF1AverageType::Weighted;
            if (config.GetParamsMap().contains("average")) {
                averageType = FromString<EF1AverageType>(config.GetParamsMap().at("average"));
            }

            if (config.approxDimension == 1) {
                result.emplace_back(MakeHolder<TTotalF1CachingMetric>(config.params, config.GetPredictionBorderOrDefault(), averageType));
            } else {
                result.emplace_back(MakeHolder<TTotalF1CachingMetric>(config.params, config.approxDimension, averageType));
            }
            return result;
        }
        case ELossFunction::MCC: {
            return CreateMetric<TMCCCachingMetric>(config.approxDimension, config);
        }
        case ELossFunction::BrierScore: {
            CB_ENSURE(config.approxDimension == 1, "Brier Score is used only for binary classification problems.");
            result.emplace_back(MakeBrierScoreMetric(config.params));
            return result;
        }
        case ELossFunction::ZeroOneLoss: {
            return CreateMetric<TZeroOneLossCachingMetric>(config.approxDimension, config);
        }
        case ELossFunction::Accuracy: {
            return CreateMetric<TAccuracyCachingMetric>(config.approxDimension, config);
        }
        case ELossFunction::CtrFactor: {
            result.emplace_back(MakeCtrFactorMetric(config.params));
            return result;
        }
        case ELossFunction::Precision: {
            return CreateMetricClasswise<TPrecisionCachingMetric>(config.approxDimension, config);
        }
        case ELossFunction::Recall: 
            return CreateMetricClasswise<TRecallCachingMetric>(config.approxDimension, config);
            break;
        case ELossFunction::Kappa:
            AppendTemporaryMetricsVector(TKappaMetric::Create(config), &result);
            break;
        
        case ELossFunction::WKappa:
            AppendTemporaryMetricsVector(TWKappaMetric::Create(config), &result);
            break;
        default:
            break;
    }

    return result;
}
