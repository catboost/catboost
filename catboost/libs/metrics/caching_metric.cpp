#include "caching_metric.h"
#include "metric.h"
#include "description_utils.h"
#include "classification_utils.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <util/generic/string.h>
#include <util/generic/set.h>

constexpr int BinaryClassesCount = 2;
static TString ConfusionMatrixCacheKey = "Confusion Matrix";

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
        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor
        ) const override {
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
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
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
    const TVector<TVector<double>>& approx,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    double border
) {
    const bool isMultiClass = approx.size() > 1;
    const int classesCount = isMultiClass ? approx.size() : BinaryClassesCount;

    const auto buildImpl = [&](auto useWeights, auto isMultiClass) {
        TMetricHolder confusionMatrix(classesCount * classesCount);
        for (int idx = begin; idx < end; ++idx) {
            int approxClass = GetApproxClass(approx, idx);
            int targetClass = isMultiClass ? static_cast<int>(target[idx]) : target[idx] > border;
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
    struct TMCCCachingMetric : public TCachingMetric {
        explicit TMCCCachingMetric(int classesCount)
            : ClassesCount(classesCount)
            , IsMultiClass(true) {
        }
        explicit TMCCCachingMetric(double border)
            : Border(border)
            , IsMultiClass(false) {
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
            TMaybe<TCache*> cache
        ) const override;
        double GetFinalError(const TMetricHolder &error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue *valueType, float *bestPossibleValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        int ClassesCount = BinaryClassesCount;
        double Border = 0.0;
        bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeMCCMetric(int classesCount) {
    return MakeHolder<TMCCCachingMetric>(classesCount);
}

TMetricHolder TMCCCachingMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
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
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end, Border);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), Border);

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

TString TMCCCachingMetric::GetDescription() const {
    if (true || IsMultiClass) {
        return BuildDescription(ELossFunction::MCC, UseWeights);
    } else {
        return BuildDescription(ELossFunction::MCC, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

void TMCCCachingMetric::GetBestValue(EMetricBestValue *valueType, float *) const {
    *valueType = EMetricBestValue::Max;
}

/* Zero one loss caching metric */

namespace {
    struct TZeroOneLossCachingMetric: public TCachingMetric {
        explicit TZeroOneLossCachingMetric(int classesCount)
            : ClassesCount(classesCount)
            , IsMultiClass(true) {
        }
        explicit TZeroOneLossCachingMetric(double border)
            : Border(border)
            , IsMultiClass(false) {
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
            TMaybe<TCache*> cache
        ) const override;
        double GetFinalError(const TMetricHolder &error) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue *valueType, float *bestPossibleValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        int ClassesCount = BinaryClassesCount;
        double Border = GetDefaultTargetBorder();
        bool IsMultiClass = false;
    };
}

TMetricHolder TZeroOneLossCachingMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
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
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end, Border);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), Border);

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

TString TZeroOneLossCachingMetric::GetDescription() const {
    if (IsMultiClass) {
        return BuildDescription(ELossFunction::ZeroOneLoss, UseWeights);
    } else {
        return BuildDescription(ELossFunction::ZeroOneLoss, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

void TZeroOneLossCachingMetric::GetBestValue(EMetricBestValue *valueType, float *) const {
    *valueType = EMetricBestValue::Min;
}

THolder<IMetric> MakeZeroOneLossMetric(double border) {
    return MakeHolder<TZeroOneLossCachingMetric>(border);
}

THolder<IMetric> MakeZeroOneLossMetric(int classCount) {
    return MakeHolder<TZeroOneLossCachingMetric>(classCount);
}

/* Accuracy caching metric */

namespace {
    struct TAccuracyCachingMetric : public TCachingMetric {
        explicit TAccuracyCachingMetric(double border)
            : Border(border)
            , IsMultiClass(false) {
        }
        explicit TAccuracyCachingMetric(int classesCount)
            : ClassesCount(classesCount)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TVector<TVector<double>> &approx,
            const TVector<TVector<double>> &approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        double Border = GetDefaultTargetBorder();
        int ClassesCount = BinaryClassesCount;
        bool IsMultiClass = false;
    };
}

TMetricHolder TAccuracyCachingMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
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
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end, Border);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), Border);

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

TString TAccuracyCachingMetric::GetDescription() const {
    if (IsMultiClass) {
        return BuildDescription(ELossFunction::Accuracy, UseWeights);
    } else {
        return BuildDescription(ELossFunction::Accuracy, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

void TAccuracyCachingMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
}

/* Recall caching metric */

namespace {
    struct TRecallCachingMetric: public TCachingMetric {
        explicit TRecallCachingMetric(double border)
            : ClassesCount(BinaryClassesCount)
            , Border(border)
            , IsMultiClass(false) {
        }

        explicit TRecallCachingMetric(int classesCount, int positiveClass)
            : ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            const TVector<TVector<double>>& approxDelta,
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
        int ClassesCount = BinaryClassesCount;
        int PositiveClass = 1;
        double Border = GetDefaultTargetBorder();
        bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassRecallMetric(double border) {
    return MakeHolder<TRecallCachingMetric>(border);
}

THolder<IMetric> MakeMultiClassRecallMetric(int classesCount, int positiveClass) {
    return MakeHolder<TRecallCachingMetric>(classesCount, positiveClass);
}

TMetricHolder TRecallCachingMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
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
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end, Border);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), Border);

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
        return BuildDescription(ELossFunction::Recall, UseWeights, "%.3g", MakeBorderParam(Border));
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
    struct TPrecisionCachingMetric: public TCachingMetric {
        explicit TPrecisionCachingMetric(double border)
            : ClassesCount(BinaryClassesCount)
            , Border(border)
            , IsMultiClass(false) {
        }

        explicit TPrecisionCachingMetric(int classesCount, int positiveClass)
            : ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TVector<TVector<double>> &approx,
            const TVector<TVector<double>> &approxDelta,
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
        int ClassesCount = BinaryClassesCount;
        int PositiveClass = 1;
        double Border = GetDefaultTargetBorder();
        bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassPrecisionMetric(double border) {
    return MakeHolder<TPrecisionCachingMetric>(border);
}

THolder<IMetric> MakeMultiClassPrecisionMetric(int classesCount, int positiveClass) {
    return MakeHolder<TPrecisionCachingMetric>(classesCount, positiveClass);
}

TMetricHolder TPrecisionCachingMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
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
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end, Border);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), Border);

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
        return BuildDescription(ELossFunction::Precision, UseWeights, "%.3g", MakeBorderParam(Border));
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
    struct TF1CachingMetric: public TCachingMetric {
        explicit TF1CachingMetric(double border)
            : ClassesCount(BinaryClassesCount)
            , Border(border)
            , IsMultiClass(false) {
        }

        explicit TF1CachingMetric(int classesCount, int positiveClass)
            : ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TVector<TVector<double>> &approx,
            const TVector<TVector<double>> &approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        TVector<TString> GetStatDescriptions() const override;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        int ClassesCount = BinaryClassesCount;
        int PositiveClass = 1;
        double Border = GetDefaultTargetBorder();
        bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassF1Metric(double border) {
    return MakeHolder<TF1CachingMetric>(border);
}

THolder<IMetric> MakeMultiClassF1Metric(int classesCount, int positiveClass) {
    return MakeHolder<TF1CachingMetric>(classesCount, positiveClass);
}

TMetricHolder TF1CachingMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
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
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end, Border);
    };
    auto confusionMatrix = cache.Empty() ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), Border);

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
        return BuildDescription(ELossFunction::F1, UseWeights, "%.3g", MakeBorderParam(Border));
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
    struct TTotalF1CachingMetric: public TCachingMetric {
        static constexpr int StatsCardinality = 3;

        explicit TTotalF1CachingMetric(double border)
            : ClassesCount(BinaryClassesCount)
            , Border(border)
            , IsMultiClass(false) {
        }

        explicit TTotalF1CachingMetric(int classesCount)
            : ClassesCount(classesCount)
            , IsMultiClass(true) {
        }

        TMetricHolder Eval(
            const TVector<TVector<double>> &approx,
            const TVector<TVector<double>> &approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        TVector<TString> GetStatDescriptions() const override;
        TString GetDescription() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

    private:
        int ClassesCount = BinaryClassesCount;
        double Border = 0.0;
        bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeTotalF1Metric(int classesCount) {
    return MakeHolder<TTotalF1CachingMetric>(classesCount);
}

TMetricHolder TTotalF1CachingMetric::Eval(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
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
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end, Border);
    };
    auto confusionMatrix = cache.Empty() || 1 ? MakeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, MakeMatrix, bool(UseWeights), Border);

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

TString TTotalF1CachingMetric::GetDescription() const {
    if (true || IsMultiClass) {
        return BuildDescription(ELossFunction::TotalF1, UseWeights);
    } else {
        return BuildDescription(ELossFunction::TotalF1, UseWeights, "%.3g", MakeBorderParam(Border));
    }
}

double TTotalF1CachingMetric::GetFinalError(const TMetricHolder& error) const {
    double numerator = 0.0;
    double denumerator = 0.0;

    for (auto i : xrange(ClassesCount)) {
        double classSize = error.Stats[StatsCardinality * i + 0];
        double classPredictedSize = error.Stats[StatsCardinality * i + 1];
        double truePositive = error.Stats[StatsCardinality * i + 2];

        double f1 = classSize + classPredictedSize != 0 ? 2 * truePositive / (classSize + classPredictedSize) : 0.0;
        numerator += f1 * classSize;
        denumerator += classSize;
    }

    return numerator / denumerator;
}

void TTotalF1CachingMetric::GetBestValue(EMetricBestValue* valueType, float*) const {
    *valueType = EMetricBestValue::Max;
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
        return metric->Eval(approx, approxDelta, isExpApprox, metric->NeedTarget() ? target[0] : TConstArrayRef<float>(),
                            weight, queriesInfo, from, to, cache);
    };
    const auto calcNonCaching = [&](auto metric, auto from, auto to) {
        CB_ENSURE(!metric->NeedTarget() || target.size() == 1, "Metric [" + metric->GetDescription() + "] requires "
                  << (target.size() > 1 ? "one-dimensional" : "") <<  "target");
        return metric->Eval(approx, approxDelta, isExpApprox, metric->NeedTarget() ? target[0] : TConstArrayRef<float>(),
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
static TVector<THolder<IMetric>> CreateMetric(int approxDimension) {
    TVector<THolder<IMetric>> result;
    if (approxDimension == 1) {
        result.emplace_back(MakeHolder<TMetricType>(GetDefaultTargetBorder()));
    } else {
        result.emplace_back(MakeHolder<TMetricType>(approxDimension));
    }
    return result;
}

template <typename TMetricType>
static TVector<THolder<IMetric>> CreateMetricClasswise(int approxDimension) {
    TVector<THolder<IMetric>> result;
    if (approxDimension == 1) {
        result.emplace_back(MakeHolder<TMetricType>(GetDefaultTargetBorder()));
    } else {
        for (int i : xrange(approxDimension)) {
            result.emplace_back(MakeHolder<TMetricType>(approxDimension, i));
        }
    }
    return result;
}

TVector<THolder<IMetric>> CreateCachingMetrics(ELossFunction metric, const TMap<TString, TString>& params, int approxDimension, TSet<TString>* validParams) {
    Y_UNUSED(params);
    *validParams = TSet<TString>{};

    switch(metric) {
        case ELossFunction::F1: {
            return CreateMetricClasswise<TF1CachingMetric>(approxDimension);
        }
        case ELossFunction::TotalF1: {
            return CreateMetric<TTotalF1CachingMetric>(approxDimension);
        }
        case ELossFunction::MCC: {
            return CreateMetric<TMCCCachingMetric>(approxDimension);
        }
        case ELossFunction::BrierScore: {
            CB_ENSURE(approxDimension == 1, "Brier Score is used only for binary classification problems.");
            TVector<THolder<IMetric>> result;
            result.emplace_back(MakeBrierScoreMetric());
            return result;
        }
        case ELossFunction::ZeroOneLoss: {
            return CreateMetric<TZeroOneLossCachingMetric>(approxDimension);
        }
        case ELossFunction::Accuracy: {
            return CreateMetric<TAccuracyCachingMetric>(approxDimension);
        }
        case ELossFunction::CtrFactor: {
            TVector<THolder<IMetric>> result;
            result.emplace_back(MakeCtrFactorMetric(GetDefaultTargetBorder()));
            return result;
        }
        case ELossFunction::Precision: {
            return CreateMetricClasswise<TPrecisionCachingMetric>(approxDimension);
        }
        case ELossFunction::Recall: {
            return CreateMetricClasswise<TRecallCachingMetric>(approxDimension);
        }
        default: {
            return {};
        }
    }
}
