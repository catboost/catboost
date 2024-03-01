#include "caching_metric.h"
#include "metric.h"
#include "description_utils.h"
#include "classification_utils.h"
#include "kappa.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <util/generic/string.h>
#include <util/generic/set.h>

using NCB::AppendTemporaryMetricsVector;
using NCB::AsVector;

constexpr int BinaryClassesCount = 2;
static const TString ConfusionMatrixCacheKey = "Confusion Matrix";
static const TString MultiLabelConfusionMatrixCacheKey = "MultiLabel Confusion Matrix";

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

    struct ICachingSingleTargetEval {
        virtual TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const = 0;
    };

    struct ICachingMultiTargetEval {
        virtual TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const = 0;
    };

    template <typename TEvalFunc>
    static TMetricHolder ParallelEvalIfPossible(
        const IMetric& metric,
        TEvalFunc&& evalFunc,
        int begin,
        int end,
        NPar::ILocalExecutor& executor
    ) {
        if (metric.IsAdditiveMetric()) {
            return ParallelEvalMetric(evalFunc, GetMinBlockSize(end - begin), begin, end, executor);
        } else {
            return evalFunc(begin, end);
        }
    }

    struct TCachingSingleTargetMetric: public TSingleTargetMetric, ICachingSingleTargetEval {
        explicit TCachingSingleTargetMetric(ELossFunction lossFunction, const TLossParams& params)
            : TSingleTargetMetric(lossFunction, params)
            {}
        using ICachingSingleTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::ILocalExecutor& executor
        ) const override {
            const auto evalMetric = [&](int from, int to) {
                return Eval(approx, approxDelta, isExpApprox, target, weight, queriesInfo, from, to, Nothing());
            };
            return ParallelEvalIfPossible(*this, evalMetric, begin, end, executor);
        }
    };

    struct TCachingMultiTargetMetric: public TMultiTargetMetric, ICachingMultiTargetEval {
        explicit TCachingMultiTargetMetric(ELossFunction lossFunction, const TLossParams& params)
            : TMultiTargetMetric(lossFunction, params)
            {}
        using ICachingMultiTargetEval::Eval;
        virtual TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end,
            NPar::ILocalExecutor& executor
        ) const override {
            const auto evalMetric = [&](int from, int to) {
                return Eval(approx, approxDelta, target, weight, from, to, Nothing());
            };
            return ParallelEvalIfPossible(*this, evalMetric, begin, end, executor);
        }
    };

    struct TCachingUniversalMetric: public TUniversalMetric, ICachingSingleTargetEval, ICachingMultiTargetEval {
        explicit TCachingUniversalMetric(ELossFunction lossFunction, const TLossParams& params)
            : TUniversalMetric(lossFunction, params)
            {}
        using ISingleTargetEval::Eval;
        using ICachingSingleTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::ILocalExecutor& executor
        ) const override {
            const auto evalMetric = [&](int from, int to) {
                return Eval(approx, approxDelta, isExpApprox, target, weight, queriesInfo, from, to, Nothing());
            };
            return ParallelEvalIfPossible(*this, evalMetric, begin, end, executor);
        }

        using IMultiTargetEval::Eval;
        using ICachingMultiTargetEval::Eval;
        virtual TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end,
            NPar::ILocalExecutor& executor
        ) const override {
            const auto evalMetric = [&](int from, int to) {
                return Eval(approx, approxDelta, target, weight, from, to, Nothing());
            };
            return ParallelEvalIfPossible(*this, evalMetric, begin, end, executor);
        }
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

/* MultiLabel Confusion matrix */
constexpr size_t BinaryConfusionMatrixSize = 4;

static TMetricHolder BuildConfusionMatrix(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    double targetBorder,
    double predictionBorder
) {
    const int classesCount = approx.size();
    TMetricHolder confusionMatrix(BinaryConfusionMatrixSize * classesCount);
    for (int classIdx = 0; classIdx < classesCount; ++classIdx) {
        TConstArrayRef<TConstArrayRef<double>> classApprox(&approx[classIdx], 1);
        TMetricHolder classConfusionMatrix = BuildConfusionMatrix(
            classApprox, target[classIdx], weight, begin, end, targetBorder, predictionBorder
        );
        for (size_t i = 0; i < BinaryConfusionMatrixSize; ++i) {
            confusionMatrix.Stats[classIdx * BinaryConfusionMatrixSize + i] = classConfusionMatrix.Stats[i];
        }
    }
    return confusionMatrix;
}

/* MCC caching metric */

namespace {
    struct TMCCCachingMetric final : public TCachingSingleTargetMetric {
        explicit TMCCCachingMetric(const TLossParams& params,
                                   int classesCount)
            : TCachingSingleTargetMetric(ELossFunction::MCC, params)
            , ClassesCount(classesCount) {
        }
        explicit TMCCCachingMetric(const TLossParams& params,
                                   double predictionBorder)
            : TCachingSingleTargetMetric(ELossFunction::MCC, params)
            , PredictionBorder(predictionBorder)
        {
        }
        using TSingleTargetMetric::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
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
        static TVector<TParamSet> ValidParamSets();

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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights),
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

void TMCCCachingMetric::GetBestValue(EMetricBestValue *valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TMCCCachingMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Zero one loss caching metric */

namespace {
    struct TZeroOneLossCachingMetric final: public TCachingSingleTargetMetric {
        explicit TZeroOneLossCachingMetric(const TLossParams& params,
                                           int classesCount)
            : TCachingSingleTargetMetric(ELossFunction::ZeroOneLoss, params)
            , ClassesCount(classesCount) {
        }
        explicit TZeroOneLossCachingMetric(const TLossParams& params,
                                           double predictionBorder)
            : TCachingSingleTargetMetric(ELossFunction::ZeroOneLoss, params)
            , PredictionBorder(predictionBorder) {
        }
        using TSingleTargetMetric::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

        static TVector<TParamSet> ValidParamSets();

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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights),
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

void TZeroOneLossCachingMetric::GetBestValue(EMetricBestValue *valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TZeroOneLossCachingMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Accuracy caching metric */

namespace {
    struct TAccuracyCachingMetric final: public TCachingUniversalMetric {
        explicit TAccuracyCachingMetric(const TLossParams& params,
                                        double predictionBorder)
            : TCachingUniversalMetric(ELossFunction::Accuracy, params)
            , PredictionBorder(predictionBorder) {
        }
        explicit TAccuracyCachingMetric(const TLossParams& params,
                                        int classesCount)
            : TCachingUniversalMetric(ELossFunction::Accuracy, params)
            , ClassesCount(classesCount) {
        }
        explicit TAccuracyCachingMetric(const TLossParams& params,
                                        int classesCount,
                                        int classIdx)
            : TCachingUniversalMetric(ELossFunction::Accuracy, params)
            , ClassesCount(classesCount)
            , AccuracyType(EAccuracyType::PerClass)
            , ClassIdx(classIdx) {
        }
        using ISingleTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        using IMultiTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const override;

        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        TString GetDescription() const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

        static TVector<TParamSet> ValidParamSets();

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const double PredictionBorder = GetDefaultPredictionBorder();
        const int ClassesCount = BinaryClassesCount;
        const EAccuracyType AccuracyType = EAccuracyType::Classic;
        const int ClassIdx = 0;
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
    CB_ENSURE(AccuracyType == EAccuracyType::Classic, "PerClass accuracy is meaningfull for multilabel only");

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights),
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

TMetricHolder TAccuracyCachingMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(approxDelta.empty());

    TMetricHolder error(2);
    if (AccuracyType == EAccuracyType::PerClass) {
        const auto makeMatrix = [&]() {
            return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                    TargetBorder, PredictionBorder);
        };
        auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(MultiLabelConfusionMatrixCacheKey,
                                                                                  makeMatrix, bool(UseWeights),
                                                                                  TargetBorder, PredictionBorder);

        const auto getStats = [&](int i, int j) {
            return confusionMatrix.Stats[ClassIdx * BinaryConfusionMatrixSize + i * 2 + j];
        };
        for (auto i : xrange(2)) {
            error.Stats[0] += getStats(i, i);
            for (auto j : xrange(2)) {
                error.Stats[1] += getStats(i, j);
            }
        }
    } else {
        const double predictionLogitBorder = NCB::Logit(PredictionBorder);
        const bool useWeights = !weight.empty();

        for (int idx = begin; idx < end; ++idx) {
            bool correct = true;
            for (int targetDim = 0; targetDim < target.ysize() && correct; ++targetDim) {
                const int approxClass = approx[targetDim][idx] > predictionLogitBorder;
                const int targetClass = target[targetDim][idx] > TargetBorder;
                correct &= approxClass == targetClass;
            }
            const double w = useWeights ? weight[idx] : 1.0f;
            error.Stats[0] += w * correct;
            error.Stats[1] += w;
        }
    }
    return error;
}

void TAccuracyCachingMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TString TAccuracyCachingMetric::GetDescription() const {
    if (AccuracyType == EAccuracyType::PerClass) {
        const TMetricParam<int> classIdx("class", ClassIdx, /*userDefined*/true);
        return BuildDescription(ELossFunction::Accuracy, UseWeights, classIdx);
    } else {
        return TCachingUniversalMetric::GetDescription();
    }
}

TVector<TParamSet> TAccuracyCachingMetric::ValidParamSets() {
    return {TParamSet{
        {
            TParamInfo{"use_weights", false, true},
            TParamInfo{"type", false, ToString(EAccuracyType::Classic)}
        },
        ""
    }};
};

/* HammingLoss caching metric */

namespace {
    struct THammingLossCachingMetric final: public TCachingUniversalMetric {
        explicit THammingLossCachingMetric(const TLossParams& params,
                                           double predictionBorder)
            : TCachingUniversalMetric(ELossFunction::HammingLoss, params)
            , PredictionBorder(predictionBorder) {
        }
        explicit THammingLossCachingMetric(const TLossParams& params,
                                           int classesCount)
            : TCachingUniversalMetric(ELossFunction::HammingLoss, params)
            , ClassesCount(classesCount) {
        }

        using ISingleTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        using IMultiTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const override;

        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

        static TVector<TParamSet> ValidParamSets();

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const double PredictionBorder = GetDefaultPredictionBorder();
        const int ClassesCount = BinaryClassesCount;
    };
}

TMetricHolder THammingLossCachingMetric::Eval(
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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights),
            TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(2);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[0] += getStats(i, i);
        for (auto j : xrange(ClassesCount)) {
            error.Stats[1] += getStats(i, j);
        }
    }
    error.Stats[0] = error.Stats[1] - error.Stats[0];
    return error;
}

TMetricHolder THammingLossCachingMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(approxDelta.empty());

    TMetricHolder error(2);
    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(MultiLabelConfusionMatrixCacheKey,
                                                                              makeMatrix, bool(UseWeights),
                                                                              TargetBorder, PredictionBorder);

    for (auto classIdx : xrange(ClassesCount)) {
        const auto getStats = [&](int i, int j) {
            return confusionMatrix.Stats[classIdx * BinaryConfusionMatrixSize + i * 2 + j];
        };
        for (auto i : xrange(2)) {
            error.Stats[0] += getStats(i, 1 - i);
            for (auto j : xrange(2)) {
                error.Stats[1] += getStats(i, j);
            }
        }
    }

    return error;
}

void THammingLossCachingMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> THammingLossCachingMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"border", false, TargetBorder}
            },
            ""
        }
    };
};

/* Recall caching metric */

namespace {
    struct TRecallCachingMetric final: public TCachingUniversalMetric {
        explicit TRecallCachingMetric(const TLossParams& params,
                                      double predictionBorder)
            : TCachingUniversalMetric(ELossFunction::Recall, params)
            , ClassesCount(BinaryClassesCount)
            , PredictionBorder(predictionBorder)
            , IsMultiClass(false) {
        }

        explicit TRecallCachingMetric(const TLossParams& params,
                                      int classesCount, int positiveClass)
            : TCachingUniversalMetric(ELossFunction::Recall, params)
            , ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        using ISingleTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const override;

        using IMultiTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
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

        static TVector<TParamSet> ValidParamSets();

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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(2);
    error.Stats[0] = getStats(PositiveClass, PositiveClass);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[1] += getStats(i, PositiveClass);
    }
    return error;
}

TMetricHolder TRecallCachingMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(approxDelta.empty());

    TMetricHolder error(2);
    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(MultiLabelConfusionMatrixCacheKey,
                                                                              makeMatrix, bool(UseWeights),
                                                                              TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) {
        return confusionMatrix.Stats[PositiveClass * BinaryConfusionMatrixSize + i * 2 + j];
    };
    error.Stats[0] = getStats(1, 1);
    error.Stats[1] = getStats(1, 1) + getStats(0, 1);

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

void TRecallCachingMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TRecallCachingMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Precision caching metric */

namespace {
    struct TPrecisionCachingMetric final: public TCachingUniversalMetric {
        explicit TPrecisionCachingMetric(const TLossParams& params,
                                         double predictionBorder)
            : TCachingUniversalMetric(ELossFunction::Precision, params)
            , ClassesCount(BinaryClassesCount)
            , PredictionBorder(predictionBorder)
            , IsMultiClass(false) {
        }

        explicit TPrecisionCachingMetric(const TLossParams& params,
                                         int classesCount, int positiveClass)
            : TCachingUniversalMetric(ELossFunction::Precision, params)
            , ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }

        using ISingleTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        using IMultiTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
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

        static TVector<TParamSet> ValidParamSets();

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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(2);
    error.Stats[0] = getStats(PositiveClass, PositiveClass);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[1] += getStats(PositiveClass, i);
    }

    return error;
}

TMetricHolder TPrecisionCachingMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(approxDelta.empty());

    TMetricHolder error(2);
    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(MultiLabelConfusionMatrixCacheKey,
                                                                              makeMatrix, bool(UseWeights),
                                                                              TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) {
        return confusionMatrix.Stats[PositiveClass * BinaryConfusionMatrixSize + i * 2 + j];
    };
    error.Stats[0] = getStats(1, 1);
    error.Stats[1] = getStats(1, 1) + getStats(1, 0);

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
    if (error.Stats[1] == 0) {
        CATBOOST_WARNING_LOG << "Number of the positive class predictions is 0. "
            "Setting Precision metric value to the default 0\n";
        return 0.0;
    } else {
        return error.Stats[0] / error.Stats[1];
    }
}

void TPrecisionCachingMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TPrecisionCachingMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* F beta caching metric */

namespace {
    struct TFCachingMetric: public TCachingUniversalMetric {
        explicit TFCachingMetric(const TLossParams& params, double beta,
                                 double predictionBorder)
            : TCachingUniversalMetric(ELossFunction::F, params)
            , ClassesCount(BinaryClassesCount)
            , Beta(beta)
            , PredictionBorder(predictionBorder)
            , IsMultiClass(false)
        {
            Y_ASSERT(Beta > 0);
        }

        explicit TFCachingMetric(const TLossParams& params, double beta,
                                 int classesCount, int positiveClass)
            : TCachingUniversalMetric(ELossFunction::F, params)
            , ClassesCount(classesCount)
            , PositiveClass(positiveClass)
            , Beta(beta)
            , IsMultiClass(true)
        {
            Y_ASSERT(Beta > 0);
        }

        using ISingleTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int begin,
            int end,
            TMaybe<TCache *> cache
        ) const override;

        using IMultiTargetEval::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end,
            TMaybe<TCache*> cache
        ) const override;

        TString GetDescription() const override;
        TVector<TString> GetStatDescriptions() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const override {
            return true;
        }

        static TVector<TParamSet> ValidParamSets();

    protected:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int ClassesCount = BinaryClassesCount;
        const int PositiveClass = 1;
        const double Beta = 1;
        const double PredictionBorder = GetDefaultPredictionBorder();
        const bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassFMetric(const TLossParams& params, double beta,
                                     double predictionBorder) {
    return MakeHolder<TFCachingMetric>(params, beta, predictionBorder);
}

THolder<IMetric> MakeMultiClassFMetric(const TLossParams& params, double beta,
                                       int classesCount, int positiveClass) {
    return MakeHolder<TFCachingMetric>(params, beta, classesCount, positiveClass);
}

TMetricHolder TFCachingMetric::Eval(
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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                                    TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) { return confusionMatrix.Stats[i * ClassesCount + j]; };
    TMetricHolder error(3);
    error.Stats[0] = getStats(PositiveClass, PositiveClass);
    for (auto i : xrange(ClassesCount)) {
        error.Stats[1] += getStats(PositiveClass, i);
        error.Stats[2] += getStats(i, PositiveClass);
    }
    return error;
}

TMetricHolder TFCachingMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end,
    TMaybe<TCache*> cache
) const {
    Y_ASSERT(approxDelta.empty());

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                                    TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() ? makeMatrix() : cache.GetRef()->Get(MultiLabelConfusionMatrixCacheKey,
                                                                              makeMatrix, bool(UseWeights),
                                                                              TargetBorder, PredictionBorder);

    const auto getStats = [&](int i, int j) {
        return confusionMatrix.Stats[PositiveClass * BinaryConfusionMatrixSize + i * 2 + j];
    };
    TMetricHolder error(3);
    error.Stats[0] = getStats(1, 1);
    error.Stats[1] = getStats(1, 1) + getStats(1, 0);
    error.Stats[2] = getStats(1, 1) + getStats(0, 1);
    return error;
}

TVector<TString> TFCachingMetric::GetStatDescriptions() const {
    return {"TP", "TP+FP", "TP+FN"};
}

TString TFCachingMetric::GetDescription() const {
    const TMetricParam<double> beta("beta", Beta, /*userDefined*/true);
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::F, UseWeights, "%.3g", beta, positiveClass);
    } else {
        return BuildDescription(ELossFunction::F, UseWeights, "%.3g", beta,"%.3g", MakeTargetBorderParam(TargetBorder),
                                MakePredictionBorderParam(PredictionBorder));
    }
}

double TFCachingMetric::GetFinalError(const TMetricHolder& error) const {
    double precision = error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1.0;
    double recall = error.Stats[2] != 0 ? error.Stats[0] / error.Stats[2] : 1.0;
    double beta_square = Beta * Beta;
    return precision + recall != 0 ? (1 + beta_square) * precision * recall / (beta_square * precision + recall) : 0.0;
}

void TFCachingMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TFCachingMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"beta", true, {}}
            },
            ""
        }
    };
}

/* F1 caching metric */

namespace {
    struct TF1CachingMetric final : public TFCachingMetric {
        explicit TF1CachingMetric(const TLossParams &params,
                                  double predictionBorder)
            : TFCachingMetric(params, 1.0, predictionBorder) {
        }

        explicit TF1CachingMetric(const TLossParams &params,
                                  int classesCount, int positiveClass)
            : TFCachingMetric(params, 1.0, classesCount, positiveClass) {
        }
        TString GetDescription() const override;

        static TVector<TParamSet> ValidParamSets();
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



TString TF1CachingMetric::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::F1, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::F1, UseWeights, "%.3g", MakeTargetBorderParam(TargetBorder),
                                MakePredictionBorderParam(PredictionBorder));
    }
}


TVector<TParamSet> TF1CachingMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* TotalF1 caching metric */

namespace {
    struct TTotalF1CachingMetric final: public TCachingSingleTargetMetric {
        static constexpr int StatsCardinality = 3;

        explicit TTotalF1CachingMetric(const TLossParams& params,
                                       double predictionBorder, EF1AverageType averageType)
            : TCachingSingleTargetMetric(ELossFunction::TotalF1, params)
            , ClassesCount(BinaryClassesCount)
            , PredictionBorder(predictionBorder)
            , AverageType(averageType) {
        }

        explicit TTotalF1CachingMetric(const TLossParams& params,
                                       int classesCount, EF1AverageType averageType)
            : TCachingSingleTargetMetric(ELossFunction::TotalF1, params)
            , ClassesCount(classesCount)
            , AverageType(averageType) {
        }

        using TSingleTargetMetric::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

        static TVector<TParamSet> ValidParamSets();

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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    auto confusionMatrix = cache.Empty() || 1 ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);

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

void TTotalF1CachingMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TTotalF1CachingMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"average", false, ToString(EF1AverageType::Weighted)}
            },
            ""
        }
    };
};

/* Kappa */

namespace {
    struct TKappaMetric final: public TCachingSingleTargetMetric {
        explicit TKappaMetric(const TLossParams& params,
                              int classCount = 2, double predictionBorder = GetDefaultPredictionBorder())
            : TCachingSingleTargetMetric(ELossFunction::Kappa, params)
            , TargetBorder(GetDefaultTargetBorder())
            , PredictionBorder(predictionBorder)
            , ClassCount(classCount) {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        using TSingleTargetMetric::Eval;
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
    if (config.ApproxDimension == 1) {
        config.ValidParams->insert("border");
        return AsVector(MakeHolder<TKappaMetric>(config.Params, /*classCount=*/2, config.GetPredictionBorderOrDefault()));
    } else {
        return AsVector(MakeHolder<TKappaMetric>(config.Params, config.ApproxDimension));
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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    return cache.Empty() || 1 ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);
}

TString TKappaMetric::GetDescription() const {
    return BuildDescription(ELossFunction::Kappa, "%.3g", MakeTargetBorderParam(TargetBorder),
                            MakePredictionBorderParam(PredictionBorder));
}

void TKappaMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

double TKappaMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Cohen);
}

TVector<TParamSet> TKappaMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* WKappa */

namespace {
    struct TWKappaMetric final: public TCachingSingleTargetMetric {
        explicit TWKappaMetric(const TLossParams& params,
                               int classCount = 2, double predictionBorder = GetDefaultPredictionBorder())
            : TCachingSingleTargetMetric(ELossFunction::WKappa, params)
            , TargetBorder(GetDefaultTargetBorder())
            , PredictionBorder(predictionBorder)
            , ClassCount(classCount) {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);

        static TVector<TParamSet> ValidParamSets();

        using TSingleTargetMetric::Eval;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
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
    if (config.ApproxDimension == 1) {
        config.ValidParams->insert("border");
        return AsVector(MakeHolder<TWKappaMetric>(config.Params, /*classCount=*/2, config.GetPredictionBorderOrDefault()));
    } else {
        return AsVector(MakeHolder<TWKappaMetric>(config.Params, config.ApproxDimension));
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

    const auto makeMatrix = [&]() {
        return BuildConfusionMatrix(approx, target, UseWeights ? weight : TVector<float>{}, begin, end,
                TargetBorder, PredictionBorder);
    };
    return cache.Empty() || 1 ? makeMatrix() : cache.GetRef()->Get(ConfusionMatrixCacheKey, makeMatrix, bool(UseWeights), TargetBorder, PredictionBorder);
}

TString TWKappaMetric::GetDescription() const {
    return BuildDescription(ELossFunction::WKappa, "%.3g", MakeTargetBorderParam(TargetBorder),
                            MakePredictionBorderParam(PredictionBorder));
}

void TWKappaMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

double TWKappaMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcKappa(error, ClassCount, EKappaMetricType::Weighted);
}

TVector<TParamSet> TWKappaMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

TVector<TMetricHolder> EvalErrorsWithCaching(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    TConstArrayRef<const IMetric*> metrics,
    NPar::ILocalExecutor* localExecutor
) {
    const auto threadCount = localExecutor->GetThreadCount() + 1;
    const auto objectCount = approx.front().size();
    const auto queryCount = queriesInfo.size();

    const auto approxRef = To2DConstArrayRef<double>(approx);
    const auto approxDeltaRef = To2DConstArrayRef<double>(approxDelta);

    const auto calcCaching = [&](auto metric, auto from, auto to, auto *cache) {
        if (target.size() <= 1 && dynamic_cast<const ICachingSingleTargetEval*>(metric) != nullptr) {
            CB_ENSURE(!metric->NeedTarget() || target.size() == 1, "Metric [" + metric->GetDescription() + "] requires "
                    << (target.size() > 1 ? "one-dimensional" : "") <<  "target");
            return dynamic_cast<const ICachingSingleTargetEval*>(metric)->Eval(
                approxRef, approxDeltaRef, isExpApprox,
                metric->NeedTarget() ? target[0] : TConstArrayRef<float>(),
                weight, queriesInfo, from, to, cache
            );
        } else {
            CB_ENSURE(!isExpApprox, "Metric [" << metric->GetDescription() << "] does not support exponentiated approxes");
            return dynamic_cast<const ICachingMultiTargetEval*>(metric)->Eval(
                approxRef, approxDeltaRef, target, weight, from, to, cache
            );
        }
    };
    const auto calcNonCaching = [&](auto metric, auto from, auto to) {
        if (target.size() <= 1 && dynamic_cast<const ISingleTargetEval*>(metric) != nullptr) {
            CB_ENSURE(!metric->NeedTarget() || target.size() == 1, "Metric [" + metric->GetDescription() + "] requires "
                    << (target.size() > 1 ? "one-dimensional" : "") <<  "target");
            return dynamic_cast<const ISingleTargetEval*>(metric)->Eval(
                approxRef, approxDeltaRef, isExpApprox,
                metric->NeedTarget() ? target[0] : TConstArrayRef<float>(),
                weight, queriesInfo, from, to, *localExecutor
            );
        } else {
            CB_ENSURE(!isExpApprox, "Metric [" << metric->GetDescription() << "] does not support exponentiated approxes");
            return dynamic_cast<const IMultiTargetEval*>(metric)->Eval(
                approxRef, approxDeltaRef, target, weight, from, to, *localExecutor
            );
        }
    };

    TVector<TMetricHolder> errors;
    errors.reserve(metrics.size());

    NPar::ILocalExecutor::TExecRangeParams objectwiseBlockParams(0, objectCount);
    if (!target.empty()) {
        const auto objectwiseEffectiveBlockCount = Min(threadCount, int(ceil(double(objectCount) / GetMinBlockSize(objectCount))));
        objectwiseBlockParams.SetBlockCount(objectwiseEffectiveBlockCount);
    }

    NPar::ILocalExecutor::TExecRangeParams querywiseBlockParams(0, queryCount);
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

        const bool isObjectwise = metric->GetErrorType() == EErrorType::PerObjectError;
        const auto end = isObjectwise ? objectCount : queryCount;
        CB_ENSURE(end > 0, "Not enough data to calculate metric: groupwise metric w/o group id's, or objectwise metric w/o samples");

        const bool isCaching = dynamic_cast<const ICachingSingleTargetEval*>(metric)
                            || dynamic_cast<const ICachingMultiTargetEval*>(metric);

        if (isCaching && metric->IsAdditiveMetric()) {
            const auto blockSize = isObjectwise ? objectwiseBlockParams.GetBlockSize() : querywiseBlockParams.GetBlockSize();
            const auto blockCount = isObjectwise ? objectwiseBlockParams.GetBlockCount() : querywiseBlockParams.GetBlockCount();

            auto &results = isObjectwise ? objectwiseBlockResults : querywiseBlockResults;
            auto &cache = isObjectwise ? objectwiseAdditiveCache : querywiseAdditiveCache;

            NPar::ParallelFor(*localExecutor, 0, blockCount, [&](auto blockId) {
                const auto from = blockId * blockSize;
                const auto to = Min<int>((blockId + 1) * blockSize, end);
                results[blockId] = calcCaching(metric, from, to, &cache[blockId]);
            });

            TMetricHolder error;
            for (const auto &blockResult : results) {
                error.Add(blockResult);
            }
            errors.push_back(error);
        } else {
            if (isCaching) {
                errors.push_back(calcCaching(metric, 0, end, &nonAdditiveCache));
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
        result.emplace_back(MakeHolder<TMetricType>(config.Params, config.GetPredictionBorderOrDefault()));
    } else {
        result.emplace_back(MakeHolder<TMetricType>(config.Params, approxDimension));
    }
    return result;
}

template <typename TMetricType>
static TVector<THolder<IMetric>> CreateMetricClasswise(int approxDimension, const TMetricConfig& config) {
    TVector<THolder<IMetric>> result;
    if (approxDimension == 1) {
        result.emplace_back(MakeHolder<TMetricType>(config.Params, config.GetPredictionBorderOrDefault()));
    } else {
        for (int i : xrange(approxDimension)) {
            result.emplace_back(MakeHolder<TMetricType>(config.Params, approxDimension, i));
        }
    }
    return result;
}

TVector<THolder<IMetric>> CreateCachingMetrics(const TMetricConfig& config) {
    *config.ValidParams = TSet<TString>{};
    TVector<THolder<IMetric>> result;

    switch(config.Metric) {
        case ELossFunction::F1: {
            return CreateMetricClasswise<TF1CachingMetric>(config.ApproxDimension, config);
        }
        case ELossFunction::F:  {
            CB_ENSURE(config.GetParamsMap().contains("beta"), "Metric " << ELossFunction::F << " requires beta as parameter");
            config.ValidParams->insert("beta");
            double beta_param = FromString<float>(config.GetParamsMap().at("beta"));

            if (config.ApproxDimension == 1) {
                result.emplace_back(MakeHolder<TFCachingMetric>(config.Params, beta_param, config.GetPredictionBorderOrDefault()));
            } else {
                for (int i : xrange(config.ApproxDimension)) {
                    result.emplace_back(MakeHolder<TFCachingMetric>(config.Params, beta_param, config.ApproxDimension, i));
                }
            }
            return result;
        }
        case ELossFunction::TotalF1: {
            config.ValidParams->insert("average");
            EF1AverageType averageType = EF1AverageType::Weighted;
            if (config.GetParamsMap().contains("average")) {
                averageType = FromString<EF1AverageType>(config.GetParamsMap().at("average"));
            }

            if (config.ApproxDimension == 1) {
                result.emplace_back(MakeHolder<TTotalF1CachingMetric>(config.Params, config.GetPredictionBorderOrDefault(), averageType));
            } else {
                result.emplace_back(MakeHolder<TTotalF1CachingMetric>(config.Params, config.ApproxDimension, averageType));
            }
            return result;
        }
        case ELossFunction::MCC: {
            return CreateMetric<TMCCCachingMetric>(config.ApproxDimension, config);
        }
        case ELossFunction::BrierScore: {
            CB_ENSURE(config.ApproxDimension == 1, "Brier Score is used only for binary classification problems.");
            result.emplace_back(MakeBrierScoreMetric(config.Params));
            return result;
        }
        case ELossFunction::ZeroOneLoss: {
            return CreateMetric<TZeroOneLossCachingMetric>(config.ApproxDimension, config);
        }
        case ELossFunction::HammingLoss: {
            return CreateMetric<THammingLossCachingMetric>(config.ApproxDimension, config);
        }
        case ELossFunction::Accuracy: {
            config.ValidParams->insert("type");
            EAccuracyType accuracyType = EAccuracyType::Classic;
            if (config.GetParamsMap().contains("type")) {
                accuracyType = FromString<EAccuracyType>(config.GetParamsMap().at("type"));
            }
            if (accuracyType == EAccuracyType::Classic) {
                return CreateMetric<TAccuracyCachingMetric>(config.ApproxDimension, config);
            } else if (accuracyType == EAccuracyType::PerClass) {
                return CreateMetricClasswise<TAccuracyCachingMetric>(config.ApproxDimension, config);
            } else {
                CB_ENSURE_INTERNAL(false, "Unhandled accuracy type " << accuracyType);
            }
        }
        case ELossFunction::CtrFactor: {
            result.emplace_back(MakeCtrFactorMetric(config.Params));
            return result;
        }
        case ELossFunction::Precision: {
            return CreateMetricClasswise<TPrecisionCachingMetric>(config.ApproxDimension, config);
        }
        case ELossFunction::Recall:
            return CreateMetricClasswise<TRecallCachingMetric>(config.ApproxDimension, config);
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

TVector<TParamSet> CachingMetricValidParamSets(ELossFunction metric) {
    switch (metric) {
        case ELossFunction::F:
            return TFCachingMetric::ValidParamSets();
        case ELossFunction::F1:
            return TF1CachingMetric::ValidParamSets();
        case ELossFunction::TotalF1:
            return TTotalF1CachingMetric::ValidParamSets();
        case ELossFunction::MCC:
            return TMCCCachingMetric::ValidParamSets();
        case ELossFunction::ZeroOneLoss:
            return TZeroOneLossCachingMetric::ValidParamSets();
        case ELossFunction::Accuracy:
            return TAccuracyCachingMetric::ValidParamSets();
        case ELossFunction::HammingLoss:
            return THammingLossCachingMetric::ValidParamSets();
        case ELossFunction::Precision:
            return TPrecisionCachingMetric::ValidParamSets();
        case ELossFunction::Recall:
            return TRecallCachingMetric::ValidParamSets();
        case ELossFunction::Kappa:
            return TKappaMetric::ValidParamSets();
        case ELossFunction::WKappa:
            return TWKappaMetric::ValidParamSets();
        default:
            CB_ENSURE(false, "Unsupported metric: " << metric);
    }
}
