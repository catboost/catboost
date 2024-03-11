#include "metric.h"
#include "caching_metric.h"
#include "auc.h"
#include "auc_mu.h"
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
#include "description_utils.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/short_vector_ops.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>

#include <library/cpp/fast_exp/fast_exp.h>
#include <library/cpp/fast_log/fast_log.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/ymath.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/string/printf.h>
#include <util/system/yassert.h>

#include <limits>
#include <tuple>


using NCB::AppendTemporaryMetricsVector;
using NCB::AsVector;

/* TMetric */

static inline double OverflowSafeLogitProb(double approx) {
    double expApprox = exp(approx);
    return approx < 200 ? expApprox / (1.0 + expApprox) : 1.0;
}

TMetric::TMetric(ELossFunction lossFunction, TLossParams descriptionParams)
    : LossFunction(lossFunction)
    , DescriptionParams(std::move(descriptionParams)) {
}

EErrorType TMetric::GetErrorType() const {
    return EErrorType::PerObjectError;
}

double TMetric::GetFinalError(const TMetricHolder& error) const {
    Y_ASSERT(error.Stats.size() == 2);
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
}

TVector<TString> TMetric::GetStatDescriptions() const {
    return {"SumError", "SumWeight"};
}

const TMap<TString, TString>& TMetric::GetHints() const {
    return Hints;
}

TString TMetric::GetDescription() const {
    TLossParams descriptionParamsCopy = DescriptionParams;
    descriptionParamsCopy.Erase("hints");
    if (UseWeights.IsUserDefined()) {
        descriptionParamsCopy.Put(UseWeights.GetName(), UseWeights.Get() ? "true" : "false");
    }
    return BuildDescriptionFromParams(LossFunction, descriptionParamsCopy);
}

void TMetric::AddHint(const TString& key, const TString& value) {
    Hints[key] = value;
}

bool TMetric::NeedTarget() const {
    return GetErrorType() != EErrorType::PairwiseError;
}


namespace {
    struct TAdditiveMultiTargetMetric: public TMultiTargetMetric {
        explicit TAdditiveMultiTargetMetric(ELossFunction lossFunction, const TLossParams& descriptionParams)
            : TMultiTargetMetric(lossFunction, descriptionParams) {}
        bool IsAdditiveMetric() const final {
            return true;
        }
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
                return EvalSingleThread(
                    approx, approxDelta, target, UseWeights.IsIgnored() || UseWeights ? weight : TVector<float>{}, from, to
                );
            };
            return ParallelEvalMetric(evalMetric, GetMinBlockSize(end - begin), begin, end, executor);
        }
        virtual TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end
        ) const = 0;
    };

    struct TAdditiveSingleTargetMetric: public TSingleTargetMetric {
        explicit TAdditiveSingleTargetMetric(ELossFunction lossFunction, const TLossParams& descriptionParams)
            : TSingleTargetMetric(lossFunction, descriptionParams) {}
        bool IsAdditiveMetric() const final {
            return true;
        }
        virtual TMetricHolder Eval(
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
                return EvalSingleThread(
                    approx, approxDelta, isExpApprox, target, UseWeights.IsIgnored() || UseWeights ? weight : TVector<float>{}, queriesInfo, from, to
                );
            };
            return ParallelEvalMetric(evalMetric, GetMinBlockSize(end - begin), begin, end, executor);
        }
        virtual TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const = 0;
    };

    struct TNonAdditiveSingleTargetMetric: public TSingleTargetMetric {
        explicit TNonAdditiveSingleTargetMetric(ELossFunction lossFunction, const TLossParams& descriptionParams)
            : TSingleTargetMetric(lossFunction, descriptionParams) {}
        bool IsAdditiveMetric() const final {
            return false;
        }
    };
}

static inline TConstArrayRef<double> GetRowRef(TConstArrayRef<TConstArrayRef<double>> matrix, size_t rowIdx) {
    if (matrix.empty()) {
        return TArrayRef<double>();
    } else {
        return matrix[rowIdx];
    }
}

/* CrossEntropy */

namespace {
    struct TCrossEntropyMetric final: public TAdditiveSingleTargetMetric {
        explicit TCrossEntropyMetric(ELossFunction lossFunction, const TLossParams& params);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        ELossFunction LossFunction;
    };
} // anonymous namespace

TVector<THolder<IMetric>> TCrossEntropyMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TCrossEntropyMetric>(config.Metric, config.Params));
}

TCrossEntropyMetric::TCrossEntropyMetric(ELossFunction lossFunction, const TLossParams& params)
        : TAdditiveSingleTargetMetric(lossFunction, params)
        , LossFunction(lossFunction)
{
    CB_ENSURE_INTERNAL(
        lossFunction == ELossFunction::Logloss || lossFunction == ELossFunction::CrossEntropy,
        "lossFunction " << lossFunction
    );
    if (lossFunction == ELossFunction::CrossEntropy) {
        CB_ENSURE(TargetBorder == GetDefaultTargetBorder(), "TargetBorder is meaningless for crossEntropy metric");
    }
}

TMetricHolder TCrossEntropyMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
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

    CB_ENSURE(approxRef.size() == 1, "Metric logloss supports only single-dimensional data");

    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight, auto isLogloss) {
        float targetBorder = TargetBorder;
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
        int tailBegin;
        auto holder = NMixedSimdOps::EvalCrossEntropyVectorized(
            isExpApprox,
            hasDelta,
            hasWeight,
            isLogloss,
            approx,
            approxDelta,
            target,
            weight,
            targetBorder,
            begin,
            end,
            &tailBegin);
        for (int i = tailBegin; i < end; ++i) {
            const float w = hasWeight ? weight[i] : 1;
            const float prob = isLogloss ? target[i] > targetBorder : target[i];
            if (isExpApprox) {
                double expApprox = approx[i];
                double nonExpApprox = FastLogf(expApprox);
                if (hasDelta) {
                    expApprox *= approxDelta[i];
                    nonExpApprox += FastLogf(approxDelta[i]);
                }
                holder.Stats[0] += w * (IsFinite(expApprox) ? FastLogf(1 + expApprox) - prob * nonExpApprox : (1 - prob) * nonExpApprox);
            } else {
                double nonExpApprox = approx[i];
                if (hasDelta) {
                    nonExpApprox += approxDelta[i];
                }
                const double expApprox = exp(nonExpApprox);
                holder.Stats[0] += w * (IsFinite(expApprox) ? log(1 + expApprox) - prob * nonExpApprox : (1 - prob) * nonExpApprox);
            }
            holder.Stats[1] += w;
        }
        return holder;
    };
    return DispatchGenericLambda(impl, isExpApprox, !approxDeltaRef.empty(), !weight.empty(), LossFunction == ELossFunction::Logloss);
}

TVector<TParamSet> TCrossEntropyMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

void TCrossEntropyMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

/* CtrFactor */

namespace {
    class TCtrFactorMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TCtrFactorMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::CtrFactor, params) {
        }
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
    };
}

THolder<IMetric> MakeCtrFactorMetric(const TLossParams& params) {
    return MakeHolder<TCtrFactorMetric>(params);
}

TMetricHolder TCtrFactorMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);

    const auto& approxVec = approx.front();
    Y_ASSERT(approxVec.size() == target.size());

    TMetricHolder holder(2);
    const double* approxPtr = approxVec.data();
    const float* targetPtr = target.data();
    for (int i = begin; i < end; ++i) {
        float w = weight.empty() ? 1 : weight[i];
        const float targetVal = targetPtr[i] > TargetBorder;
        holder.Stats[0] += w * targetVal;

        const double p = OverflowSafeLogitProb(approxPtr[i]);
        holder.Stats[1] += w * p;
    }
    return holder;
}

void TCtrFactorMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::FixedValue;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TCtrFactorMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* SurvivalAFT */
namespace {
    struct TSurvivalAftMetric final: public TAdditiveMultiTargetMetric {
        explicit TSurvivalAftMetric(const TLossParams& params)
            : TAdditiveMultiTargetMetric(ELossFunction::SurvivalAft, params) {
            }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
    };
}

TVector<THolder<IMetric>> TSurvivalAftMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("scale");
    config.ValidParams->insert("dist");
    return AsVector(MakeHolder<TSurvivalAftMetric>(config.Params));
}

TVector<TParamSet> TSurvivalAftMetric::ValidParamSets() {
    // TODO(akhropov): SurvivalAft has 'scale' and 'dist' when it is used as an objective but no such params
    // when used as a metric, it's better to have objectives as separate entities
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
}

TMetricHolder TSurvivalAftMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end
) const {
    const auto evalImpl = [=](auto useWeights, auto hasDelta) {
        const auto realApprox = [=](int dim, int idx) { return fast_exp(approx[dim][idx] + (hasDelta ? approxDelta[dim][idx] : 0)); };
        const auto realTarget = [=](int dim, int idx) { return target[dim][idx] == -1 ? std::numeric_limits<float>::infinity() : target[dim][idx]; };
        const auto realWeight = [=](int idx) { return useWeights ? weight[idx] : 1; };

        TMetricHolder error(2);
        for (auto i : xrange(begin, end)) {
            if ((realApprox(0, i) <= realTarget(0, i)) || (realApprox(0, i) >= realTarget(1, i))) {
                double distanceFromInterval = Min(Abs(realApprox(0, i) - realTarget(0, i)), Abs(realApprox(0, i) - realTarget(1, i)));
                error.Stats[0] += distanceFromInterval * realWeight(i);
            }
            error.Stats[1] += realWeight(i);
        }
        return error;
    };

    return DispatchGenericLambda(evalImpl, !weight.empty(), !approxDelta.empty());
}

double TSurvivalAftMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] == 0 ? 0 : error.Stats[0] / error.Stats[1];
}

void TSurvivalAftMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

/* MultiRMSE */
namespace {
    struct TMultiRMSEMetric final: public TAdditiveMultiTargetMetric {
        explicit TMultiRMSEMetric(const TLossParams& params)
            : TAdditiveMultiTargetMetric(ELossFunction::MultiRMSE, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
    };
}

// static
TVector<THolder<IMetric>> TMultiRMSEMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMultiRMSEMetric>(config.Params));
}

TMetricHolder TMultiRMSEMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end
) const {
    const auto evalImpl = [=](auto useWeights, auto hasDelta) {
        const auto realApprox = [=](int dim, int idx) { return approx[dim][idx] + (hasDelta ? approxDelta[dim][idx] : 0); };
        const auto realWeight = [=](int idx) { return useWeights ? weight[idx] : 1; };

        TMetricHolder error(2);
        for (auto dim : xrange(target.size())) {
            for (auto i : xrange(begin, end)) {
                error.Stats[0] += realWeight(i) * Sqr(realApprox(dim, i) - target[dim][i]);
            }
        }
        for (auto i : xrange(begin, end)) {
            error.Stats[1] += realWeight(i);
        }

        return error;
    };

    return DispatchGenericLambda(evalImpl, !weight.empty(), !approxDelta.empty());
}

double TMultiRMSEMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] == 0 ? 0 : sqrt(error.Stats[0] / error.Stats[1]);
}

void TMultiRMSEMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMultiRMSEMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* MultiRMSEWithMissingValues */
namespace {
    struct TMultiRMSEWithMissingValues final: public TAdditiveMultiTargetMetric {
        explicit TMultiRMSEWithMissingValues(const TLossParams& params)
           : TAdditiveMultiTargetMetric(ELossFunction::MultiRMSEWithMissingValues, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
    };
}

// static
TVector<THolder<IMetric>> TMultiRMSEWithMissingValues::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMultiRMSEWithMissingValues>(config.Params));
}

TMetricHolder TMultiRMSEWithMissingValues::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end
) const {
    const auto evalImpl = [=](auto useWeights, auto hasDelta) {
        const auto realApprox = [=](int dim, int idx) { return approx[dim][idx] + (hasDelta ? approxDelta[dim][idx] : 0); };
        const auto realWeight = [=](int idx) { return useWeights ? weight[idx] : 1; };

        TMetricHolder error(target.size() * 2);
        for (auto dim : xrange(target.size())) {
            double sumWeights = 0.0;
            double sumErrors = 0.0;
            for (auto i : xrange(begin, end)) {
                if (!IsNan(target[dim][i])) {
                    sumErrors += realWeight(i) * Sqr(realApprox(dim, i) - target[dim][i]);
                    sumWeights += realWeight(i);
                }
            }
            error.Stats[dim * 2] += sumErrors;
            error.Stats[dim * 2 + 1] += sumWeights;
        }

        return error;
    };

    return DispatchGenericLambda(evalImpl, !weight.empty(), !approxDelta.empty());
}

double TMultiRMSEWithMissingValues::GetFinalError(const TMetricHolder& error) const {
    double finalError = 0.0;
    for (size_t dim = 0; dim < error.Stats.size(); dim += 2) {
        if (error.Stats[dim + 1] != 0) {
            finalError += error.Stats[dim] / error.Stats[dim+1];
        }
    }
    return sqrt(finalError);
}

void TMultiRMSEWithMissingValues::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMultiRMSEWithMissingValues::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* RMSEWithUncertainty */
namespace {
    class TRMSEWithUncertaintyMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TRMSEWithUncertaintyMetric(
            ELossFunction lossFunction,
            const TLossParams& descriptionParams);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;

    };
}

TRMSEWithUncertaintyMetric::TRMSEWithUncertaintyMetric(
    ELossFunction lossFunction,
    const TLossParams& descriptionParams)
    : TAdditiveSingleTargetMetric(lossFunction, descriptionParams)
{}

TVector<THolder<IMetric>> TRMSEWithUncertaintyMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TRMSEWithUncertaintyMetric>(config.Metric, config.Params));
}

TMetricHolder TRMSEWithUncertaintyMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weights,
    TConstArrayRef<TQueryInfo> /* queriesInfo */,
    int begin,
    int end
) const {
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(approx.size() == 2,
              "Approx dimension for RMSEWithUncertainty metric should be 2, found " << approx.size() <<
              ", probably your model was trained not with RMSEWithUncertainty loss function");
    const auto evalImpl = [=](auto useWeights, auto hasDelta) {
        const auto realApprox = [=](int dim, int idx) { return approx[dim][idx] + (hasDelta ? approxDelta[dim][idx] : 0); };
        const auto realWeight = [=](int idx) { return useWeights ? weights[idx] : 1; };

        TMetricHolder error(2);
        double stats0 = 0;
        double stats1 = 0;
        for (auto i : xrange(begin, end)) {
            double weight = realWeight(i);
            double expSum = -2 * realApprox(1, i);
            FastExpInplace(&expSum, /*count*/ 1);
            // np.log(2 * np.pi) / 2.0
            stats0 += weight * (0.9189385332046 + realApprox(1, i) + 0.5 * expSum * Sqr(realApprox(0, i) - target[i]));
            stats1 += weight;
        }
        error.Stats[0] += stats0;
        error.Stats[1] += stats1;
        return error;
    };

    return DispatchGenericLambda(evalImpl, !weights.empty(), !approxDelta.empty());
}

double TRMSEWithUncertaintyMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] == 0 ? 0 : error.Stats[0] / error.Stats[1];
}

void TRMSEWithUncertaintyMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TRMSEWithUncertaintyMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* RMSE */

namespace {
    struct TRMSEMetric final: public TAdditiveSingleTargetMetric {
        explicit TRMSEMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::RMSE, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TVector<THolder<IMetric>> TRMSEMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TRMSEMetric>(config.Params));
}

TMetricHolder TRMSEMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(!isExpApprox);

    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
}

double TRMSEMetric::GetFinalError(const TMetricHolder& error) const {
    return sqrt(error.Stats[0] / (error.Stats[1] + 1e-38));
}

void TRMSEMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TRMSEMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Log Cosh loss */

namespace {
    struct TLogCoshMetric final: public TAdditiveSingleTargetMetric {
        explicit TLogCoshMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::LogCosh, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TVector<THolder<IMetric>> TLogCoshMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TLogCoshMetric>(config.Params));
}

TMetricHolder TLogCoshMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(!isExpApprox);
    const double METRIC_APPROXIMATION_THRESHOLD = 12;

    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
        TMetricHolder error(2);
        for (int k : xrange(begin, end)) {
            double targetMismatch = approx[k] - target[k];
            if (hasDelta) {
                targetMismatch += approxDelta[k];
            }
            const float w = hasWeight ? weight[k] : 1;
            if (abs(targetMismatch) >= METRIC_APPROXIMATION_THRESHOLD)
                error.Stats[0] += (abs(targetMismatch) - FastLogf(2)) * w;
            else
                error.Stats[0] += FastLogf(cosh(targetMismatch)) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
}

double TLogCoshMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[0] / (error.Stats[1] + 1e-38);
}

void TLogCoshMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TLogCoshMetric::ValidParamSets() {
    return {
        TParamSet{
            {TParamInfo{"use_weights", false, true}},
            ""
        }
    };
};

/* Cox partial loss */

namespace {
    struct TCoxMetric final: public TNonAdditiveSingleTargetMetric {
        explicit TCoxMetric(const TLossParams& params)
            : TNonAdditiveSingleTargetMetric(ELossFunction::Cox, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

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
        ) const override;

        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TVector<THolder<IMetric>> TCoxMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TCoxMetric>(config.Params));
}

TVector<TParamSet> TCoxMetric::ValidParamSets() {
    return {TParamSet{{}, ""}};
};

TMetricHolder TCoxMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> targets,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int /*begin*/,
    int /*end*/,
    NPar::ILocalExecutor& /* executor */
) const {
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(2);
    error.Stats[1] = 1;

    const auto ndata = targets.ysize();
    TVector<int> labelOrder(ndata);
    std::iota(labelOrder.begin(), labelOrder.end(), 0);
    std::sort(
        labelOrder.begin(),
        labelOrder.end(),
        [=] (int lhs, int rhs) {
            return std::abs(targets[lhs]) < std::abs(targets[rhs]);
        }
    );

    const auto approxRef = approx[0];
    const auto approxDeltaRef = GetRowRef(approxDelta, /*row idx*/ 0);
    const auto getApprox = [=] (int i) {
        return approxRef[i] + (approxDelta.empty() ? 0 : approxDeltaRef[i]);
    };

    double expPSum = 0;
    for (auto i = 0; i < ndata; ++i) {
        expPSum += std::exp(getApprox(i));
    }

    double lastExpP = 0.0;
    double accumulatedSum = 0;
    for (auto i : xrange(ndata)) {
        const int ind = labelOrder[i];

        const double y = targets[ind];

        const double p = getApprox(ind);
        const double expP = std::exp(p);

        accumulatedSum += lastExpP;

        if (y > 0) {
            expPSum -= accumulatedSum;
            accumulatedSum = 0;
            error.Stats[0] += p - std::log(std::max(expPSum, 1e-20));
        }

        lastExpP = expP;
    }
    error.Stats[0] = error.Stats[0];

    return error;
}

double TCoxMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[0];
}

void TCoxMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

/* Lq */

namespace {
    struct TLqMetric final: public TAdditiveSingleTargetMetric {
        explicit TLqMetric(double q, const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::Lq, params)
            , Q(q) {
            CB_ENSURE(Q >= 1, "Lq metric is defined for q >= 1, got " << q);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const double Q;
    };
}

TVector<TParamSet> TLqMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"q", true, {}}
            },
            ""
        }
    };
}

// static
TVector<THolder<IMetric>> TLqMetric::Create(const TMetricConfig& config) {
    CB_ENSURE(config.GetParamsMap().contains("q"), "Metric " << ELossFunction::Lq << " requires q as parameter");
    config.ValidParams->insert("q");
    return AsVector(MakeHolder<TLqMetric>(FromString<float>(config.GetParamsMap().at("q")),
                                          config.Params));
}

TMetricHolder TLqMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approxRef,
        TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
}

void TLqMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

/* Quantile */

namespace {
    class TQuantileMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TQuantileMetric(ELossFunction lossFunction,
                                 const TLossParams& params, double alpha, double delta);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static constexpr double MaeAlpha = 0.5;
        static constexpr double MaeDelta = 1e-6;
        ELossFunction LossFunction;
        double Alpha;
        double Delta;
    };
}

// static.
TVector<THolder<IMetric>> TQuantileMetric::Create(const TMetricConfig& config) {
    switch (config.Metric) {
        case ELossFunction::MAE:
            return AsVector(MakeHolder<TQuantileMetric>(config.Metric, config.Params, MaeAlpha, MaeDelta));
            break;
        case ELossFunction::Quantile: {
            double alpha = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "alpha", 0.5);
            double delta = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "delta", 1e-6);

            config.ValidParams->insert("alpha");
            config.ValidParams->insert("delta");
            return AsVector(MakeHolder<TQuantileMetric>(config.Metric, config.Params, alpha, delta));
            break;
        }
        default:
            // Unreachable.
            CB_ENSURE(false, "Unreachable.");
    }
}

TQuantileMetric::TQuantileMetric(ELossFunction lossFunction, const TLossParams& params, double alpha, double delta)
        : TAdditiveSingleTargetMetric(lossFunction, params)
        , LossFunction(lossFunction)
        , Alpha(alpha)
        , Delta(delta)
{
    CB_ENSURE(
            Delta >= 0 && Delta <= 1e-2,
            "Parameter delta for quantile metric should be in interval [0, 0.01]"
    );
    CB_ENSURE_INTERNAL(
        lossFunction == ELossFunction::Quantile || lossFunction == ELossFunction::MAE,
        "lossFunction " << lossFunction
    );
    CB_ENSURE(lossFunction == ELossFunction::Quantile || alpha == 0.5, "Alpha parameter should not be used for MAE loss");
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6, "Alpha parameter for quantile metric should be in interval [0, 1]");
}

TMetricHolder TQuantileMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight, auto isMAE) {
        double alpha = Alpha;
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
        TMetricHolder error(2);
        for (int i : xrange(begin, end)) {
            double val = target[i] - approx[i];
            if (hasDelta) {
                val -= approxDelta[i];
            }
            const double multiplier = (abs(val) < Delta) ? 0 : ((val > 0) ? alpha : -(1 - alpha));
            if (val < -Delta) {
                val += Delta;
            } else if (val > Delta) {
                val -= Delta;
            }

            const float w = hasWeight ? weight[i] : 1;
            error.Stats[0] += (multiplier * val) * w;
            error.Stats[1] += w;
        }
        if (isMAE) {
            error.Stats[0] *= 2;
        }
        return error;
    };
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty(), LossFunction == ELossFunction::MAE);
}

TString TQuantileMetric::GetDescription() const {
    if (LossFunction == ELossFunction::Quantile) {
        if (Delta == 1e-6) {
            const TMetricParam<double> alpha("alpha", Alpha, /*userDefined*/true);
            return BuildDescription(LossFunction, UseWeights, "%.3g", alpha);
        }
        const TMetricParam<double> alpha("alpha", Alpha, /*userDefined*/true);
        const TMetricParam<double> delta("delta", Delta, /*userDefined*/true);
        return BuildDescription(LossFunction, UseWeights, "%.3g", alpha, "%g", delta);
    } else {
        return BuildDescription(LossFunction, UseWeights);
    }
}

void TQuantileMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TQuantileMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"alpha", false, MaeAlpha},
                TParamInfo{"delta", false, MaeDelta}
            },
            ""
        }
    };
}

/* Expectile */
namespace {
    class TExpectileMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TExpectileMetric(const TLossParams& params, double alpha);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const double Alpha;
        static constexpr double DefaultAlpha = 0.5;
    };
}

// static.
TVector<THolder<IMetric>> TExpectileMetric::Create(const TMetricConfig& config) {
    auto it = config.GetParamsMap().find("alpha");
    config.ValidParams->insert("alpha");
    return AsVector(MakeHolder<TExpectileMetric>(
        config.Params,
        it != config.GetParamsMap().end() ? FromString<float>(it->second) : DefaultAlpha));
}

TExpectileMetric::TExpectileMetric(const TLossParams& params, double alpha)
        : TAdditiveSingleTargetMetric(ELossFunction::Expectile, params)
        , Alpha(alpha)
{
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6, "Alpha parameter for expectile metric should be in interval [0, 1]");
}

TMetricHolder TExpectileMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        double alpha = Alpha;
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
        TMetricHolder error(2);
        for (int i : xrange(begin, end)) {
            double val = target[i] - approx[i];
            if (hasDelta) {
                val -= approxDelta[i];
            }
            const double multiplier = (val > 0) ? alpha: (1 - alpha);
            const float w = hasWeight ? weight[i] : 1;
            error.Stats[0] += multiplier * Sqr(val) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
}

void TExpectileMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TExpectileMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"alpha", false, DefaultAlpha}
            },
            ""
        }
    };
};

/* LogLinQuantile */

namespace {
    class TLogLinQuantileMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TLogLinQuantileMetric(const TLossParams& params, double alpha);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const double Alpha;
        static constexpr double DefaultAlpha = 0.5;
    };
}

// static.
TVector<THolder<IMetric>> TLogLinQuantileMetric::Create(const TMetricConfig& config) {
    auto it = config.GetParamsMap().find("alpha");
    config.ValidParams->insert("alpha");
    return AsVector(MakeHolder<TLogLinQuantileMetric>(config.Params,
        it != config.GetParamsMap().end() ? FromString<float>(it->second) : DefaultAlpha));
}

TLogLinQuantileMetric::TLogLinQuantileMetric(const TLossParams& params, double alpha)
        : TAdditiveSingleTargetMetric(ELossFunction::LogLinQuantile, params)
        , Alpha(alpha)
{
    CB_ENSURE(Alpha > -1e-6 && Alpha < 1.0 + 1e-6, "Alpha parameter for log-linear quantile metric should be in interval (0, 1)");
}

TMetricHolder TLogLinQuantileMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight) {
        double alpha = Alpha;
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
    return DispatchGenericLambda(impl, isExpApprox, !approxDeltaRef.empty(), !weight.empty());
}

void TLogLinQuantileMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TLogLinQuantileMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"alpha", false, DefaultAlpha}
            },
            ""
        }
    };
}

/* MAPE */

namespace {
    struct TMAPEMetric final : public TAdditiveSingleTargetMetric {
        explicit TMAPEMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::MAPE, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

// static.
TVector<THolder<IMetric>> TMAPEMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMAPEMetric>(config.Params));
}

TMetricHolder TMAPEMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
        TMetricHolder error(2);
        for (int k : xrange(begin, end)) {
            const float w = hasWeight ? weight[k] : 1;
            const double delta = hasDelta ? approxDelta[k] : 0;
            error.Stats[0] += Abs(target[k] - (approx[k] + delta)) / Max(1.f, Abs(target[k])) * w;
            error.Stats[1] += w;
        }
        return error;
    };
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
}

void TMAPEMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMAPEMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Greater K */

namespace {
    struct TNumErrorsMetric final: public TAdditiveSingleTargetMetric {
        explicit TNumErrorsMetric(const TLossParams& params, double greaterThen)
            : TAdditiveSingleTargetMetric(ELossFunction::NumErrors, params)
            , GreaterThan(greaterThen) {
            CB_ENSURE(greaterThen > 0, "Error: NumErrors metric requires num_erros > 0 parameter, got " << greaterThen);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const double GreaterThan;
    };
}

TVector<THolder<IMetric>> TNumErrorsMetric::Create(const TMetricConfig& config) {
    CB_ENSURE(config.GetParamsMap().contains("greater_than"), "Metric " << ELossFunction::NumErrors << " requires greater_than as parameter");
    config.ValidParams->insert("greater_than");
    return AsVector(MakeHolder<TNumErrorsMetric>(config.Params,
                                                 FromString<double>(config.GetParamsMap().at("greater_than"))));
}

TMetricHolder TNumErrorsMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
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

void TNumErrorsMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TNumErrorsMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"greater_than", true, {}}
            },
            ""
        }
    };
};

/* Poisson */

namespace {
    struct TPoissonMetric final: public TAdditiveSingleTargetMetric {
        explicit TPoissonMetric(const TLossParams& params)
        : TAdditiveSingleTargetMetric(ELossFunction::Poisson, params) {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

// static
TVector<THolder<IMetric>> TPoissonMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TPoissonMetric>(config.Params));
}

TMetricHolder TPoissonMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
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

    CB_ENSURE(approxRef.size() == 1, "Metric Poisson supports only single-dimensional data");
    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
    return DispatchGenericLambda(impl, isExpApprox, !approxDeltaRef.empty(), !weight.empty());
}

void TPoissonMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TPoissonMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Tweedie */

namespace {
    struct TTweedieMetric final: public TAdditiveSingleTargetMetric {
        explicit TTweedieMetric(const TLossParams& params, double variance_power)
            : TAdditiveSingleTargetMetric(ELossFunction::Tweedie, params)
            , VariancePower(variance_power) {
            CB_ENSURE(VariancePower > 1 && VariancePower < 2, "Tweedie metric is defined for 1 < variance_power < 2, got " << variance_power);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const double VariancePower;
    };
}

// static
TVector<THolder<IMetric>> TTweedieMetric::Create(const TMetricConfig& config) {
    CB_ENSURE(config.GetParamsMap().contains("variance_power"), "Metric " << ELossFunction::Tweedie << " requires variance_power as parameter");
    config.ValidParams->insert("variance_power");
    return AsVector(MakeHolder<TTweedieMetric>(config.Params,
                                               FromString<float>(config.GetParamsMap().at("variance_power"))));
}

TMetricHolder TTweedieMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approxRef,
        TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
        TMetricHolder error(2);
        for (int k : xrange(begin, end)) {
            double curApprox = approx[k];
            if (hasDelta) {
                curApprox += approxDelta[k];
            }
            const float w = hasWeight ? weight[k] : 1;
            double margin = -target[k] * std::exp((1 - VariancePower) * curApprox) / (1 - VariancePower);
            margin += std::exp((2 - VariancePower) * curApprox) / (2 - VariancePower);
            error.Stats[0] += w * margin;
            error.Stats[1] += w;
        }
        return error;
    };
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
}

void TTweedieMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TTweedieMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"variance_power", true, {}}
            },
            ""
        }
    };
}

/* Focal loss */

namespace {
    struct TFocalMetric final: public TAdditiveSingleTargetMetric {
        explicit TFocalMetric(const TLossParams& params, double focal_alpha, double focal_gamma)
            : TAdditiveSingleTargetMetric(ELossFunction::Focal, params)
            , FocalAlpha(focal_alpha), FocalGamma(focal_gamma) {
            CB_ENSURE(FocalAlpha > 0 && FocalAlpha < 1, "Focal metric is defined for 0 < focal_alpha < 1, got " << focal_alpha);
            CB_ENSURE(FocalGamma > 0, "Focal metric is defined for 0 < focal_gamma, got " << focal_gamma);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                const TConstArrayRef<TConstArrayRef<double>> approx,
                const TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const double FocalAlpha;
        const double FocalGamma;
    };
}

// static
TVector<THolder<IMetric>> TFocalMetric::Create(const TMetricConfig& config) {
    CB_ENSURE(config.GetParamsMap().contains("focal_alpha"), "Metric " << ELossFunction::Focal << " requires focal_alpha as parameter");
    CB_ENSURE(config.GetParamsMap().contains("focal_gamma"), "Metric " << ELossFunction::Focal << " requires focal_gamma as parameter");
    config.ValidParams->insert("focal_alpha");
    config.ValidParams->insert("focal_gamma");
    return AsVector(MakeHolder<TFocalMetric>(config.Params,
                                               FromString<float>(config.GetParamsMap().at("focal_alpha")),
                                               FromString<float>(config.GetParamsMap().at("focal_gamma"))
                                            )
                    );
}

TMetricHolder TFocalMetric::EvalSingleThread(
        const TConstArrayRef<TConstArrayRef<double>> approxRef,
        const TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
        TMetricHolder error(2);
        for (int k : xrange(begin, end)) {
            double curApprox = approx[k];
            if (hasDelta) {
                curApprox += approxDelta[k];
            }
            curApprox = 1 / (1 + exp(-curApprox));
            const float w = hasWeight ? weight[k] : 1;
            double at = target[k] == 1 ? FocalAlpha : 1 - FocalAlpha;
            double p = std::clamp(curApprox, 0.0000000000001, 0.9999999999999);
            double pt = target[k] == 1 ? p : 1 - p;
            double margin =  -at * pow((1 - pt), FocalGamma) * log(pt);
            error.Stats[0] += w * margin;
            error.Stats[1] += w;
            }
            return error;
    };
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
}

void TFocalMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TFocalMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"focal_alpha", true, {}},
                TParamInfo{"focal_gamma", true, {}}
            },
            ""
        }
    };
}

/* Mean squared logarithmic error */

namespace {
    struct TMSLEMetric final: public TAdditiveSingleTargetMetric {
        explicit TMSLEMetric(const TLossParams& params)
        : TAdditiveSingleTargetMetric(ELossFunction::MSLE, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

// static.
TVector<THolder<IMetric>> TMSLEMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMSLEMetric>(config.Params));
}

TMetricHolder TMSLEMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
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

void TMSLEMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMSLEMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Median absolute error */

namespace {
    struct TMedianAbsoluteErrorMetric final: public TNonAdditiveSingleTargetMetric {
        explicit TMedianAbsoluteErrorMetric(const TLossParams& params)
            : TNonAdditiveSingleTargetMetric(ELossFunction::MedianAbsoluteError, params) {
            UseWeights.MakeIgnored();
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

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
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

// static.
TVector<THolder<IMetric>> TMedianAbsoluteErrorMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMedianAbsoluteErrorMetric>(config.Params));
}

TMetricHolder TMedianAbsoluteErrorMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::ILocalExecutor& /* executor */
) const {
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

void TMedianAbsoluteErrorMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMedianAbsoluteErrorMetric::ValidParamSets() {
    return {TParamSet{{}, ""}};
};

/* Symmetric mean absolute percentage error */

namespace {
    struct TSMAPEMetric final: public TAdditiveSingleTargetMetric {
        explicit TSMAPEMetric(const TLossParams& params)
        : TAdditiveSingleTargetMetric(ELossFunction::SMAPE, params)
        {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

// static.
TVector<THolder<IMetric>> TSMAPEMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TSMAPEMetric>(config.Params));
}

TMetricHolder TSMAPEMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
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

void TSMAPEMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TSMAPEMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* loglikelihood of prediction */

namespace {
    struct TLLPMetric final: public TAdditiveSingleTargetMetric {
        explicit TLLPMetric(const TLossParams& params)
        : TAdditiveSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, params)
        {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        TVector<TString> GetStatDescriptions() const override;
    };
}

// static.
TVector<THolder<IMetric>> TLLPMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TLLPMetric>(config.Params));
}

TMetricHolder TLLPMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

void TLLPMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TString> TLLPMetric::GetStatDescriptions() const {
    return {"intermediate result", "clicks", "shows"};
}

TVector<TParamSet> TLLPMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* MultiClass */

namespace {
    struct TMultiClassMetric final: public TAdditiveSingleTargetMetric {
        explicit TMultiClassMetric(const TLossParams& params)
        : TAdditiveSingleTargetMetric(ELossFunction::MultiClass, params)
        {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TVector<THolder<IMetric>> TMultiClassMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMultiClassMetric>(config.Params));
}

static void GetMultiDimensionalApprox(int idx, TConstArrayRef<TConstArrayRef<double>> approx, TConstArrayRef<TConstArrayRef<double>> approxDelta, TArrayRef<double> evaluatedApprox) {
    const auto approxDimension = approx.size();
    CB_ENSURE(
        approxDimension == evaluatedApprox.size(),
        "evaluatedApprox size " << evaluatedApprox.size() << " != " << approxDimension
    );
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
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

    constexpr int UnrollMaxCount = 16;
    TVector<TVector<double>> evaluatedApprox(UnrollMaxCount, TVector<double>(approxDimension));
    for (int idx = begin; idx < end; idx += UnrollMaxCount) {
        const int unrollCount = Min(UnrollMaxCount, end - idx);
        SumTransposedBlocks(idx, idx + unrollCount, MakeArrayRef(approx), MakeArrayRef(approxDelta), MakeArrayRef(evaluatedApprox));
        for (int unrollIdx : xrange(unrollCount)) {
            auto approxRef = MakeArrayRef(evaluatedApprox[unrollIdx]);
            const double maxApprox = *MaxElement(approxRef.begin(), approxRef.end());
            for (auto& approxValue : approxRef) {
                approxValue -= maxApprox;
            }

            const int targetClass = static_cast<int>(target[idx + unrollIdx]);
            CB_ENSURE_INTERNAL(
                targetClass >= 0 && targetClass < approxDimension,
                "Inappropriate targetClass " << targetClass
            );
            const double targetClassApprox = approxRef[targetClass];

            FastExpInplace(approxRef.data(), approxRef.size());
            const double sumExpApprox = Accumulate(approxRef, /*val*/0.0);

            const float w = weight.empty() ? 1 : weight[idx + unrollIdx];
            error.Stats[0] -= (targetClassApprox - log(sumExpApprox)) * w;
            error.Stats[1] += w;
        }
    }

    return error;
}

void TMultiClassMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMultiClassMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* MultiClassOneVsAll */

namespace {
    struct TMultiClassOneVsAllMetric final: public TAdditiveSingleTargetMetric {
        explicit TMultiClassOneVsAllMetric(const TLossParams& params)
        : TAdditiveSingleTargetMetric(ELossFunction::MultiClassOneVsAll, params)
        {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

// static
TVector<THolder<IMetric>> TMultiClassOneVsAllMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMultiClassOneVsAllMetric>(config.Params));
}

TMetricHolder TMultiClassOneVsAllMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
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
        CB_ENSURE_INTERNAL(
            targetClass >= 0 && targetClass < approxDimension,
            "Inappropriate targetClass " << targetClass
        );
        sumDimErrors += evaluatedApprox[targetClass];

        const float w = weight.empty() ? 1 : weight[k];
        error.Stats[0] -= sumDimErrors / approxDimension * w;
        error.Stats[1] += w;
    }
    return error;
}

void TMultiClassOneVsAllMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMultiClassOneVsAllMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};


/*MultiQuantile*/

namespace {
    class TMultiQuantileMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TMultiQuantileMetric(const TLossParams& params, const TVector<double>& alpha, double delta);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const TVector<double> Alpha;
        double Delta;
    };
}

// static.
TVector<THolder<IMetric>> TMultiQuantileMetric::Create(const TMetricConfig& config) {
    const auto& lossParams = config.GetParamsMap();
    auto alpha = NCatboostOptions::GetAlphaMultiQuantile(lossParams);
    double delta = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "delta", 1e-6);

    config.ValidParams->insert("alpha");
    config.ValidParams->insert("delta");
    return AsVector(MakeHolder<TMultiQuantileMetric>(config.Params, alpha, delta));
}

TMultiQuantileMetric::TMultiQuantileMetric(const TLossParams& params, const TVector<double>& alpha, double delta)
        : TAdditiveSingleTargetMetric(ELossFunction::MultiQuantile, params)
        , Alpha(alpha)
        , Delta(delta)
{
    CB_ENSURE(
        Delta >= 0 && Delta <= 1e-2,
        "Parameter delta for quantile metric should be in interval [0, 0.01]"
    );
    CB_ENSURE(AllOf(Alpha, [] (double a) { return a > -1e-6 && a < 1.0 + 1e-6; }), "Parameter alpha for quantile metric should be in interval [0, 1]");
}

TMetricHolder TMultiQuantileMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == Alpha.size(), "Metric MultiQuantile expects same number of predictions and quantiles");
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TMetricHolder error(2);
        for (auto j : xrange(approx.size())) {
            const auto alpha = Alpha[j];
            for (int i : xrange(begin, end)) {
                double val = target[i] - approx[j][i];
                if (hasDelta) {
                    val -= approxDelta[j][i];
                }
                const double multiplier = (abs(val) < Delta) ? 0 : ((val > 0) ? alpha : -(1 - alpha));
                if (val < -Delta) {
                    val += Delta;
                } else if (val > Delta) {
                    val -= Delta;
                }

                const float w = hasWeight ? weight[i] : 1;
                error.Stats[0] += (multiplier * val) * w;
                error.Stats[1] += w;
            }
        }
        return error;
    };
    return DispatchGenericLambda(impl, !approxDelta.empty(), !weight.empty());
}

TString TMultiQuantileMetric::GetDescription() const {
    const TMetricParam<TVector<double>> alpha("alpha", Alpha, /*userDefined*/true);
    if (Delta == 1e-6) {
        return BuildDescription(ELossFunction::MultiQuantile, UseWeights, "%.3g", alpha);
    }
    const TMetricParam<double> delta("delta", Delta, /*userDefined*/true);
    return BuildDescription(ELossFunction::MultiQuantile, UseWeights, "%.3g", alpha, "%g", delta);
}

void TMultiQuantileMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMultiQuantileMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"alpha", false, 0.5},
                TParamInfo{"delta", false, 1e-6}
            },
            ""
        }
    };
}


/* PairLogit */

namespace {
    struct TPairLogitMetric final: public TAdditiveSingleTargetMetric {
        explicit TPairLogitMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::PairLogit, params) {
            UseWeights.SetDefaultValue(true);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TVector<THolder<IMetric>> TPairLogitMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("max_pairs");
    return AsVector(MakeHolder<TPairLogitMetric>(config.Params));
}

TMetricHolder TPairLogitMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> /*target*/,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    CB_ENSURE(approxRef.size() == 1, "Metric PairLogit supports only single-dimensional data");

    const auto impl = [=] (auto isExpApprox, auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
                for (double& approxVal : approxExpShifted) {
                    approxVal -= maxQueryApprox;
                }
                FastExpInplace(approxExpShifted.data(), querySize);
                for (double& approxVal : approxExpShifted) {
                    approxVal += 1e-38;
                }
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
    return DispatchGenericLambda(impl, isExpApprox, !approxDeltaRef.empty(), UseWeights.Get());
}

EErrorType TPairLogitMetric::GetErrorType() const {
    return EErrorType::PairwiseError;
}

void TPairLogitMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TPairLogitMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"max_pairs", false, {}}
            },
            ""
        }
    };
}

/* QueryRMSE */

namespace {
    struct TQueryRMSEMetric final: public TAdditiveSingleTargetMetric {
        explicit TQueryRMSEMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::QueryRMSE, params) {
            UseWeights.SetDefaultValue(true);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
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

TVector<THolder<IMetric>> TQueryRMSEMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TQueryRMSEMetric>(config.Params));
}

TMetricHolder TQueryRMSEMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weight.empty());
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

double TQueryRMSEMetric::GetFinalError(const TMetricHolder& error) const {
    return sqrt(error.Stats[0] / (error.Stats[1] + 1e-38));
}

void TQueryRMSEMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TQueryRMSEMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* PFound */

namespace {
    struct TPFoundMetric final: public TAdditiveSingleTargetMetric {
        explicit TPFoundMetric(const TLossParams& params,
                int topSize, double decay);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const int TopSize;
        const double Decay;
        static constexpr int DefaultTopSize = -1;
        static constexpr double DefaultDecay = 0.85;
    };
}

// static
TVector<THolder<IMetric>> TPFoundMetric::Create(const TMetricConfig& config) {
    auto itTopSize = config.GetParamsMap().find("top");
    auto itDecay = config.GetParamsMap().find("decay");
    const int topSize = itTopSize != config.GetParamsMap().end() ? FromString<int>(itTopSize->second) : DefaultTopSize;
    const double decay = itDecay != config.GetParamsMap().end() ? FromString<double>(itDecay->second) : DefaultDecay;
    config.ValidParams->insert("top");
    config.ValidParams->insert("decay");
    return AsVector(MakeHolder<TPFoundMetric>(config.Params, topSize, decay));
}

TPFoundMetric::TPFoundMetric(const TLossParams& params, int topSize, double decay)
        : TAdditiveSingleTargetMetric(ELossFunction::PFound, params)
        , TopSize(topSize)
        , Decay(decay) {
    UseWeights.SetDefaultValue(true);
}

TMetricHolder TPFoundMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approxRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    const auto impl = [=] (auto hasDelta, auto isExpApprox) {
        TConstArrayRef<double> approx = approxRef[0];
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), isExpApprox);
}

EErrorType TPFoundMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TPFoundMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TPFoundMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TPFoundMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"decay", false, DefaultDecay},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            ""
        }
    };
};

/* NDCG@N */

namespace {
    struct TDcgMetric final: public TAdditiveSingleTargetMetric {
        explicit TDcgMetric(ELossFunction lossFunction, const TLossParams& params,
                            int topSize, ENdcgMetricType type, bool normalized, ENdcgDenominatorType denominator);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        TString GetDescription () const override;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const int TopSize;
        const ENdcgMetricType MetricType;
        const bool Normalized;
        const ENdcgDenominatorType DenominatorType;

        static constexpr int DefaultTopSize = -1;
        static constexpr ENdcgMetricType DefaultMetricType = ENdcgMetricType::Base;
        static constexpr ENdcgDenominatorType DefaultDenominatorType = ENdcgDenominatorType::LogPosition;
        static constexpr size_t LargeGroupSize = 10 * 1000;
    };
}

TVector<TParamSet> TDcgMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"top", false, DefaultTopSize},
                TParamInfo{"type", false, ToString(DefaultMetricType)},
                TParamInfo{"denominator", false, ToString(DefaultDenominatorType)},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            ""
        }
    };
};

// static
TVector<THolder<IMetric>> TDcgMetric::Create(const TMetricConfig& config) {
    auto itTopSize = config.GetParamsMap().find("top");
    auto itType = config.GetParamsMap().find("type");
    auto itDenominator = config.GetParamsMap().find("denominator");
    int topSize = itTopSize != config.GetParamsMap().end() ? FromString<int>(itTopSize->second) : DefaultTopSize;

    ENdcgMetricType type = DefaultMetricType;

    if (itType != config.GetParamsMap().end()) {
        type = FromString<ENdcgMetricType>(itType->second);
    }

    ENdcgDenominatorType denominator = DefaultDenominatorType;

    if (itDenominator != config.GetParamsMap().end()) {
        denominator = FromString<ENdcgDenominatorType>(itDenominator->second);
    }
    config.ValidParams->insert("top");
    config.ValidParams->insert("type");
    config.ValidParams->insert("denominator");

    return AsVector(MakeHolder<TDcgMetric>(config.Metric, config.Params, topSize, type,
                                           config.Metric == ELossFunction::NDCG, denominator));
}

TString TDcgMetric::GetDescription() const {
    const TMetricParam<int> topSize("top", TopSize, TopSize != DefaultTopSize);
    const TMetricParam<ENdcgMetricType> type("type", MetricType, true);
    return BuildDescription(Normalized ? ELossFunction::NDCG : ELossFunction::DCG, UseWeights, topSize, type);
}

TDcgMetric::TDcgMetric(ELossFunction lossFunction, const TLossParams& params,
                       int topSize, ENdcgMetricType type, bool normalized, ENdcgDenominatorType denominator)
    : TAdditiveSingleTargetMetric(lossFunction, params)
    , TopSize(topSize)
    , MetricType(type)
    , Normalized(normalized)
    , DenominatorType(denominator) {
    UseWeights.SetDefaultValue(true);
}

TMetricHolder TDcgMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    Y_ASSERT(!isExpApprox);
    TConstArrayRef<double> approxesRef;
    TVector<double> approxWithDelta;
    if (approxDelta.empty()) {
        approxesRef = approx.front();
    } else {
        Y_ASSERT(approx.front().size() == approxDelta.front().size());
        approxWithDelta.yresize(approx.front().size());
        for (size_t doc = 0; doc < approx.front().size(); ++doc) {
            approxWithDelta[doc] = approx.front()[doc] + approxDelta.front()[doc];
        }
        approxesRef = approxWithDelta;
    }

    TMetricHolder error(2);
    TVector<NMetrics::TSample> samples;
    TVector<double> decay;
    decay.yresize(LargeGroupSize);
    FillDcgDecay(DenominatorType, Nothing(), decay);
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        const auto queryBegin = queriesInfo[queryIndex].Begin;
        const auto queryEnd = queriesInfo[queryIndex].End;
        const auto querySize = queryEnd - queryBegin;
        const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.f;
        NMetrics::TSample::FromVectors(
            MakeArrayRef(target.data() + queryBegin, querySize),
            MakeArrayRef(approxesRef.data() + queryBegin, querySize),
            &samples);
        if (decay.size() < querySize) {
            decay.resize(2 * querySize);
            FillDcgDecay(DenominatorType, Nothing(), decay);
        }
        if (Normalized) {
            error.Stats[0] += queryWeight * CalcNdcg(samples, decay, MetricType, TopSize);
        } else {
            error.Stats[0] += queryWeight * CalcDcg(samples, decay, MetricType, TopSize);
        }
        error.Stats[1] += queryWeight;
    }
    return error;
}

EErrorType TDcgMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TDcgMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TDcgMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = std::numeric_limits<double>::infinity();
    }
}

/* QuerySoftMax */

namespace {
    struct TQuerySoftMaxMetric final: public TAdditiveSingleTargetMetric {
        explicit TQuerySoftMaxMetric(const TLossParams& params)
          : TAdditiveSingleTargetMetric(ELossFunction::QuerySoftMax, params)
          , Beta(NCatboostOptions::GetQuerySoftMaxBeta(params.GetParamsMap()))
        {
            UseWeights.SetDefaultValue(true);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        TMetricHolder EvalSingleQuery(
            int start,
            int count,
            TConstArrayRef<double> approxes,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> targets,
            TConstArrayRef<float> weights,
            TArrayRef<double> softmax
        ) const;
        double Beta;

        static constexpr double DefaultBeta = 1.0;
        static constexpr double DefaultLambda = 0.01;
    };
}

// static
TVector<THolder<IMetric>> TQuerySoftMaxMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("lambda");
    config.ValidParams->insert("beta");
    return AsVector(MakeHolder<TQuerySoftMaxMetric>(config.Params));
}

TMetricHolder TQuerySoftMaxMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
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
    TConstArrayRef<double> approxesRef,
    TConstArrayRef<TConstArrayRef<double>> approxDeltaRef,
    bool isExpApprox,
    TConstArrayRef<float> targets,
    TConstArrayRef<float> weights,
    TArrayRef<double> softmax
) const {
    Y_ASSERT(!isExpApprox);
    const auto impl = [=] (auto hasDelta, auto hasWeight) {
        TConstArrayRef<double> approx = approxesRef;
        TConstArrayRef<double> approxDelta = GetRowRef(approxDeltaRef, /*rowIdx*/0);
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
            softmax[dim] = Beta * (approx[start + dim] + delta);
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
    return DispatchGenericLambda(impl, !approxDeltaRef.empty(), !weights.empty());
}

void TQuerySoftMaxMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TQuerySoftMaxMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"beta", false, DefaultBeta},
                TParamInfo{"lambda", false, DefaultLambda}
            },
            ""
        }
    };
};

/* R2 */

namespace {
    struct TR2TargetSumMetric final: public TAdditiveSingleTargetMetric {

        explicit TR2TargetSumMetric()
            : TAdditiveSingleTargetMetric(ELossFunction::R2, TLossParams()) {
            UseWeights.SetDefaultValue(true);
        }
        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override {
            return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
        }
        TString GetDescription() const override { CB_ENSURE(false, "helper class should not be used as metric"); }
        void GetBestValue(EMetricBestValue* /*valueType*/, float* /*bestValue*/) const override { CB_ENSURE(false, "helper class should not be used as metric"); }
    };

    struct TR2ImplMetric final: public TAdditiveSingleTargetMetric {
        explicit TR2ImplMetric(double targetMean)
            : TAdditiveSingleTargetMetric(ELossFunction::R2, TLossParams())
            , TargetMean(targetMean) {
            UseWeights.SetDefaultValue(true);
        }
        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        double GetFinalError(const TMetricHolder& /*error*/) const override { CB_ENSURE(false, "helper class should not be used as metric"); }
        TString GetDescription() const override { CB_ENSURE(false, "helper class should not be used as metric"); }
        void GetBestValue(EMetricBestValue* /*valueType*/, float* /*bestValue*/) const override { CB_ENSURE(false, "helper class should not be used as metric"); }

    private:
        const double TargetMean = 0.0;
    };

    struct TR2Metric final: public TNonAdditiveSingleTargetMetric {
        explicit TR2Metric(const TLossParams& params)
            : TNonAdditiveSingleTargetMetric(ELossFunction::R2, params)
        {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

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
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TMetricHolder TR2TargetSumMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> /*approx*/,
    TConstArrayRef<TConstArrayRef<double>> /*approxDelta*/,
    bool /*isExpApprox*/,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    const auto realWeight = [&](int i) { return weight.empty() ? 1.0 : weight[i]; };

    TMetricHolder error(2);
    for (auto i : xrange(begin, end)) {
        error.Stats[0] += target[i] * realWeight(i);
        error.Stats[1] += realWeight(i);
    }
    return error;
}

TMetricHolder TR2ImplMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool /*isExpApprox*/,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    const auto realApprox = [&](int i) { return approx.front()[i] + (approxDelta.empty() ? 0.0 : approxDelta.front()[i]); };
    const auto realWeight = [&](int i) { return weight.empty() ? 1.0 : weight[i]; };

    TMetricHolder error(2);
    for (auto i : xrange(begin, end)) {
        error.Stats[0] += Sqr(realApprox(i) - target[i]) * realWeight(i);
        error.Stats[1] += Sqr(target[i] - TargetMean) * realWeight(i);
    }
    return error;
}

TVector<THolder<IMetric>> TR2Metric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TR2Metric>(config.Params));
}

TMetricHolder TR2Metric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::ILocalExecutor& executor
) const {
    Y_ASSERT(!isExpApprox);

    auto targetMeanCalcer = TR2TargetSumMetric();
    auto targetMean = targetMeanCalcer.GetFinalError(
        targetMeanCalcer.Eval(
            /*approx*/{}, /*approxDelta*/{}, /*isExpApprox*/false,
            target, UseWeights ? weight : TVector<float>(),
            /*queriesInfo*/{},
            begin, end, executor
        )
    );

    return TR2ImplMetric(targetMean).Eval(
        approx, approxDelta,
        /*isExpApprox*/false,
        target, UseWeights ? weight : TVector<float>(),
        /*queriesInfo*/{},
        begin, end, executor
    );
}

double TR2Metric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? 1 - error.Stats[0] / error.Stats[1] : 1;
}

void TR2Metric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TR2Metric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* AUC */

template <typename T>
static TVector<TVector<T>> ConstructSquareMatrix(const TString& matrixString) {
    const TVector<TString> matrixVector = StringSplitter(matrixString).Split('/');
    ui32 size = 0;
    while (size * size < matrixVector.size()) {
        size++;
    }
    CB_ENSURE(size * size == matrixVector.size(), "Size of Matrix should be a square of integer.");
    TVector<TVector<T>> result(size);
    for (ui32 i = 0; i < size; ++i) {
        result[i].resize(size);
        for (ui32 j = 0; j < size; ++j) {
            CB_ENSURE(TryFromString<T>(matrixVector[i * size + j], result[i][j]), "Error while parsing AUC Mu missclassification matrix. Building matrix with size "
                    << size << ", cannot parse \"" << matrixVector[i * size + j] << "\" as a float.");
        }
    }
    return result;
}

namespace {
    struct TAUCMetric final: public TNonAdditiveSingleTargetMetric {
        explicit TAUCMetric(const TLossParams& params, EAucType singleClassType)
            : TNonAdditiveSingleTargetMetric(ELossFunction::AUC, params)
            , Type(singleClassType) {
            UseWeights.SetDefaultValue(false);
        }

        explicit TAUCMetric(const TLossParams& params, int positiveClass)
            : TNonAdditiveSingleTargetMetric(ELossFunction::AUC, params)
            , PositiveClass(positiveClass)
            , Type(EAucType::OneVsAll) {
            UseWeights.SetDefaultValue(false);
        }

        explicit TAUCMetric(const TLossParams& params, const TMaybe<TVector<TVector<double>>>& misclassCostMatrix = Nothing())
            : TNonAdditiveSingleTargetMetric(ELossFunction::AUC, params)
            , Type(EAucType::Mu)
            , MisclassCostMatrix(misclassCostMatrix) {
            UseWeights.SetDefaultValue(false);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::ILocalExecutor& executor) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        int PositiveClass = 1;
        EAucType Type;
        TMaybe<TVector<TVector<double>>> MisclassCostMatrix = Nothing();
    };
}

TVector<THolder<IMetric>> TAUCMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("type");
    EAucType aucType = config.ApproxDimension == 1 ? EAucType::Classic : EAucType::Mu;
    if (config.GetParamsMap().contains("type")) {
        const TString name = config.GetParamsMap().at("type");
        aucType = FromString<EAucType>(name);
        if (config.ApproxDimension == 1) {
            CB_ENSURE(aucType == EAucType::Classic || aucType == EAucType::Ranking,
                      "AUC type \"" << aucType << "\" isn't a singleclass AUC type");
        } else {
            CB_ENSURE(aucType == EAucType::Mu || aucType == EAucType::OneVsAll,
                      "AUC type \"" << aucType << "\" isn't a multiclass AUC type");
        }
    }
    switch (aucType) {
        case EAucType::Classic: {
            return AsVector(MakeHolder<TAUCMetric>(config.Params, EAucType::Classic));
            break;
        }
        case EAucType::Ranking: {
            return AsVector(MakeHolder<TAUCMetric>(config.Params, EAucType::Ranking));
            break;
        }
        case EAucType::Mu: {
            config.ValidParams->insert("misclass_cost_matrix");
            TMaybe<TVector<TVector<double>>> misclassCostMatrix = Nothing();
            if (config.GetParamsMap().contains("misclass_cost_matrix")) {
                misclassCostMatrix.ConstructInPlace(ConstructSquareMatrix<double>(
                        config.GetParamsMap().at("misclass_cost_matrix")));
            }
            if (misclassCostMatrix) {
                for (ui32 i = 0; i < misclassCostMatrix->size(); ++i) {
                    CB_ENSURE((*misclassCostMatrix)[i][i] == 0, "Diagonal elements of the misclass cost matrix should be equal to 0.");
                }
            }
            return AsVector(MakeHolder<TAUCMetric>(config.Params, misclassCostMatrix));
            break;
        }
        case EAucType::OneVsAll: {
            TVector<THolder<IMetric>> metrics;
            for (int i = 0; i < config.ApproxDimension; ++i) {
                metrics.push_back(MakeHolder<TAUCMetric>(config.Params, i));
            }
            return metrics;
            break;
        }
        default: {
            CB_ENSURE(false, "Unexpected AUC type");
        }
    }
}

TVector<TParamSet> TAUCMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, false},
                TParamInfo{"type", false, ToString(EAucType::Classic)},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            ""
        },
        TParamSet{
            {
                TParamInfo{"use_weights", false, false},
                TParamInfo{"type", false, ToString(EAucType::Mu)},
                TParamInfo{"misclass_cost_matrix", false, {}},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            "Multiclass"
        }
    };
};


THolder<IMetric> MakeMultiClassAucMetric(const TLossParams& params, int positiveClass) {
    return MakeHolder<TAUCMetric>(params, positiveClass);
}

TMetricHolder TAUCMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::ILocalExecutor& executor
) const {
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(
        (approx.size() > 1) == (Type == EAucType::Mu || Type == EAucType::OneVsAll),
        "Not single dimension approxes are supported only for AUC::Mu and AUC::OneVsAll"
    );
    CB_ENSURE_INTERNAL(
        approx.front().size() == target.size(),
        "Inconsistent approx and target dimension"
    );
    if (Type == EAucType::Mu && MisclassCostMatrix) {
        CB_ENSURE(MisclassCostMatrix->size() == approx.size(), "Number of classes should be equal to the size of the misclass cost matrix.");
    }

    TVector<TVector<double>> currentApprox;
    if (Type == EAucType::Mu || Type == EAucType::OneVsAll) {
        ResizeRank2(approx.size(), approx[0].size(), currentApprox);
        AssignRank2(MakeArrayRef(approx), &currentApprox);
        if (!approxDelta.empty()) {
            for (ui32 i = 0; i < approx.size(); ++i) {
                for (ui32 j = 0; j < approx[i].size(); ++j) {
                    currentApprox[i][j] += approxDelta[i][j];
                }
            }
        }
    }
    if (Type == EAucType::Mu) {
        TMetricHolder error(2);
        error.Stats[0] = CalcMuAuc(currentApprox, target, UseWeights ? weight : TConstArrayRef<float>(), &executor, MisclassCostMatrix);
        error.Stats[1] = 1;
        return error;
    }

    TVector<double> probability;
    if (Type == EAucType::OneVsAll) {
        probability = CalcSoftmax(currentApprox, &executor)[PositiveClass];

    }
    const auto realApprox = [&](int idx) {
        return Type == EAucType::OneVsAll ? probability[idx] : approx[0][idx] + (approxDelta.empty() ? 0.0 : approxDelta[0][idx]);
    };
    const auto realWeight = [&](int idx) {
        return UseWeights && !weight.empty() ? weight[idx] : 1.0;
    };
    const auto realTarget = [&](int idx) {
        return Type == EAucType::OneVsAll ? target[idx] == static_cast<double>(PositiveClass) : target[idx];
    };

    TMetricHolder error(2);
    error.Stats[1] = 1.0;

    if (Type == EAucType::Ranking) {
        TVector<NMetrics::TSample> samples;
        samples.reserve(end - begin);
        for (int i : xrange(begin, end)) {
            samples.emplace_back(realTarget(i), realApprox(i), realWeight(i));
        }
        error.Stats[0] = CalcAUC(&samples, nullptr, nullptr, &executor);
    } else {
        TVector<NMetrics::TBinClassSample> positiveSamples, negativeSamples;
        for (int i : xrange(begin, end)) {
            const auto currentTarget = realTarget(i);
            CB_ENSURE(0 <= currentTarget && currentTarget <= 1, "All target values should be in the segment [0, 1], for Ranking AUC please use type=Ranking.");
            if (currentTarget > 0) {
                positiveSamples.emplace_back(realApprox(i), currentTarget * realWeight(i));
            }
            if (currentTarget < 1) {
                negativeSamples.emplace_back(realApprox(i), (1 - currentTarget) * realWeight(i));
            }
        }
        error.Stats[0] = CalcBinClassAuc(&positiveSamples, &negativeSamples, &executor);
    }

    return error;
}

template<typename T>
static TString ConstructDescriptionOfSquareMatrix(const TVector<TVector<T>>& matrix) {
    TString matrixInString = "";
    for (ui32 i = 0; i < matrix.size(); ++i) {
        for (ui32 j = 0; j < matrix.size(); ++j) {
            matrixInString += ToString(matrix[i][j]);
            if (i + 1 != matrix.size() || j + 1 != matrix.size()) {
                matrixInString += "/";
            }
        }
    }
    return matrixInString;
}

TString TAUCMetric::GetDescription() const {
    switch (Type) {
        case EAucType::OneVsAll: {
            const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
            return BuildDescription(ELossFunction::AUC, UseWeights, positiveClass);
        }
        case EAucType::Mu: {
            TMetricParam<TString> aucType("type", ToString(EAucType::Mu), /*userDefined*/true);
            if (MisclassCostMatrix) {
                TMetricParam<TString> misclassCostMatrix("misclass_cost_matrix", ConstructDescriptionOfSquareMatrix<double>(*MisclassCostMatrix), /*UserDefined*/true);
                return BuildDescription(ELossFunction::AUC, UseWeights, aucType, misclassCostMatrix);
            }
            return BuildDescription(ELossFunction::AUC, UseWeights, aucType);
        }
        case EAucType::Classic: {
            return BuildDescription(ELossFunction::AUC, UseWeights);
        }
        case EAucType::Ranking: {
            return BuildDescription(ELossFunction::AUC, UseWeights, TMetricParam<TString>("type", ToString(EAucType::Ranking), /*userDefined*/true));
        }
        default: {
            CB_ENSURE(false, "Unexpected AUC type");
        }
    }
}

void TAUCMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

/* Normalized Gini metric */

namespace {
    struct TNormalizedGini final: public TNonAdditiveSingleTargetMetric {
        explicit TNormalizedGini(const TLossParams& params)
            : TNonAdditiveSingleTargetMetric(ELossFunction::NormalizedGini, params)
            , IsMultiClass(false) {
        }
        explicit TNormalizedGini(const TLossParams& params, int positiveClass)
            : TNonAdditiveSingleTargetMetric(ELossFunction::NormalizedGini, params)
            , PositiveClass(positiveClass)
            , IsMultiClass(true) {
        }
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::ILocalExecutor& executor) const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int PositiveClass = 1;
        const bool IsMultiClass = false;
    };
}

TVector<THolder<IMetric>> TNormalizedGini::Create(const TMetricConfig& config) {
    if (config.ApproxDimension == 1) {
        return AsVector(MakeHolder<TNormalizedGini>(config.Params));
    }
    TVector<THolder<IMetric>> result;
    for (int i : xrange(config.ApproxDimension)) {
        result.push_back(MakeHolder<TNormalizedGini>(config.Params, i));
    }
    return result;
}

TMetricHolder TNormalizedGini::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::ILocalExecutor& executor
) const {
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(
            (approx.size() > 1) == IsMultiClass,
            "Not single dimension approxes are supported only for Multiclass"
    );
    CB_ENSURE_INTERNAL(
        approx.front().size() == target.size(),
        "Inconsistent approx and target dimension"
    );

    const auto realApprox = [&](int idx) {
        return approx[IsMultiClass ? PositiveClass : 0][idx]
        + (approxDelta.empty() ? 0.0 : approxDelta[IsMultiClass ? PositiveClass : 0][idx]);
    };
    const auto realWeight = [&](int idx) {
        return UseWeights && !weight.empty() ? weight[idx] : 1.0;
    };
    const auto realTarget = [&](int idx) {
        return IsMultiClass ? target[idx] == static_cast<double>(PositiveClass) : target[idx] > TargetBorder;
    };

    TVector<NMetrics::TSample> samples;
    for (auto i : xrange(begin, end)) {
        samples.emplace_back(realTarget(i), realApprox(i), realWeight(i));
    }

    TMetricHolder error(2);
    error.Stats[0] = 2.0 * CalcAUC(&samples, nullptr, nullptr, &executor) - 1.0;
    error.Stats[1] = 1.0;
    return error;
}

TString TNormalizedGini::GetDescription() const {
    if (IsMultiClass) {
        const TMetricParam<int> positiveClass("class", PositiveClass, /*userDefined*/true);
        return BuildDescription(ELossFunction::NormalizedGini, UseWeights, positiveClass);
    } else {
        return BuildDescription(ELossFunction::NormalizedGini, UseWeights, "%.3g", MakeTargetBorderParam(TargetBorder));
    }
}

void TNormalizedGini::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TNormalizedGini::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"border", false, TargetBorder}
            },
            ""
        },
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"border", false, NJson::JSON_NULL}
            },
            "Multiclass"
        }
    };
};

/* Fair Loss metric */

namespace {
    struct TFairLossMetric final: public TAdditiveSingleTargetMetric {
        static constexpr double DefaultSmoothness = 1.0;

        explicit TFairLossMetric(const TLossParams& params, double smoothness)
            : TAdditiveSingleTargetMetric(ELossFunction::FairLoss, params)
            , Smoothness(smoothness) {
            CB_ENSURE(smoothness > 0.0, "Fair loss is not defined for negative smoothness");
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    private:
        const double Smoothness;
    };
}

// static.
TVector<THolder<IMetric>> TFairLossMetric::Create(const TMetricConfig& config) {
    double smoothness = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "smoothness", DefaultSmoothness);
    config.ValidParams->insert("smoothness");
    return AsVector(MakeHolder<TFairLossMetric>(config.Params, smoothness));
}

TMetricHolder TFairLossMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    CB_ENSURE(approx.size() == 1, "Fair Loss metric supports only single-dimentional data");
    Y_ASSERT(approx.front().size() == target.size());
    Y_ASSERT(!isExpApprox);
    const auto realApprox = [&](int idx) { return approx[0][idx] + (approxDelta.empty() ? 0.0 : approxDelta[0][idx]);  };
    const auto realWeight = [&](int idx) { return weight.empty() ? 1.0 : weight[idx]; };
    TMetricHolder error(2);
    for (int i : xrange(begin, end)) {
        double smoothMismatch = Abs(realApprox(i) - target[i])/Smoothness;
        error.Stats[0] += Power(Smoothness, 2)*(smoothMismatch - Log2(smoothMismatch + 1)/M_LN2_INV)*realWeight(i);
        error.Stats[1] += realWeight(i);
    }
    return error;
}

void TFairLossMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TFairLossMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"smoothness", false, DefaultSmoothness}
            },
            ""
        }
    };
};

/* Balanced Accuracy */

namespace {
    struct TBalancedAccuracyMetric final: public TAdditiveSingleTargetMetric {
        explicit TBalancedAccuracyMetric(const TLossParams& params,
                                         double predictionBorder)
                : TAdditiveSingleTargetMetric(ELossFunction::BalancedAccuracy, params)
                , PredictionBorder(predictionBorder)
        {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int PositiveClass = 1;
        const double PredictionBorder = GetDefaultPredictionBorder();
    };
}

TVector<THolder<IMetric>> TBalancedAccuracyMetric::Create(const TMetricConfig& config) {
    CB_ENSURE(config.ApproxDimension == 1,
              "Balanced accuracy is used only for binary classification problems.");
    config.ValidParams->insert("border");
    return AsVector(MakeHolder<TBalancedAccuracyMetric>(config.Params,
                                                        config.GetPredictionBorderOrDefault()));
}

TMetricHolder TBalancedAccuracyMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return CalcBalancedAccuracyMetric(approx, target, weight, begin, end, PositiveClass, TargetBorder, PredictionBorder);
}

void TBalancedAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

double TBalancedAccuracyMetric::GetFinalError(const TMetricHolder& error) const {
    return CalcBalancedAccuracyMetric(error);
}

TVector<TParamSet> TBalancedAccuracyMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"border", false, TargetBorder}
            },
            ""
        }
    };
}

/* Balanced Error Rate */

namespace {
    struct TBalancedErrorRate final: public TAdditiveSingleTargetMetric {
        explicit TBalancedErrorRate(const TLossParams& params,
                                    double predictionBorder)
                : TAdditiveSingleTargetMetric(ELossFunction::BalancedErrorRate, params)
                , PredictionBorder(predictionBorder)
        {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
        const int PositiveClass = 1;
        const double PredictionBorder = GetDefaultPredictionBorder();
    };
}

// static.
TVector<THolder<IMetric>> TBalancedErrorRate::Create(const TMetricConfig& config) {
    CB_ENSURE(config.ApproxDimension == 1, "Balanced Error Rate is used only for binary classification problems.");
    config.ValidParams->insert("border");
    return AsVector(MakeHolder<TBalancedErrorRate>(config.Params,
                                                   config.GetPredictionBorderOrDefault()));
}

TMetricHolder TBalancedErrorRate::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return CalcBalancedAccuracyMetric(approx, target, weight, begin, end, PositiveClass, TargetBorder, PredictionBorder);
}

void TBalancedErrorRate::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

double TBalancedErrorRate::GetFinalError(const TMetricHolder& error) const {
    return 1 - CalcBalancedAccuracyMetric(error);
}

TVector<TParamSet> TBalancedErrorRate::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"border", false, TargetBorder}
            },
            ""
        }
    };
}

/* Brier Score */

namespace {
    struct TBrierScoreMetric final: public TAdditiveSingleTargetMetric {
        explicit TBrierScoreMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::BrierScore, params) {
        }
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
    };
}

THolder<IMetric> MakeBrierScoreMetric(const TLossParams& params) {
    return MakeHolder<TBrierScoreMetric>(params);
}

TMetricHolder TBrierScoreMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

void TBrierScoreMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

double TBrierScoreMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
}

TVector<TParamSet> TBrierScoreMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* Hinge loss */

namespace {
    struct THingeLossMetric final: public TAdditiveSingleTargetMetric {
        explicit THingeLossMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::HingeLoss, params)
            {}

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;

    private:
        static constexpr double TargetBorder = GetDefaultTargetBorder();
    };
}

TVector<THolder<IMetric>> THingeLossMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("border");
    return AsVector(MakeHolder<THingeLossMetric>(config.Params));
}

TMetricHolder THingeLossMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    return ComputeHingeLossMetric(approx, target, weight, begin, end, TargetBorder);
}

void THingeLossMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

double THingeLossMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
}

TVector<TParamSet> THingeLossMetric::ValidParamSets() {
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

/* PairAccuracy */

namespace {
    struct TPairAccuracyMetric final: public TAdditiveSingleTargetMetric {
        explicit TPairAccuracyMetric(const TLossParams& params)
            : TAdditiveSingleTargetMetric(ELossFunction::PairAccuracy, params) {
            UseWeights.SetDefaultValue(true);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TVector<THolder<IMetric>> TPairAccuracyMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TPairAccuracyMetric>(config.Params));
}

TMetricHolder TPairAccuracyMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

void TPairAccuracyMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TPairAccuracyMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

/* PrecisionAtK */

namespace {
    struct TPrecisionAtKMetric final: public TAdditiveSingleTargetMetric {
        explicit TPrecisionAtKMetric(const TLossParams& params,
                                     int topSize,
                                     float targetBorder);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    private:
        static constexpr int DefaultTopSize = -1;
        const int TopSize;
        const float TargetBorder;
    };
}

// static.
TVector<THolder<IMetric>> TPrecisionAtKMetric::Create(const TMetricConfig& config) {
    const int topSize = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "top", DefaultTopSize);
    const float targetBorder = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "border", GetDefaultTargetBorder());
    config.ValidParams->insert("top");
    config.ValidParams->insert("border");
    return AsVector(MakeHolder<TPrecisionAtKMetric>(config.Params, topSize, targetBorder));
}

TPrecisionAtKMetric::TPrecisionAtKMetric(const TLossParams& params, int topSize, float targetBorder)
        : TAdditiveSingleTargetMetric(ELossFunction::PrecisionAt, params)
        , TopSize(topSize)
        , TargetBorder(targetBorder) {
    UseWeights.SetDefaultValue(true);
}

TMetricHolder TPrecisionAtKMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

        error.Stats[0] += CalcPrecisionAtK(approxCopy, targetCopy, TopSize, TargetBorder);
        error.Stats[1]++;
    }
    return error;
}

EErrorType TPrecisionAtKMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TPrecisionAtKMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1;
}

void TPrecisionAtKMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TPrecisionAtKMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"top", false, DefaultTopSize},
                TParamInfo{"border", false, GetDefaultTargetBorder()}
            },
            ""
        }
    };
};

/* RecallAtK */

namespace {
    struct TRecallAtKMetric final: public TAdditiveSingleTargetMetric {
        explicit TRecallAtKMetric(const TLossParams& params, int topSize, float targetBorder);
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static constexpr int DefaultTopSize = -1;
        const int TopSize;
        const float TargetBorder;
    };
}

TVector<THolder<IMetric>> TRecallAtKMetric::Create(const TMetricConfig& config) {
    const int topSize = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "top", DefaultTopSize);
    const float targetBorder = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "border", GetDefaultTargetBorder());
    config.ValidParams->insert("top");
    config.ValidParams->insert("border");
    return AsVector(MakeHolder<TRecallAtKMetric>(config.Params, topSize, targetBorder));
}

TRecallAtKMetric::TRecallAtKMetric(const TLossParams& params, int topSize, float targetBorder)
        : TAdditiveSingleTargetMetric(ELossFunction::RecallAt, params)
        , TopSize(topSize)
        , TargetBorder(targetBorder) {
    UseWeights.SetDefaultValue(true);
}

TMetricHolder TRecallAtKMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

        error.Stats[0] += CalcRecallAtK(approxCopy, targetCopy, TopSize, TargetBorder);
        error.Stats[1]++;
    }
    return error;
}

EErrorType TRecallAtKMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TRecallAtKMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 1;
}

void TRecallAtKMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TRecallAtKMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"top", false, DefaultTopSize},
                TParamInfo{"border", false, GetDefaultTargetBorder()}
            },
            ""
        }
    };
};

/* Mean Average Precision at k */

namespace {
    struct TMAPKMetric final: public TAdditiveSingleTargetMetric {
        explicit TMAPKMetric(const TLossParams& params, int topSize, float targetBorder);
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        double GetFinalError(const TMetricHolder& error) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static constexpr int DefaultTopSize = -1;
        const int TopSize;
        const float TargetBorder;
    };
}

// static.
TVector<THolder<IMetric>> TMAPKMetric::Create(const TMetricConfig& config) {
    const int topSize = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "top", DefaultTopSize);
    const float targetBorder = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "border", GetDefaultTargetBorder());
    config.ValidParams->insert("top");
    config.ValidParams->insert("border");
    return AsVector(MakeHolder<TMAPKMetric>(config.Params, topSize, targetBorder));
}

TMAPKMetric::TMAPKMetric(const TLossParams& params, int topSize, float targetBorder)
        : TAdditiveSingleTargetMetric(ELossFunction::MAP, params)
        , TopSize(topSize)
        , TargetBorder(targetBorder) {
    UseWeights.SetDefaultValue(true);
}

TMetricHolder TMAPKMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

        error.Stats[0] += CalcAveragePrecisionK(approxCopy, targetCopy, TopSize, TargetBorder);
        error.Stats[1]++;
    }
    return error;
}

EErrorType TMAPKMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

double TMAPKMetric::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[1] != 0 ? error.Stats[0] / error.Stats[1] : 0;
}

void TMAPKMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TMAPKMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"top", false, DefaultTopSize},
                TParamInfo{"border", false, GetDefaultTargetBorder()}
            },
            ""
        }
    };
};

/* Precision-Recall AUC */

namespace {
    struct TPRAUCMetric : public TNonAdditiveSingleTargetMetric {
        explicit TPRAUCMetric(const TLossParams& params, int positiveClass)
            : TNonAdditiveSingleTargetMetric(ELossFunction::PRAUC, params),
             PositiveClass(positiveClass), IsMultiClass(true) {
            UseWeights.SetDefaultValue(false);
        }

        explicit TPRAUCMetric(const TLossParams& params)
            : TNonAdditiveSingleTargetMetric(ELossFunction::PRAUC, params) {
            UseWeights.SetDefaultValue(false);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder Eval(
                const TVector<TVector<double>>& approx,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end,
                NPar::ILocalExecutor& executor) const override {
                    return Eval(To2DConstArrayRef<double>(approx), /*approxDelta*/{}, /*isExpApprox*/false, target, weight, queriesInfo, begin, end, executor);
                }
        TMetricHolder Eval(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end,
                NPar::ILocalExecutor& executor) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

        private:
            int PositiveClass = 1;
            bool IsMultiClass = false;
    };
}

THolder<IMetric> MakeBinClassPRAUCMetric(const TLossParams& params) {
    return MakeHolder<TPRAUCMetric>(params);
}

THolder<IMetric> MakeMultiClassPRAUCMetric(const TLossParams& params, int positiveClass) {
    return MakeHolder<TPRAUCMetric>(params, positiveClass);
}

TVector<THolder<IMetric>> TPRAUCMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("type");
    EAucType aucType = config.ApproxDimension == 1 ? EAucType::Classic : EAucType::OneVsAll;
    if (config.Params.GetParamsMap().contains("type")) {
        const TString name = config.Params.GetParamsMap().at("type");
        aucType = FromString<EAucType>(name);
        if (config.ApproxDimension == 1) {
            CB_ENSURE(aucType == EAucType::Classic,
                      "PRAUC type \"" << aucType << "\" isn't a singleclass PRAUC type");
        } else {
            CB_ENSURE(aucType == EAucType::OneVsAll,
                      "PRAUC type \"" << aucType << "\" isn't a multiclass PRAUC type");
        }
    }
    switch (aucType) {
        case EAucType::Classic: {
            return AsVector(MakeHolder<TPRAUCMetric>(config.Params));
        }
        case EAucType::OneVsAll: {
            TVector<THolder<IMetric>> metrics;
            for (int i = 0; i < config.ApproxDimension; ++i) {
                metrics.push_back(MakeHolder<TPRAUCMetric>(config.Params, i));
            }
            return metrics;
        }
        default: {
            CB_ENSURE(false, "Unexpected AUC type");
        }
    }
}

TMetricHolder TPRAUCMetric::Eval(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int begin,
    int end,
    NPar::ILocalExecutor& /*executor*/
) const {
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(
        (approx.size() > 1) == IsMultiClass,
        "Not single dimension approxes are supported only for Multiclass"
    );
    CB_ENSURE_INTERNAL(approx[0].size() == target.size(), "Inconsistent approx and target size");

    TMetricHolder error(2);
    error.Stats[1] = 1;

    int elemCount = end - begin;

    struct Sample {
        double approx;
        float target;
        float weight;
    };

    TVector<Sample> sortedSamples;
    sortedSamples.reserve(elemCount);

    const auto realApprox = [&](int idx) {
        return approx[IsMultiClass ? PositiveClass : 0][idx]
        + (approxDelta.empty() ? 0.0 : approxDelta[IsMultiClass ? PositiveClass : 0][idx]);
    };
    const auto realWeight = [&](int idx) {
        return UseWeights && !weight.empty() ? weight[idx] : 1.0f;
    };

    for (int i = begin; i < end; ++i) {
        sortedSamples.push_back({realApprox(i), target[i], realWeight(i)});
    }

    std::sort(sortedSamples.begin(), sortedSamples.end(), [](const Sample& l, const Sample& r) {return l.approx < r.approx;});

    int curNegCount = 0;
    double curTp = 0;
    double curFp = 0;
    double curFn = 0;

    for (size_t i = 0; i < target.size(); ++i) {
        if (sortedSamples[i].target == PositiveClass) {
            curTp += sortedSamples[i].weight;
        } else {
            curFp += sortedSamples[i].weight;
        }
    }

    CB_ENSURE(curTp > 0, "No element of a positive class");

    double prevPrecision = 0;
    double prevRecall = 1;
    double& auc = error.Stats[0];

    auto isApproxesEqual = [] (double approxL, double approxR) {
        return Abs(approxL - approxR) < 1e-8;
    };

    while (curNegCount <= elemCount) {
        double precision = (curTp == 0 && curFp == 0) ? 1 : curTp / (curTp + curFp + 0.0);
        double recall = curTp / (curTp + curFn + 0.0);

        auc += (prevRecall - recall) * (prevPrecision + precision) / 2;

        prevRecall = recall;
        prevPrecision = precision;

        auto moveOneBorder = [&]() {
            if (curNegCount < elemCount) {
                auto curWeight = sortedSamples[curNegCount].weight;
                if (sortedSamples[curNegCount].target == PositiveClass) {
                    curTp -= curWeight;
                    curFn += curWeight;
                } else {
                    curFp -= curWeight;
                }
            }
            ++curNegCount;
        };

        moveOneBorder();
        while (curNegCount < elemCount && isApproxesEqual(sortedSamples[curNegCount - 1].approx, sortedSamples[curNegCount].approx)) {
            moveOneBorder();
        }
    }

    return error;
}

void TPRAUCMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TPRAUCMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, false},
                TParamInfo{"type", false, ToString(EAucType::Classic)},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            ""
        },
        TParamSet{
            {
                TParamInfo{"use_weights", false, false},
                TParamInfo{"type", false, ToString(EAucType::OneVsAll)},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            "Multiclass"
        }
    };
};

/* Custom */

namespace {
    class TCustomMetric: public TSingleTargetMetric {
    public:
        explicit TCustomMetric(const TCustomMetricDescriptor& descriptor);
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> /* queriesInfo */,
            int begin,
            int end,
            NPar::ILocalExecutor& /* executor */
        ) const override {
            CB_ENSURE_INTERNAL(!isExpApprox, "Custom metrics do not support exponentiated approxes");
            TVector<TConstArrayRef<double>> approxRef;
            approxRef.assign(approx.begin(), approx.end());
            TVector<TVector<double>> updatedApprox;
            if (!approxDelta.empty()) {
                const auto approxDim = approx.size();
                ResizeRank2(approxDim, target.size(), updatedApprox); // allocate full approx, fill only [begin, end)
                for (auto i : xrange(approxDim)) {
                    for (auto j : xrange(begin, end)) {
                        updatedApprox[i][j] = approx[i][j] + approxDelta[i][j];
                    }
                }
                approxRef = To2DConstArrayRef<double>(updatedApprox);
            }
            approx = MakeArrayRef(approxRef);
            TMetricHolder result = (*(Descriptor.EvalFunc))(approx, target, UseWeights ? weight : TConstArrayRef<float>{}, begin, end, Descriptor.CustomData);
            CB_ENSURE(
                result.Stats.ysize() == 2,
                "Custom metric evaluate() returned incorrect value."\
                " Expected tuple of size 2, got tuple of size " << result.Stats.ysize() << "."
            );
            return result;
        }
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
        bool IsAdditiveMetric() const final;
        // be conservative by default
        bool NeedTarget() const override {
            return true;
        }
    private:
        TCustomMetricDescriptor Descriptor;
    };
}

TCustomMetric::TCustomMetric(const TCustomMetricDescriptor& descriptor)
    : TSingleTargetMetric(ELossFunction::PythonUserDefinedPerObject, TLossParams())
    , Descriptor(descriptor)
{
    UseWeights.SetDefaultValue(true);
}

TString TCustomMetric::GetDescription() const {
    TString description = Descriptor.GetDescriptionFunc(Descriptor.CustomData);
    return BuildDescription(description, UseWeights);
}

void TCustomMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    bool isMaxOptimal = Descriptor.IsMaxOptimalFunc(Descriptor.CustomData);
    *valueType = isMaxOptimal ? EMetricBestValue::Max : EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = isMaxOptimal ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }
}

double TCustomMetric::GetFinalError(const TMetricHolder& error) const {
    return Descriptor.GetFinalErrorFunc(error, Descriptor.CustomData);
}

bool TCustomMetric::IsAdditiveMetric() const {
    return Descriptor.IsAdditiveFunc(Descriptor.CustomData);
}

/* CustomMultiTarget */

namespace {
    class TMultiTargetCustomMetric: public TMultiTargetMetric {
    public:
        explicit TMultiTargetCustomMetric(const TCustomMetricDescriptor& descriptor);

        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end,
            NPar::ILocalExecutor& /* executor */
        ) const override {
            CB_ENSURE_INTERNAL(approxDelta.empty(), "Custom metrics do not support approx deltas and exponentiated approxes");
            TMetricHolder result = (*(Descriptor.EvalMultiTargetFunc))(approx, target, UseWeights ? weight : TConstArrayRef<float>{}, begin, end, Descriptor.CustomData);
            CB_ENSURE(
                result.Stats.ysize() == 2,
                "Custom metric evaluate() returned incorrect value."\
                " Expected tuple of size 2, got tuple of size " << result.Stats.ysize() << "."
            );
            return result;
        }

        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
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
    };
}

TMultiTargetCustomMetric::TMultiTargetCustomMetric(const TCustomMetricDescriptor& descriptor)
    : TMultiTargetMetric(ELossFunction::PythonUserDefinedMultiTarget, TLossParams())
    , Descriptor(descriptor)
{
    UseWeights.SetDefaultValue(true);
}

TString TMultiTargetCustomMetric::GetDescription() const {
    TString description = Descriptor.GetDescriptionFunc(Descriptor.CustomData);
    return BuildDescription(description, UseWeights);
}

void TMultiTargetCustomMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    bool isMaxOptimal = Descriptor.IsMaxOptimalFunc(Descriptor.CustomData);
    *valueType = isMaxOptimal ? EMetricBestValue::Max : EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = isMaxOptimal ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }
}

double TMultiTargetCustomMetric::GetFinalError(const TMetricHolder& error) const {
    return Descriptor.GetFinalErrorFunc(error, Descriptor.CustomData);
}


THolder<IMetric> MakeCustomMetric(const TCustomMetricDescriptor& descriptor) {
    if (descriptor.IsMultiTargetMetric()) {
        return MakeHolder<TMultiTargetCustomMetric>(descriptor);
    } else {
        return MakeHolder<TCustomMetric>(descriptor);
    }
}

/* UserDefinedPerObjectMetric */

namespace {
    class TUserDefinedPerObjectMetric : public TMetric, ISingleTargetEval {
    public:
        explicit TUserDefinedPerObjectMetric(const TLossParams& params);
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int begin,
            int end,
            NPar::ILocalExecutor& executor
        ) const override;
        TMetricHolder Eval(
            TConstArrayRef<TConstArrayRef<double>> /*approx*/,
            TConstArrayRef<TConstArrayRef<double>> /*approxDelta*/,
            bool /*isExpApprox*/,
            TConstArrayRef<float> /*target*/,
            TConstArrayRef<float> /*weight*/,
            TConstArrayRef<TQueryInfo> /*queriesInfo*/,
            int /*begin*/,
            int /*end*/,
            NPar::ILocalExecutor& /*executor*/
        ) const override {
            CB_ENSURE(
                false,
                "User-defined per object metrics do not support approx deltas and exponentiated approxes");
            return TMetricHolder();
        }
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        bool IsAdditiveMetric() const final {
            return true;
        }

    private:
        const double Alpha;
        static constexpr double DefaultAlpha = 0.0;
    };
}

// static.
TVector<THolder<IMetric>> TUserDefinedPerObjectMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("alpha");
    return AsVector(MakeHolder<TUserDefinedPerObjectMetric>(config.Params));
}

TUserDefinedPerObjectMetric::TUserDefinedPerObjectMetric(const TLossParams& params)
        : TMetric(ELossFunction::UserPerObjMetric, params)
        , Alpha(params.GetParamsMap().contains("alpha") ? FromString<float>(params.GetParamsMap().at("alpha")) : DefaultAlpha) {
    UseWeights.MakeIgnored();
}

TVector<TParamSet> TUserDefinedPerObjectMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"alpha", false, DefaultAlpha}
            },
            ""
        }
    };
};

TMetricHolder TUserDefinedPerObjectMetric::Eval(
    const TVector<TVector<double>>& /*approx*/,
    TConstArrayRef<float> /*target*/,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int /*begin*/,
    int /*end*/,
    NPar::ILocalExecutor& /*executor*/
) const {
    CB_ENSURE(false, "Not implemented for TUserDefinedPerObjectMetric metric.");
    TMetricHolder metric(2);
    return metric;
}

void TUserDefinedPerObjectMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

/* UserDefinedQuerywiseMetric */

namespace {
    class TUserDefinedQuerywiseMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TUserDefinedQuerywiseMetric(const TLossParams& params);
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const double Alpha;
        static constexpr double DefaultAlpha = 0.0;
    };
}

// static.
TVector<THolder<IMetric>> TUserDefinedQuerywiseMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("alpha");
    return AsVector(MakeHolder<TUserDefinedQuerywiseMetric>(config.Params));
}

TUserDefinedQuerywiseMetric::TUserDefinedQuerywiseMetric(const TLossParams& params)
    : TAdditiveSingleTargetMetric(ELossFunction::UserQuerywiseMetric, params)
    , Alpha(params.GetParamsMap().contains("alpha") ? FromString<float>(params.GetParamsMap().at("alpha")) : DefaultAlpha)
{
    UseWeights.MakeIgnored();
}

TMetricHolder TUserDefinedQuerywiseMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> /*approx*/,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
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
    return TMetricHolder(2);
}

EErrorType TUserDefinedQuerywiseMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

void TUserDefinedQuerywiseMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TUserDefinedQuerywiseMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"alpha", false, DefaultAlpha}
            },
            ""
        }
    };
};

/* Huber loss */

namespace {
    struct THuberLossMetric final: public TAdditiveSingleTargetMetric {

        explicit THuberLossMetric(const TLossParams& params,
                                  double delta)
            : TAdditiveSingleTargetMetric(ELossFunction::Huber, params)
            , Delta(delta) {
            CB_ENSURE(delta >= 0, "Huber metric is defined for delta >= 0, got " << delta);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;

        void GetBestValue(EMetricBestValue *valueType, float *bestValue) const override;

    private:
        const double Delta;
    };
}

// static.
TVector<THolder<IMetric>> THuberLossMetric::Create(const TMetricConfig& config) {
    CB_ENSURE(config.GetParamsMap().contains("delta"), "Metric " << ELossFunction::Huber << " requires delta as parameter");
    config.ValidParams->insert("delta");
    return AsVector(MakeHolder<THuberLossMetric>(
        config.Params, FromString<float>(config.GetParamsMap().at("delta"))));
}

TMetricHolder THuberLossMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int begin,
        int end
) const {
    Y_ASSERT(approxDelta.empty());
    Y_ASSERT(!isExpApprox);
    const auto approxVals = approx[0];
    Y_ASSERT(target.size() == approxVals.size());

    TMetricHolder error(2);

    bool hasWeight = !weight.empty();

    for (int k : xrange(begin, end)) {
        double targetMismatch = fabs(approxVals[k] - target[k]);
        const float w = hasWeight ? weight[k] : 1;
        if (targetMismatch < Delta) {
            error.Stats[0] += 0.5 * Sqr(targetMismatch) * w;
        } else {
            error.Stats[0] += Delta * (targetMismatch - 0.5 * Delta) * w;
        }
        error.Stats[1] += w;
    }
    return error;
}

void THuberLossMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> THuberLossMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"delta", true, {}}
            },
            ""
        }
    };
};

/* FilteredNdcg */

namespace {
    class TFilteredDcgMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TFilteredDcgMetric(const TLossParams& params,
                                    ENdcgMetricType metricType,
                                    ENdcgDenominatorType denominatorType,
                                    ENdcgSortType sortType);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int begin,
                int end
        ) const override;

        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        const ENdcgMetricType MetricType;
        const ENdcgDenominatorType DenominatorType;
        const ENdcgSortType SortType;

        static constexpr ENdcgMetricType DefaultMetricType = ENdcgMetricType::Base;
        static constexpr ENdcgDenominatorType DefaultDenominatorType= ENdcgDenominatorType::Position;
        static constexpr size_t LargeGroupSize = 10 * 1000;
    };
}

// static.
TVector<THolder<IMetric>> TFilteredDcgMetric::Create(const TMetricConfig& config) {
    auto type = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "type", DefaultMetricType);
    auto denominator = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "denominator", DefaultDenominatorType);
    auto sort = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "sort", ENdcgSortType::None);
    config.ValidParams->insert("type");
    config.ValidParams->insert("denominator");
    config.ValidParams->insert("sort");
    return AsVector(MakeHolder<TFilteredDcgMetric>(config.Params, type, denominator, sort));
}

TFilteredDcgMetric::TFilteredDcgMetric(const TLossParams& params,
                                       ENdcgMetricType metricType,
                                       ENdcgDenominatorType denominatorType,
                                       ENdcgSortType sortType)
    : TAdditiveSingleTargetMetric(ELossFunction::FilteredDCG, params)
    , MetricType(metricType)
    , DenominatorType(denominatorType)
    , SortType(sortType) {
    UseWeights.MakeIgnored();
}

TMetricHolder TFilteredDcgMetric::EvalSingleThread(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<TConstArrayRef<double>> approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int queryBegin,
        int queryEnd
) const {
    Y_ASSERT(!isExpApprox);
    CB_ENSURE(weight.empty(), "Weights are not supported for DCG metric");

    TMetricHolder metric(2);
    TVector<double> filteredApprox;
    TVector<double> filteredTarget;
    TVector<NMetrics::TSample> samples;
    TVector<double> decay;
    decay.yresize(LargeGroupSize);
    FillDcgDecay(DenominatorType, Nothing(), decay);

    for(int queryIndex = queryBegin; queryIndex < queryEnd; ++queryIndex) {
        const int begin = queriesInfo[queryIndex].Begin;
        const int end = queriesInfo[queryIndex].End;
        filteredApprox.clear();
        filteredTarget.clear();
        filteredApprox.reserve(end - begin);
        filteredTarget.reserve(end - begin);
        for (int i = begin; i < end; ++i) {
            const double currentApprox = approxDelta.empty() ? approx[0][i] : approx[0][i] + approxDelta[0][i];
            if (currentApprox >= 0.0) {
                filteredApprox.push_back(currentApprox);
                filteredTarget.push_back(target[i]);
            }
        }
        if (filteredApprox.empty()) {
            continue;
        }
        if (begin + decay.size() < (size_t)end) {
            decay.resize(2 * (end - begin));
            FillDcgDecay(DenominatorType, Nothing(), decay);
        }
        switch (SortType) {
            case ENdcgSortType::None:
                metric.Stats[0] += CalcDcgSorted(filteredTarget, decay, MetricType);
                break;
            case ENdcgSortType::ByPrediction: {
                NMetrics::TSample::FromVectors(filteredTarget, filteredApprox, &samples);
                metric.Stats[0] += CalcDcg(samples, decay, MetricType, Max<ui32>());
                break;
            }
            case ENdcgSortType::ByTarget: {
                NMetrics::TSample::FromVectors(filteredTarget, filteredApprox, &samples);
                metric.Stats[0] += CalcIDcg(samples, decay, MetricType, Max<ui32>());
                break;
            }
        }
    }
    metric.Stats[1] = queryEnd - queryBegin;
    return metric;
}

EErrorType TFilteredDcgMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

void TFilteredDcgMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = std::numeric_limits<double>::infinity();
    }
}

TVector<TParamSet> TFilteredDcgMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"type", false, ToString(DefaultMetricType)},
                TParamInfo{"denominator", false, ToString(DefaultDenominatorType)},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            ""
        }
    };
};

/* AverageGain */

namespace {
    class TAverageGain final: public TAdditiveSingleTargetMetric {
    public:
        explicit TAverageGain(ELossFunction lossFunction, const TLossParams& params, float topSize)
            : TAdditiveSingleTargetMetric(lossFunction, params)
            , TopSize(topSize) {
            CB_ENSURE(topSize > 0, "top size for AverageGain should be greater than 0");
            CB_ENSURE(topSize == (int)topSize, "top size for AverageGain should be an integer value");
            UseWeights.SetDefaultValue(true);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    private:
        const int TopSize;
    };
}

TVector<THolder<IMetric>> TAverageGain::Create(const TMetricConfig& config) {
    auto it = config.GetParamsMap().find("top");
    CB_ENSURE(it != config.GetParamsMap().end(), "AverageGain metric should have top parameter");
    config.ValidParams->insert("top");
    return AsVector(MakeHolder<TAverageGain>(config.Metric, config.Params, FromString<float>(it->second)));
}

TMetricHolder TAverageGain::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

void TAverageGain::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TAverageGain::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"top", true, {}}
            },
            ""
        }
    };
}

/* QueryAUC */

namespace {
    class TQueryAUCMetric final: public TAdditiveSingleTargetMetric {
    public:
        explicit TQueryAUCMetric(const TLossParams& params, EAucType singleClassType)
        : TAdditiveSingleTargetMetric(ELossFunction::QueryAUC, params)
        , Type(singleClassType) {
            UseWeights.SetDefaultValue(false);
        }

        explicit TQueryAUCMetric(const TLossParams& params, int positiveClass)
        : TAdditiveSingleTargetMetric(ELossFunction::QueryAUC, params)
        , PositiveClass(positiveClass)
        , Type(EAucType::OneVsAll) {
            UseWeights.SetDefaultValue(false);
        }

        explicit TQueryAUCMetric(const TLossParams& params, const TMaybe<TVector<TVector<double>>>& misclassCostMatrix = Nothing())
            : TAdditiveSingleTargetMetric(ELossFunction::QueryAUC, params)
            , Type(EAucType::Mu)
            , MisclassCostMatrix(misclassCostMatrix) {
            UseWeights.SetDefaultValue(false);
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    private:
        int PositiveClass = 1;
        EAucType Type;
        TMaybe<TVector<TVector<double>>> MisclassCostMatrix = Nothing();
    };
}

TVector<THolder<IMetric>> TQueryAUCMetric::Create(const TMetricConfig& config) {
    config.ValidParams->insert("type");
    EAucType aucType = config.ApproxDimension == 1 ? EAucType::Classic : EAucType::Mu;
    if (config.GetParamsMap().contains("type")) {
        const TString name = config.GetParamsMap().at("type");
        aucType = FromString<EAucType>(name);
        if (config.ApproxDimension == 1) {
            CB_ENSURE(aucType == EAucType::Classic || aucType == EAucType::Ranking,
                      "QueryAUC type \"" << aucType << "\" isn't a singleclass AUC type");
        } else {
            CB_ENSURE(aucType == EAucType::Mu || aucType == EAucType::OneVsAll,
                      "QueryAUC type \"" << aucType << "\" isn't a multiclass AUC type");
        }
    }

    switch (aucType) {
        case EAucType::Classic: {
            return AsVector(MakeHolder<TQueryAUCMetric>(config.Params, EAucType::Classic));
            break;
        }
        case EAucType::Ranking: {
            return AsVector(MakeHolder<TQueryAUCMetric>(config.Params, EAucType::Ranking));
            break;
        }
        case EAucType::Mu: {
            config.ValidParams->insert("misclass_cost_matrix");
            TMaybe<TVector<TVector<double>>> misclassCostMatrix = Nothing();
            if (config.GetParamsMap().contains("misclass_cost_matrix")) {
                misclassCostMatrix.ConstructInPlace(ConstructSquareMatrix<double>(
                        config.GetParamsMap().at("misclass_cost_matrix")));
            }
            if (misclassCostMatrix) {
                for (ui32 i = 0; i < misclassCostMatrix->size(); ++i) {
                    CB_ENSURE((*misclassCostMatrix)[i][i] == 0, "Diagonal elements of the misclass cost matrix should be equal to 0.");
                }
            }
            return AsVector(MakeHolder<TQueryAUCMetric>(config.Params, misclassCostMatrix));
            break;
        }
        case EAucType::OneVsAll: {
            TVector<THolder<IMetric>> metrics;
            for (int i = 0; i < config.ApproxDimension; ++i) {
                metrics.push_back(MakeHolder<TQueryAUCMetric>(config.Params, i));
            }
            return metrics;
            break;
        }
        default: {
            CB_ENSURE(false, "Unexpected AUC type");
        }
    }
}

TMetricHolder TQueryAUCMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int queryStartIndex,
    int queryEndIndex
) const {
    Y_ASSERT(!isExpApprox);

    TMetricHolder error(2);

    const auto realApprox = [&](int idx) {
        return approx[Type == EAucType::OneVsAll ? PositiveClass : 0][idx]
        + (approxDelta.empty() ? 0.0 : approxDelta[Type == EAucType::OneVsAll ? PositiveClass : 0][idx]);
    };
    const auto realWeight = [&](int idx) {
        return UseWeights && !weight.empty() ? weight[idx] : 1.0;
    };
    const auto realTarget = [&](int idx) {
        return Type == EAucType::OneVsAll ? target[idx] == static_cast<double>(PositiveClass) : target[idx];
    };

    TVector<NMetrics::TSample> samples;
    for (int queryIndex = queryStartIndex; queryIndex < queryEndIndex; ++queryIndex) {
        auto startIdx = queriesInfo[queryIndex].Begin;
        auto endIdx = queriesInfo[queryIndex].End;
        auto querySize = endIdx - startIdx;
        const float queryWeight = UseWeights ? queriesInfo[queryIndex].Weight : 1.0;

        if (Type == EAucType::Ranking) {
            samples.clear();
            samples.reserve(endIdx - startIdx);
            for (int i : xrange(startIdx, endIdx)) {
                samples.emplace_back(realTarget(i), realApprox(i), realWeight(i));
            }

            error.Stats[0] += CalcAUC(&samples) * queryWeight;
        } else if (Type == EAucType::Mu) {
            TConstArrayRef<float> currentTarget(target.begin() + startIdx, target.begin() + endIdx);
            TConstArrayRef<float> currentWeight(weight.begin() + startIdx, weight.begin() + endIdx);

            TVector<TVector<double>> currentApprox;
            TVector<TVector<double>> currentApproxDelta;

            ResizeRank2(querySize, approx[0].size(), currentApprox);
            AssignRank2(TArrayRef(approx.begin() + startIdx, approx.begin() + endIdx), &currentApprox);

            if (!approxDelta.empty()) {
                ResizeRank2(querySize, approxDelta[0].size(), currentApproxDelta);
                AssignRank2(TArrayRef(approxDelta.begin() + startIdx, approxDelta.begin() + endIdx), &currentApproxDelta);

                for (ui32 i = 0; i < currentApprox.size(); ++i) {
                    for (ui32 j = 0; j < currentApprox[i].size(); ++j) {
                        currentApprox[i][j] += currentApproxDelta[i][j];
                    }
                }
            }

            error.Stats[0] = CalcMuAuc(currentApprox, currentTarget, UseWeights ? currentWeight : TConstArrayRef<float>(), 1, MisclassCostMatrix) * queryWeight;
        }
        else {
            TVector<NMetrics::TBinClassSample> positiveSamples, negativeSamples;
            for (int i : xrange(startIdx, endIdx)) {
                const auto currentTarget = realTarget(i);
                CB_ENSURE(0 <= currentTarget && currentTarget <= 1, "All target values should be in the segment [0, 1], for Ranking AUC please use type=Ranking.");
                if (currentTarget > 0) {
                    positiveSamples.emplace_back(realApprox(i), currentTarget * realWeight(i));
                }
                if (currentTarget < 1) {
                    negativeSamples.emplace_back(realApprox(i), (1 - currentTarget) * realWeight(i));
                }
            }
            error.Stats[0] += CalcBinClassAuc(&positiveSamples, &negativeSamples) * queryWeight;
        }
        error.Stats[1] += queryWeight;
    }

    return error;
}

EErrorType TQueryAUCMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

void TQueryAUCMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TQueryAUCMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, false},
                TParamInfo{"type", false, ToString(EAucType::Classic)},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            ""
        },
        TParamSet{
            {
                TParamInfo{"use_weights", false, false},
                TParamInfo{"type", false, ToString(EAucType::Mu)},
                TParamInfo{"hints", false, "skip_train~true"}
            },
            "Multiclass"
        }
    };
};

/* CombinationLoss */

namespace {
    class TCombinationLoss final: public TAdditiveSingleTargetMetric {
    public:
        explicit TCombinationLoss(const TLossParams& params)
        : TAdditiveSingleTargetMetric(ELossFunction::Combination, params)
        , Params(params.GetParamsMap())
        {
        }

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            bool isExpApprox,
            TConstArrayRef<float> target,
            TConstArrayRef<float> weight,
            TConstArrayRef<TQueryInfo> queriesInfo,
            int queryStartIndex,
            int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        TString GetDescription() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
        double GetFinalError(const TMetricHolder& error) const override;
    private:
        const TMap<TString, TString> Params;
    };
}

// static.
TVector<THolder<IMetric>> TCombinationLoss::Create(const TMetricConfig& config) {
    CB_ENSURE(config.ApproxDimension == 1, "Combination loss cannot be used in multi-classification");
    CB_ENSURE(config.GetParamsMap().size() >= 2, "Combination loss must have 2 or more parameters");
    CB_ENSURE(config.GetParamsMap().size() % 2 == 0, "Combination loss must have even number of parameters, not " << config.GetParamsMap().size());
    const ui32 lossCount = config.GetParamsMap().size() / 2;
    for (ui32 idx : xrange(lossCount)) {
        config.ValidParams->insert(GetCombinationLossKey(idx));
        config.ValidParams->insert(GetCombinationWeightKey(idx));
    }
    return AsVector(MakeHolder<TCombinationLoss>(config.Params));
}

TMetricHolder TCombinationLoss::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> /*approx*/,
    TConstArrayRef<TConstArrayRef<double>> /*approxDelta*/,
    bool /*isExpApprox*/,
    TConstArrayRef<float> /*target*/,
    TConstArrayRef<float> /*weight*/,
    TConstArrayRef<TQueryInfo> /*queriesInfo*/,
    int /*queryStartIndex*/,
    int /*queryEndIndex*/
) const {
    CB_ENSURE(false, "Combination loss is implemented only on GPU");
}

EErrorType TCombinationLoss::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TString TCombinationLoss::GetDescription() const {
    TString description;
    for (const auto& [param, value] : Params) {
        description += BuildDescription(TMetricParam<TString>(param, value, /*userDefined*/true));
    }
    return description;
}

void TCombinationLoss::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

double TCombinationLoss::GetFinalError(const TMetricHolder& error) const {
    return error.Stats[0];
}

TVector<TParamSet> TCombinationLoss::ValidParamSets() {
    return {TParamSet{{}, ""}};
};

// TQueryCrossEntropyMetric

static inline bool IsSingleClassQuery(const float* targets, int querySize) {
    for (int i = 1; i < querySize; ++i) {
        if (Abs(targets[i] - targets[0]) > 1e-20) {
            return false;
        }
    }
    return true;
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

namespace {
    struct TQueryCrossEntropyMetric final: public TAdditiveSingleTargetMetric {
        explicit TQueryCrossEntropyMetric(const TLossParams& params,
                                          double alpha);
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        void AddSingleQuery(const double* approxes,
                            const float* target,
                            const float* weight,
                            int querySize,
                            TMetricHolder* metricHolder) const;
    private:
        const double Alpha;
        static constexpr double DefaultAlpha = 0.95;
    };
}

// static.
TVector<THolder<IMetric>> TQueryCrossEntropyMetric::Create(const TMetricConfig& config) {
    auto it = config.GetParamsMap().find("alpha");
    config.ValidParams->insert("alpha");
    config.ValidParams->insert("raw_values_scale");
    return AsVector(MakeHolder<TQueryCrossEntropyMetric>(
        config.Params,
        it != config.GetParamsMap().end() ? FromString<float>(it->second) : DefaultAlpha));
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


TMetricHolder TQueryCrossEntropyMetric::EvalSingleThread(TConstArrayRef<TConstArrayRef<double>> approx,
                                                         TConstArrayRef<TConstArrayRef<double>> approxDelta,
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

TQueryCrossEntropyMetric::TQueryCrossEntropyMetric(const TLossParams& params,
                                                   double alpha)
        : TAdditiveSingleTargetMetric(ELossFunction::QueryCrossEntropy, params)
        , Alpha(alpha) {
    UseWeights.SetDefaultValue(true);
}

void TQueryCrossEntropyMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TQueryCrossEntropyMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"alpha", false, DefaultAlpha},
                TParamInfo{"raw_values_scale", false, {}}
            },
            ""
        }
    };
};

namespace {
    struct TMRRMetric final: public TAdditiveSingleTargetMetric {
        explicit TMRRMetric(const TLossParams& params, int topSize, float targetBorder);
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static inline double CalcQueryReciprocalRank(const double* approxes, const float* target,
                                                     int querySize, int topSize, float targetBorder);
        static constexpr int DefaultTopSize = -1;
        const int TopSize;
        const float TargetBorder;
    };
}

TVector<THolder<IMetric>> TMRRMetric::Create(const TMetricConfig& config) {
    const int topSize = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "top", DefaultTopSize);
    const float targetBorder = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "border", GetDefaultTargetBorder());
    config.ValidParams->insert("top");
    config.ValidParams->insert("border");
    return AsVector(MakeHolder<TMRRMetric>(config.Params, topSize, targetBorder));
}

TVector<TParamSet> TMRRMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"top", false, DefaultTopSize},
                TParamInfo{"border", false, GetDefaultTargetBorder()}
            },
            ""
        }
    };
};

double TMRRMetric::CalcQueryReciprocalRank(
    const double* approxes,
    const float* targets,
    int querySize,
    int topSize,
    float targetBorder
) {
    bool foundRelevantApprox = false;
    double maxRelevantApprox = std::numeric_limits<double>::lowest();
    for (int i = 0; i < querySize; ++i) {
        if (targets[i] > targetBorder) {
            foundRelevantApprox = true;
            maxRelevantApprox = Max(maxRelevantApprox, approxes[i]);
        }
    }
    if (!foundRelevantApprox) {
        return 0.0;
    }

    int pos = 1;
    const int maxPos = topSize == -1 ? querySize : Min(querySize, topSize);
    for (int i = 0; i < querySize && pos <= maxPos; ++i) {
        pos += approxes[i] > maxRelevantApprox || approxes[i] == maxRelevantApprox && targets[i] <= targetBorder;
    }
    return pos <= maxPos ? 1.0 / pos : 0.0;
}

TMetricHolder TMRRMetric::EvalSingleThread(TConstArrayRef<TConstArrayRef<double>> approx,
                                           TConstArrayRef<TConstArrayRef<double>> approxDelta,
                                           bool /*isExpApprox*/,
                                           TConstArrayRef<float> target,
                                           TConstArrayRef<float> /*weight*/,
                                           TConstArrayRef<TQueryInfo> queriesInfo,
                                           int queryStartIndex,
                                           int queryEndIndex) const {
    Y_ASSERT(approxDelta.empty());
    TMetricHolder result(2);
    for (int qid = queryStartIndex; qid < queryEndIndex; ++qid) {
        auto& qidInfo = queriesInfo[qid];
        const double qrr = CalcQueryReciprocalRank(
                approx[0].data() + qidInfo.Begin,
                target.data() + qidInfo.Begin,
                qidInfo.End - qidInfo.Begin,
                TopSize,
                TargetBorder);
        const float queryWeight = UseWeights ? qidInfo.Weight : 1.f;
        result.Stats[0] += queryWeight * qrr;
        result.Stats[1] += queryWeight;
    }
    return result;
}

EErrorType TMRRMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TMRRMetric::TMRRMetric(const TLossParams& params, int topSize, float targetBorder)
    : TAdditiveSingleTargetMetric(ELossFunction::MRR, params)
    , TopSize(topSize)
    , TargetBorder(targetBorder)
{
    UseWeights.SetDefaultValue(true);
}

void TMRRMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}


namespace {
    struct TERRMetric final: public TAdditiveSingleTargetMetric {
        explicit TERRMetric(const TLossParams& params, int topSize);
        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();
        TMetricHolder EvalSingleThread(
                TConstArrayRef<TConstArrayRef<double>> approx,
                TConstArrayRef<TConstArrayRef<double>> approxDelta,
                bool isExpApprox,
                TConstArrayRef<float> target,
                TConstArrayRef<float> weight,
                TConstArrayRef<TQueryInfo> queriesInfo,
                int queryStartIndex,
                int queryEndIndex
        ) const override;
        EErrorType GetErrorType() const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;

    private:
        static inline double CalcQueryERR(const double* approxes, const float* target,
                                          int querySize, int topSize, TVector<ui32>* indices);

        static constexpr int DefaultTopSize = -1;
        const int TopSize;
    };
}

TVector<THolder<IMetric>> TERRMetric::Create(const TMetricConfig& config) {
    const int topSize = NCatboostOptions::GetParamOrDefault(config.GetParamsMap(), "top", DefaultTopSize);
    config.ValidParams->insert("top");
    return AsVector(MakeHolder<TERRMetric>(config.Params, topSize));
}

double TERRMetric::CalcQueryERR(
    const double* approxes,
    const float* targets,
    int querySize,
    int topSize,
    TVector<ui32>* indicesPtr
) {
    TVector<ui32>& indices = *indicesPtr;
    if (static_cast<int>(indices.size()) < querySize) {
        indices.yresize(querySize);
    }
    Iota(indices.begin(), indices.begin() + querySize, static_cast<ui32>(0));
    const int lookupDepth = topSize == -1 ? querySize : Min(querySize, topSize);
    PartialSort(indices.begin(), indices.begin() + lookupDepth, indices.begin() + querySize, [&](ui32 a, ui32 b) {
        return approxes[a] > approxes[b] || (approxes[a] == approxes[b] && targets[a] < targets[b]);
    });
    double queryRR = 0.0;
    double pLook = 1.0;
    for (int i = 0; i < lookupDepth; ++i) {
        const ui32 docIndex = indices[i];
        queryRR += pLook * targets[docIndex] / (i + 1);
        pLook *= 1 - targets[docIndex];
    }
    return queryRR;
}

TMetricHolder TERRMetric::EvalSingleThread(TConstArrayRef<TConstArrayRef<double>> approx,
                                           TConstArrayRef<TConstArrayRef<double>> approxDelta,
                                           bool /*isExpApprox*/,
                                           TConstArrayRef<float> target,
                                           TConstArrayRef<float> /*weight*/,
                                           TConstArrayRef<TQueryInfo> queriesInfo,
                                           int queryStartIndex,
                                           int queryEndIndex) const {
    Y_ASSERT(approxDelta.empty());
    TMetricHolder result(2);
    TVector<ui32> indices;
    for (int qid = queryStartIndex; qid < queryEndIndex; ++qid) {
        auto& qidInfo = queriesInfo[qid];
        const double qrr = CalcQueryERR(
                approx[0].data() + qidInfo.Begin,
                target.data() + qidInfo.Begin,
                qidInfo.End - qidInfo.Begin,
                TopSize,
                &indices);
        const float queryWeight = UseWeights ? qidInfo.Weight : 1.f;
        result.Stats[0] += queryWeight * qrr;
        result.Stats[1] += queryWeight;
    }
    return result;
}

EErrorType TERRMetric::GetErrorType() const {
    return EErrorType::QuerywiseError;
}

TERRMetric::TERRMetric(const TLossParams& params, int topSize)
    : TAdditiveSingleTargetMetric(ELossFunction::ERR, params)
    , TopSize(topSize)
{
    UseWeights.SetDefaultValue(true);
}

void TERRMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Max;
    if (bestValue) {
        *bestValue = 1;
    }
}

TVector<TParamSet> TERRMetric::ValidParamSets() {
    return {
        TParamSet{
            {
                TParamInfo{"use_weights", false, true},
                TParamInfo{"top", false, DefaultTopSize}
            },
            ""
        }
    };
};

/* MultiCrossEntropy */
namespace {
    struct TMultiCrossEntropyMetric final: public TAdditiveMultiTargetMetric {
        explicit TMultiCrossEntropyMetric(ELossFunction lossFunction, const TLossParams& params);

        static TVector<THolder<IMetric>> Create(const TMetricConfig& config);
        static TVector<TParamSet> ValidParamSets();

        TMetricHolder EvalSingleThread(
            TConstArrayRef<TConstArrayRef<double>> approx,
            TConstArrayRef<TConstArrayRef<double>> approxDelta,
            TConstArrayRef<TConstArrayRef<float>> target,
            TConstArrayRef<float> weight,
            int begin,
            int end
        ) const override;
        void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    };
}

TMultiCrossEntropyMetric::TMultiCrossEntropyMetric(ELossFunction lossFunction, const TLossParams& params)
    : TAdditiveMultiTargetMetric(lossFunction, params)
{
    CB_ENSURE_INTERNAL(
        lossFunction == ELossFunction::MultiLogloss || lossFunction == ELossFunction::MultiCrossEntropy,
        "lossFunction " << lossFunction
    );
}

TVector<THolder<IMetric>> TMultiCrossEntropyMetric::Create(const TMetricConfig& config) {
    return AsVector(MakeHolder<TMultiCrossEntropyMetric>(config.Metric, config.Params));
}

TMetricHolder TMultiCrossEntropyMetric::EvalSingleThread(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    int begin,
    int end
) const {
    const int approxDimension = approx.ysize();

    TMetricHolder error(2);
    constexpr auto BlockSize = 32;
    std::array<double, BlockSize> zeroDelta;
    std::array<double, BlockSize> expApprox;
    std::array<float, BlockSize> unitWeight;
    zeroDelta.fill(0.0);
    unitWeight.fill(1.0f);
    double sumDimErrors = 0;
    for (int dim = 0; dim < approxDimension; ++dim) {
        for (int docIdx = begin; docIdx < end; docIdx += BlockSize) {
            auto count = Min<int>(end - docIdx, BlockSize);
            TConstArrayRef<double> approxRef(&approx[dim][docIdx], count);
            TConstArrayRef<double> approxDeltaRef(approxDelta.empty() ? &zeroDelta[0] : &approxDelta[dim][docIdx], count);
            TConstArrayRef<float> targetRef(&target[dim][docIdx], count);
            TConstArrayRef<float> weightRef(weight.empty() ? &unitWeight[0] : &weight[docIdx], count);
            for (int j = 0; j < count; ++j) {
                expApprox[j] = approxRef[j] + approxDeltaRef[j];
            }
            FastExpInplace(&expApprox[0], count);
            for (int j = 0; j < count; ++j) {
                const auto evaluatedApprox = approxRef[j] + approxDeltaRef[j];
                const auto w = weightRef[j];
                sumDimErrors += (IsFinite(expApprox[j]) ? -log(1 + expApprox[j]) : -evaluatedApprox) * w;
                sumDimErrors += (targetRef[j] * evaluatedApprox) * w;
            }
        }
    }
    error.Stats[0] = -sumDimErrors / approxDimension;
    error.Stats[1] = weight.empty() ? end - begin : Accumulate(weight, 0);
    return error;
}

void TMultiCrossEntropyMetric::GetBestValue(EMetricBestValue* valueType, float* bestValue) const {
    *valueType = EMetricBestValue::Min;
    if (bestValue) {
        *bestValue = 0;
    }
}

TVector<TParamSet> TMultiCrossEntropyMetric::ValidParamSets() {
    return {TParamSet{{TParamInfo{"use_weights", false, true}}, ""}};
};

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

static bool HintedToEvalOnTrain(const TMap<TString, TString>& params) {
    const bool hasHints = params.contains("hints");
    const auto& hints = hasHints ? ParseHintsDescription(params.at("hints")) : TMap<TString, TString>();
    return hasHints && hints.contains("skip_train") && hints.at("skip_train") == "false";
}

static bool HintedToEvalOnTrain(const NCatboostOptions::TLossDescription& metricDescription) {
    return HintedToEvalOnTrain(metricDescription.GetLossParamsMap());
}

TVector<THolder<IMetric>> CreateMetric(ELossFunction metric, const TLossParams& params, int approxDimension) {
    TVector<THolder<IMetric>> result;
    TSet<TString> validParams;
    TMetricConfig config(metric, params, approxDimension, &validParams);

    switch (metric) {
        case ELossFunction::MultiRMSE:
            AppendTemporaryMetricsVector(TMultiRMSEMetric::Create(config), &result);
            break;
        case ELossFunction::MultiRMSEWithMissingValues:
            AppendTemporaryMetricsVector(TMultiRMSEWithMissingValues::Create(config), &result);
            break;
        case ELossFunction::SurvivalAft:
           AppendTemporaryMetricsVector(TSurvivalAftMetric::Create(config), &result);
           break;
        case ELossFunction::RMSEWithUncertainty:
            AppendTemporaryMetricsVector(TRMSEWithUncertaintyMetric::Create(config), &result);
            break;
        case ELossFunction::Logloss:
            AppendTemporaryMetricsVector(TCrossEntropyMetric::Create(config), &result);
            break;
        case ELossFunction::CrossEntropy:
            AppendTemporaryMetricsVector(TCrossEntropyMetric::Create(config), &result);
            break;
        case ELossFunction::RMSE:
            AppendTemporaryMetricsVector(TRMSEMetric::Create(config), &result);
            break;
        case ELossFunction::Lq:
            AppendTemporaryMetricsVector(TLqMetric::Create(config), &result);
            break;
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            AppendTemporaryMetricsVector(TQuantileMetric::Create(config), &result);
            break;
        case ELossFunction::MultiQuantile:
            AppendTemporaryMetricsVector(TMultiQuantileMetric::Create(config), &result);
            break;
        case ELossFunction::Expectile:
            AppendTemporaryMetricsVector(TExpectileMetric::Create(config), &result);
            break;
        case ELossFunction::LogLinQuantile:
            AppendTemporaryMetricsVector(TLogLinQuantileMetric::Create(config), &result);
            break;
        case ELossFunction::AverageGain:
        case ELossFunction::QueryAverage:
            AppendTemporaryMetricsVector(TAverageGain::Create(config), &result);
            break;
        case ELossFunction::MAPE:
            AppendTemporaryMetricsVector(TMAPEMetric::Create(config), &result);
            break;
        case ELossFunction::Poisson:
            AppendTemporaryMetricsVector(TPoissonMetric::Create(config), &result);
            break;
        case ELossFunction::Tweedie:
            AppendTemporaryMetricsVector(TTweedieMetric::Create(config), &result);
            break;
        case ELossFunction::Focal:
            AppendTemporaryMetricsVector(TFocalMetric::Create(config), &result);
            break;
        case ELossFunction::LogCosh:
            AppendTemporaryMetricsVector(TLogCoshMetric::Create(config), &result);
            break;
        case ELossFunction::MedianAbsoluteError:
            AppendTemporaryMetricsVector(TMedianAbsoluteErrorMetric::Create(config), &result);
            break;
        case ELossFunction::SMAPE:
            AppendTemporaryMetricsVector(TSMAPEMetric::Create(config), &result);
            break;
        case ELossFunction::MSLE:
            AppendTemporaryMetricsVector(TMSLEMetric::Create(config), &result);
            break;
        case ELossFunction::PRAUC:
            AppendTemporaryMetricsVector(TPRAUCMetric::Create(config), &result);
            break;
        case ELossFunction::MultiClass:
            AppendTemporaryMetricsVector(TMultiClassMetric::Create(config), &result);
            break;
        case ELossFunction::MultiClassOneVsAll:
            AppendTemporaryMetricsVector(TMultiClassOneVsAllMetric::Create(config), &result);
            break;
        case ELossFunction::MultiLogloss:
        case ELossFunction::MultiCrossEntropy:
            AppendTemporaryMetricsVector(TMultiCrossEntropyMetric::Create(config), &result);
            break;
        case ELossFunction::PairLogit:
            AppendTemporaryMetricsVector(TPairLogitMetric::Create(config), &result);
            break;
        case ELossFunction::QueryRMSE:
            AppendTemporaryMetricsVector(TQueryRMSEMetric::Create(config), &result);
            break;
        case ELossFunction::QueryAUC:
            AppendTemporaryMetricsVector(TQueryAUCMetric::Create(config), &result);
            break;
        case ELossFunction::QuerySoftMax:
            AppendTemporaryMetricsVector(TQuerySoftMaxMetric::Create(config), &result);
            break;
        case ELossFunction::PFound:
            AppendTemporaryMetricsVector(TPFoundMetric::Create(config), &result);
            break;
        case ELossFunction::LogLikelihoodOfPrediction:
            AppendTemporaryMetricsVector(TLLPMetric::Create(config), &result);
            break;
        case ELossFunction::DCG:
        case ELossFunction::NDCG:
            AppendTemporaryMetricsVector(TDcgMetric::Create(config), &result);
            break;
        case ELossFunction::R2:
            AppendTemporaryMetricsVector(TR2Metric::Create(config), &result);
            break;
        case ELossFunction::NumErrors:
            AppendTemporaryMetricsVector(TNumErrorsMetric::Create(config), &result);
            break;
        case ELossFunction::AUC:
            AppendTemporaryMetricsVector(TAUCMetric::Create(config), &result);
            break;
        case ELossFunction::BalancedAccuracy:
            AppendTemporaryMetricsVector(TBalancedAccuracyMetric::Create(config), &result);
            break;
        case ELossFunction::BalancedErrorRate:
            AppendTemporaryMetricsVector(TBalancedErrorRate::Create(config), &result);
            break;
        case ELossFunction::HingeLoss:
            AppendTemporaryMetricsVector(THingeLossMetric::Create(config), &result);
            break;
        case ELossFunction::PairAccuracy:
            AppendTemporaryMetricsVector(TPairAccuracyMetric::Create(config), &result);
            break;
        case ELossFunction::PrecisionAt:
            AppendTemporaryMetricsVector(TPrecisionAtKMetric::Create(config), &result);
            break;
        case ELossFunction::RecallAt:
            AppendTemporaryMetricsVector(TRecallAtKMetric::Create(config), &result);
            break;
        case ELossFunction::MAP:
            AppendTemporaryMetricsVector(TMAPKMetric::Create(config), &result);
            break;
        case ELossFunction::UserPerObjMetric:
            AppendTemporaryMetricsVector(TUserDefinedPerObjectMetric::Create(config), &result);
            break;
        case ELossFunction::UserQuerywiseMetric:
            AppendTemporaryMetricsVector(TUserDefinedQuerywiseMetric::Create(config), &result);
            break;
        case ELossFunction::QueryCrossEntropy:
            AppendTemporaryMetricsVector(TQueryCrossEntropyMetric::Create(config), &result);
            break;
        case ELossFunction::MRR:
            AppendTemporaryMetricsVector(TMRRMetric::Create(config), &result);
            break;
        case ELossFunction::ERR:
            AppendTemporaryMetricsVector(TERRMetric::Create(config), &result);
            break;
        case ELossFunction::Huber:
            AppendTemporaryMetricsVector(THuberLossMetric::Create(config), &result);
            break;
        case ELossFunction::FilteredDCG:
            AppendTemporaryMetricsVector(TFilteredDcgMetric::Create(config), &result);
            break;
        case ELossFunction::FairLoss:
            AppendTemporaryMetricsVector(TFairLossMetric::Create(config), &result);
            break;
        case ELossFunction::NormalizedGini:
            AppendTemporaryMetricsVector(TNormalizedGini::Create(config), &result);
            break;
        case ELossFunction::Combination:
            AppendTemporaryMetricsVector(TCombinationLoss::Create(config), &result);
            break;
        case ELossFunction::Cox:
            AppendTemporaryMetricsVector(TCoxMetric::Create(config), &result);
            break;
        default: {
            result = CreateCachingMetrics(config);

            if (!result) {
                CB_ENSURE(false, "Unsupported metric: " << metric);
                return TVector<THolder<IMetric>>();
            }
            break;
        }
    }

    if (IsBinaryClassCompatibleMetric(metric)) {
        validParams.insert(NCatboostOptions::TMetricOptions::PREDICTION_BORDER_PARAM);
    }

    validParams.insert("hints");
    if (result && !result[0]->UseWeights.IsIgnored()) {
        validParams.insert("use_weights");
    }

    if (ShouldSkipCalcOnTrainByDefault(metric)) {
        for (THolder<IMetric>& metricHolder : result) {
            metricHolder->AddHint("skip_train", "true");
        }
        if (!HintedToEvalOnTrain(params.GetParamsMap())) {
            CATBOOST_INFO_LOG << "Metric " << metric << " is not calculated on train by default. To calculate this metric on train, add hints=skip_train~false to metric parameters." << Endl;
        }
    }

    if (params.GetParamsMap().contains("hints")) { // TODO(smirnovpavel): hints shouldn't be added for each metric
        TMap<TString, TString> hints = ParseHintsDescription(params.GetParamsMap().at("hints"));
        for (const auto& hint : hints) {
            for (THolder<IMetric>& metricHolder : result) {
                metricHolder->AddHint(hint.first, hint.second);
            }
        }
    }

    if (params.GetParamsMap().contains("use_weights")) {
        const bool useWeights = FromString<bool>(params.GetParamsMap().at("use_weights"));
        for (THolder<IMetric>& metricHolder : result) {
            metricHolder->UseWeights = useWeights;
        }
    }

    CheckParameters(ToString(metric), validParams, params.GetParamsMap());

    if (metric == ELossFunction::Combination) {
        CheckCombinationParameters(params.GetParamsMap());
    }

    return result;
}

TVector<THolder<TSingleTargetMetric>> CreateSingleTargetMetric(ELossFunction metric, const TLossParams& params, int approxDimension) {
    auto metrics = CreateMetric(metric, params, approxDimension);
    TVector<THolder<TSingleTargetMetric>> singleTargetMetrics;
    singleTargetMetrics.reserve(metrics.size());
    for (auto& metric : metrics) {
        singleTargetMetrics.emplace_back(dynamic_cast<TSingleTargetMetric*>(metric.Release()));
    }
    return singleTargetMetrics;
}

static TVector<THolder<IMetric>> CreateMetricFromDescription(const TString& description, int approxDimension) {
    ELossFunction metric = ParseLossType(description);
    TLossParams params = ParseLossParams(description);
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

TVector<TParamSet> ValidParamSets(ELossFunction metric) {
    switch (metric) {
        case ELossFunction::MultiRMSE:
            return TMultiRMSEMetric::ValidParamSets();
        case ELossFunction::MultiRMSEWithMissingValues:
            return TMultiRMSEWithMissingValues::ValidParamSets();
        case ELossFunction::RMSEWithUncertainty:
            return TRMSEWithUncertaintyMetric::ValidParamSets();
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
            return TCrossEntropyMetric::ValidParamSets();
        case ELossFunction::RMSE:
            return TRMSEMetric::ValidParamSets();
        case ELossFunction::Lq:
            return TLqMetric::ValidParamSets();
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            return TQuantileMetric::ValidParamSets();
        case ELossFunction::MultiQuantile:
            return TMultiQuantileMetric::ValidParamSets();
        case ELossFunction::Expectile:
            return TExpectileMetric::ValidParamSets();
        case ELossFunction::LogLinQuantile:
            return TLogLinQuantileMetric::ValidParamSets();
        case ELossFunction::AverageGain:
        case ELossFunction::QueryAverage:
            return TAverageGain::ValidParamSets();
        case ELossFunction::MAPE:
            return TMAPEMetric::ValidParamSets();
        case ELossFunction::Poisson:
            return TPoissonMetric::ValidParamSets();
        case ELossFunction::Tweedie:
            return TTweedieMetric::ValidParamSets();
        case ELossFunction::Cox:
            return TCoxMetric::ValidParamSets();
        case ELossFunction::Focal:
            return TFocalMetric::ValidParamSets();
        case ELossFunction::LogCosh:
            return TLogCoshMetric::ValidParamSets();
        case ELossFunction::MedianAbsoluteError:
            return TMedianAbsoluteErrorMetric::ValidParamSets();
        case ELossFunction::SMAPE:
            return TSMAPEMetric::ValidParamSets();
        case ELossFunction::MSLE:
            return TMSLEMetric::ValidParamSets();
        case ELossFunction::PRAUC:
            return TPRAUCMetric::ValidParamSets();
        case ELossFunction::MultiClass:
            return TMultiClassMetric::ValidParamSets();
        case ELossFunction::MultiClassOneVsAll:
            return TMultiClassOneVsAllMetric::ValidParamSets();
        case ELossFunction::MultiLogloss:
        case ELossFunction::MultiCrossEntropy:
            return TMultiCrossEntropyMetric::ValidParamSets();
        case ELossFunction::PairLogit:
        case ELossFunction::PairLogitPairwise:
            return TPairLogitMetric::ValidParamSets();
        case ELossFunction::QueryRMSE:
            return TQueryRMSEMetric::ValidParamSets();
        case ELossFunction::QueryAUC:
            return TQueryAUCMetric::ValidParamSets();
        case ELossFunction::QuerySoftMax:
            return TQuerySoftMaxMetric::ValidParamSets();
        case ELossFunction::PFound:
            return TPFoundMetric::ValidParamSets();
        case ELossFunction::LogLikelihoodOfPrediction:
            return TLLPMetric::ValidParamSets();
        case ELossFunction::DCG:
        case ELossFunction::NDCG:
            return TDcgMetric::ValidParamSets();
        case ELossFunction::R2:
            return TR2Metric::ValidParamSets();
        case ELossFunction::NumErrors:
            return TNumErrorsMetric::ValidParamSets();
        case ELossFunction::AUC:
            return TAUCMetric::ValidParamSets();
        case ELossFunction::BalancedAccuracy:
            return TBalancedAccuracyMetric::ValidParamSets();
        case ELossFunction::BalancedErrorRate:
            return TBalancedErrorRate::ValidParamSets();
        case ELossFunction::HingeLoss:
            return THingeLossMetric::ValidParamSets();
        case ELossFunction::PairAccuracy:
            return TPairAccuracyMetric::ValidParamSets();
        case ELossFunction::PrecisionAt:
            return TPrecisionAtKMetric::ValidParamSets();
        case ELossFunction::RecallAt:
            return TRecallAtKMetric::ValidParamSets();
        case ELossFunction::MAP:
            return TMAPKMetric::ValidParamSets();
        case ELossFunction::UserPerObjMetric:
            return TUserDefinedPerObjectMetric::ValidParamSets();
        case ELossFunction::UserQuerywiseMetric:
            return TUserDefinedQuerywiseMetric::ValidParamSets();
        case ELossFunction::QueryCrossEntropy:
            return TQueryCrossEntropyMetric::ValidParamSets();
        case ELossFunction::MRR:
            return TMRRMetric::ValidParamSets();
        case ELossFunction::ERR:
            return TERRMetric::ValidParamSets();
        case ELossFunction::SurvivalAft:
            return TSurvivalAftMetric::ValidParamSets();
        case ELossFunction::Huber:
            return THuberLossMetric::ValidParamSets();
        case ELossFunction::FilteredDCG:
            return TFilteredDcgMetric::ValidParamSets();
        case ELossFunction::FairLoss:
            return TFairLossMetric::ValidParamSets();
        case ELossFunction::NormalizedGini:
            return TNormalizedGini::ValidParamSets();
        case ELossFunction::Combination:
            return TCombinationLoss::ValidParamSets();
        case ELossFunction::BrierScore:
            return TBrierScoreMetric::ValidParamSets();
        case ELossFunction::CtrFactor:
            return TCtrFactorMetric::ValidParamSets();
        default:
            return CachingMetricValidParamSets(metric);
    }
}


static bool IsSkipInMetricsParamsExport(ELossFunction lossFunction) {
    switch (lossFunction) {
        case ELossFunction::YetiRank:
        case ELossFunction::YetiRankPairwise:
        case ELossFunction::StochasticFilter:
        case ELossFunction::LambdaMart:
        case ELossFunction::StochasticRank:
            // objectives, cannot be used as metrics
            return true;
        case ELossFunction::PythonUserDefinedPerObject:
        case ELossFunction::PythonUserDefinedMultiTarget:
            // user-defined metrics and objectives in Python
            return true;

        default:
            return false;
    }
}


NJson::TJsonValue ExportAllMetricsParamsToJson() {
    NJson::TJsonValue exportJson;
    for (const ELossFunction& loss : GetEnumAllValues<ELossFunction>()) {
        if (IsSkipInMetricsParamsExport(loss)) {
            continue;
        }

        NJson::TJsonValue paramSets;
        for (const auto& paramSet : ValidParamSets(loss)) {
            NJson::TJsonValue metricJson;
            metricJson.InsertValue("_name_suffix", paramSet.NameSuffix);
            for (const auto& paramInfo : paramSet.ValidParams) {
                NJson::TJsonValue paramJson;
                paramJson.InsertValue("is_mandatory", paramInfo.IsMandatory);
                paramJson.InsertValue("default_value", paramInfo.DefaultValue);
                metricJson.InsertValue(paramInfo.Name, paramJson);
            }
            paramSets.AppendValue(metricJson);
        }
        exportJson.InsertValue(ToString(loss), paramSets);
    }
    return exportJson;
}

static inline bool ShouldConsiderWeightsByDefault(const THolder<IMetric>& metric) {
    return ParseLossType(metric->GetDescription()) != ELossFunction::AUC && !metric->UseWeights.IsUserDefined() && !metric->UseWeights.IsIgnored();
}

static void SetHintToCalcMetricOnTrain(const THashSet<TString>& metricsToCalcOnTrain, TVector<THolder<IMetric>>* errors) {
    for (auto& error : *errors) {
        if (metricsToCalcOnTrain.contains(error->GetDescription())) {
            error->AddHint("skip_train", "false");
        }
    }
}

void InitializeEvalMetricIfNotSet(
    const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& objectiveMetric,
    NCatboostOptions::TOption<NCatboostOptions::TLossDescription>* evalMetric){

    CB_ENSURE(objectiveMetric.IsSet(), "Objective metric must be set.");
    const NCatboostOptions::TLossDescription& objectiveMetricDescription = objectiveMetric.Get();
    if (evalMetric->NotSet()) {
        CB_ENSURE(!IsUserDefined(objectiveMetricDescription.GetLossFunction()),
                  "If loss function is a user defined object, then the eval metric must be specified.");
        evalMetric->Set(objectiveMetricDescription);
    }
}

TVector<THolder<IMetric>> CreateMetrics(
        const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        int approxDimension,
        bool hasWeights) {

    CB_ENSURE(evalMetricOptions->ObjectiveMetric.IsSet(), "Objective metric must be set.");
    CB_ENSURE(evalMetricOptions->EvalMetric.IsSet(), "Eval metric must be set");
    const NCatboostOptions::TLossDescription& objectiveMetricDescription = evalMetricOptions->ObjectiveMetric.Get();
    const NCatboostOptions::TLossDescription& evalMetricDescription = evalMetricOptions->EvalMetric.Get();

    TVector<THolder<IMetric>> createdObjectiveMetrics;
    if (!IsUserDefined(objectiveMetricDescription.GetLossFunction())) {
        createdObjectiveMetrics = CreateMetricFromDescription(
            objectiveMetricDescription,
            approxDimension);
        if (hasWeights) {
            for (auto& metric : createdObjectiveMetrics) {
                if (!metric->UseWeights.IsIgnored() && !metric->UseWeights.IsUserDefined()) {
                    metric->UseWeights.SetDefaultValue(true);
                }
            }
        }
    }

    TVector<THolder<IMetric>> metrics;
    THashSet<TString> usedDescriptions;
    THashSet<TString> metricsToCalcOnTrain;

    if (IsUserDefined(evalMetricDescription.GetLossFunction())) {
        metrics.emplace_back(MakeCustomMetric(*evalMetricDescriptor));
    } else {
        metrics = CreateMetricFromDescription(evalMetricDescription, approxDimension);
        CB_ENSURE(metrics.size() == 1, "Eval metric should have a single value. Metric " <<
            ToString(evalMetricDescription.GetLossFunction()) <<
            " provides a value for each class, thus it cannot be used as " <<
            "a single value to select best iteration or to detect overfitting. " <<
            "If you just want to look on the values of this metric use custom_metric parameter.");
        if (hasWeights && !metrics.back()->UseWeights.IsIgnored() && ShouldConsiderWeightsByDefault(metrics.back())) {
            metrics.back()->UseWeights.SetDefaultValue(true);
        }
    }
    usedDescriptions.insert(metrics.back()->GetDescription());
    if (HintedToEvalOnTrain(evalMetricDescription)) {
            metricsToCalcOnTrain.insert(metrics.back()->GetDescription());
    }

    for (auto& metric : createdObjectiveMetrics) {
        const auto& description = metric->GetDescription();
        if (!usedDescriptions.contains(description)) {
            usedDescriptions.insert(description);
            metrics.emplace_back(std::move(metric));
        }
    }

    // if custom metric is set without 'use_weights' parameter and we have non-default weights, we calculate both versions of metric.
    for (const auto& description : evalMetricOptions->CustomMetrics.Get()) {
        TVector<THolder<IMetric>> createdCustomMetrics = CreateMetricFromDescription(description, approxDimension);
        if (hasWeights) {
            TVector<THolder<IMetric>> createdCustomMetricsCopy = CreateMetricFromDescription(description, approxDimension);
            auto iter = createdCustomMetricsCopy.begin();
            ui32 initialVectorSize = createdCustomMetrics.size();
            for (ui32 ind = 0; ind < initialVectorSize; ++ind) {
                auto& metric = createdCustomMetrics[ind];
                if (HintedToEvalOnTrain(evalMetricOptions->ObjectiveMetric.Get())) {
                    metricsToCalcOnTrain.insert(metric->GetDescription());
                }
                if (ShouldConsiderWeightsByDefault(metric)) {
                    metric->UseWeights = true;
                    (*iter)->UseWeights = false;
                    createdCustomMetrics.emplace_back(std::move(*iter));
                }
                ++iter;
            }
        }
        for (auto& metric : createdCustomMetrics) {
            if (HintedToEvalOnTrain(description)) {
                metricsToCalcOnTrain.insert(metric->GetDescription());
            }
            const auto& metricDescription = metric->GetDescription();
            if (!usedDescriptions.contains(metricDescription)) {
                usedDescriptions.insert(metricDescription);
                metrics.push_back(std::move(metric));
            }
        }
    }
    if (!hasWeights) {
        for (const auto& metric : metrics) {
            CB_ENSURE(!metric->UseWeights.IsUserDefined(),
                      "If non-default weights for objects are not set, the 'use_weights' parameter must not be specified.");
        }
    }
    SetHintToCalcMetricOnTrain(metricsToCalcOnTrain, &metrics);
    return metrics;
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
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::ILocalExecutor* localExecutor
) {
    if (error.GetErrorType() == EErrorType::PerObjectError) {
        int begin = 0, end = target.size();
        CB_ENSURE(
            end <= approx[0].ysize(),
            "Prediction and label size do not match");
        CB_ENSURE(end > 0, "Not enough data to calculate metric: groupwise metric w/o group id's, or objectwise metric w/o samples");
        return dynamic_cast<const ISingleTargetEval&>(error).Eval(approx, approxDelta, isExpApprox, target, weight, queriesInfo, begin, end, *localExecutor);
    } else {
        CB_ENSURE(
            error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError,
            "Expected querywise or pairwise metric");
        int queryStartIndex = 0, queryEndIndex = queriesInfo.size();
        CB_ENSURE(queryEndIndex > 0, "Not enough data to calculate metric: groupwise metric w/o group id's, or objectwise metric w/o samples");
        return dynamic_cast<const ISingleTargetEval&>(error).Eval(approx, approxDelta, isExpApprox, target, weight, queriesInfo, queryStartIndex, queryEndIndex, *localExecutor);
    }
}


TMetricHolder EvalErrors(
    TConstArrayRef<TConstArrayRef<double>> approx,
    TConstArrayRef<TConstArrayRef<double>> approxDelta,
    bool isExpApprox,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::ILocalExecutor* localExecutor
) {
    if (target.size() == 1 && dynamic_cast<const ISingleTargetEval*>(&error) != nullptr) {
        return EvalErrors(To2DConstArrayRef<double>(approx), To2DConstArrayRef<double>(approxDelta), isExpApprox, target[0], weight, queriesInfo, error, localExecutor);
    } else {
        CB_ENSURE_INTERNAL(dynamic_cast<const IMultiTargetEval*>(&error) != nullptr, "Cannot cast to multi-target error");
        CB_ENSURE_INTERNAL(!isExpApprox, "Exponentiated approxes are not supported for multi-target metrics");
        return dynamic_cast<const IMultiTargetEval&>(error).Eval(To2DConstArrayRef<double>(approx), To2DConstArrayRef<double>(approxDelta), target, weight, /*begin*/0, /*end*/target[0].size(), *localExecutor);
    }
}

void CheckMetrics(const TVector<THolder<IMetric>>& metrics, const ELossFunction modelLoss) {
    CB_ENSURE(!metrics.empty(), "No metrics specified for evaluation");
    for (int i = 0; i < metrics.ysize(); ++i) {
        ELossFunction metric;
        try {
            metric = ParseLossType(metrics[i]->GetDescription());
        } catch (...) {
            metric = ELossFunction::PythonUserDefinedPerObject;
        }
        CheckMetric(metric, modelLoss);
    }
}

void CheckPreprocessedTarget(
    TConstArrayRef<float> target,
    const NCatboostOptions::TLossDescription& lossDesciption,
    bool isNonEmptyAndNonConst,
    bool allowConstLabel
) {
    ELossFunction lossFunction = lossDesciption.GetLossFunction();
    if (isNonEmptyAndNonConst && (lossFunction != ELossFunction::PairLogit) && (lossFunction != ELossFunction::PairLogitPairwise)) {
        auto targetBounds = CalcMinMax(target);
        CB_ENSURE((targetBounds.Min != targetBounds.Max) || allowConstLabel, "All train targets are equal");
    }
    if (EqualToOneOf(lossFunction, ELossFunction::CrossEntropy, ELossFunction::PFound, ELossFunction::ERR)) {
        auto targetBounds = CalcMinMax(target);
        CB_ENSURE(targetBounds.Min >= 0, "Min target less than 0: " + ToString(targetBounds.Min));
        CB_ENSURE(targetBounds.Max <= 1, "Max target greater than 1: " + ToString(targetBounds.Max));
    }

    if (lossFunction == ELossFunction::QuerySoftMax) {
        float minTarget = *MinElement(target.begin(), target.end());
        CB_ENSURE(minTarget >= 0, "Min target less than 0: " + ToString(minTarget));
    }

    if (IsMultiClassOnlyMetric(lossFunction)) {
        CB_ENSURE(AllOf(target, [](float x) { return int(x) == x && x >= 0; }),
                  "metric/loss-function " << lossFunction << " is a Multiclassification metric, "
                  " each target label should be a nonnegative integer");
    }

    if (lossFunction != ELossFunction::MultiRMSEWithMissingValues) {
        for (auto objectIdx : xrange(target.size())){
            CB_ENSURE(!IsNan(target[objectIdx]), "metric/loss-function " << lossFunction << " do not allows nan value on target");
        }
    }
}

static EMetricBestValue GetOptimumType(TStringBuf lossFunction) {
    const auto metric = CreateMetricsFromDescription({TString(lossFunction)}, /*approxDim*/ 1);
    EMetricBestValue valueType;
    float bestValue;
    metric[0]->GetBestValue(&valueType, &bestValue);
    return valueType;
}

bool IsMaxOptimal(TStringBuf lossFunction) {
    return GetOptimumType(lossFunction) == EMetricBestValue::Max;
}

bool IsMinOptimal(TStringBuf lossFunction) {
    return GetOptimumType(lossFunction) == EMetricBestValue::Min;
}

bool IsQuantileLoss(const ELossFunction& loss) {
    return loss == ELossFunction::Quantile || loss == ELossFunction::MultiQuantile || loss == ELossFunction::MAE;
}

namespace NCB {

void AppendTemporaryMetricsVector(TVector<THolder<IMetric>>&& src, TVector<THolder<IMetric>>* dst) {
    std::move(src.begin(), src.end(), std::back_inserter(*dst));
}

} // namespace internal
