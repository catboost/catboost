#include "enum_helpers.h"
#include "loss_description.h"

#include <catboost/libs/logging/logging.h>

#include <util/generic/array_ref.h>
#include <util/generic/is_in.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/generic/flags.h>
#include <util/generic/maybe.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/string/cast.h>

namespace {
    enum class EMetricAttribute : ui32 {
        /* metric type */
        /** classification **/
        IsBinaryClassCompatible        = 1 << 0,
        IsMultiClassCompatible         = 1 << 1,
        /** regression **/
        IsRegression                   = 1 << 2,
        IsMultiRegression              = 1 << 3,
        /** ranking **/
        IsGroupwise                    = 1 << 4,
        IsPairwise                     = 1 << 5,

        /* various */
        IsUserDefined                  = 1 << 6,
        IsCombination                  = 1 << 7
    };

    using EMetricAttributes = TFlags<EMetricAttribute>;
    constexpr inline EMetricAttributes operator|(EMetricAttributes::TEnum l, EMetricAttributes::TEnum r) {
        return EMetricAttributes(l) | r;
    }

    class IMetricInfo {
    public:
        explicit IMetricInfo(ELossFunction loss, ERankingType rankingType, EMetricAttributes flags)
            : Loss(loss), Flags(flags), RankingType(rankingType) {
            CB_ENSURE(HasFlags(EMetricAttribute::IsGroupwise) || HasFlags(EMetricAttribute::IsPairwise),
                      "[" + ToString(loss) + "] metric cannot specify ranking type since it's not ranking");

            CB_ENSURE(HasFlags(EMetricAttribute::IsRegression)
                      || HasFlags(EMetricAttribute::IsBinaryClassCompatible)
                      || HasFlags(EMetricAttribute::IsMultiClassCompatible)
                      || HasFlags(EMetricAttribute::IsGroupwise)
                      || HasFlags(EMetricAttribute::IsPairwise)
                      || HasFlags(EMetricAttribute::IsUserDefined)
                      || HasFlags(EMetricAttribute::IsCombination)
                      || HasFlags(EMetricAttribute::IsMultiRegression),
                      "no type (regression, classification, ranking) for [" + ToString(loss) + "]");
        }

        explicit IMetricInfo(ELossFunction loss, EMetricAttributes flags)
            : Loss(loss), Flags(flags) {
            CB_ENSURE(!(HasFlags(EMetricAttribute::IsGroupwise) || HasFlags(EMetricAttribute::IsPairwise)),
                      "ranking type required for [" + ToString(loss) + "]");

            CB_ENSURE(HasFlags(EMetricAttribute::IsRegression)
                      || HasFlags(EMetricAttribute::IsBinaryClassCompatible)
                      || HasFlags(EMetricAttribute::IsMultiClassCompatible)
                      || HasFlags(EMetricAttribute::IsGroupwise)
                      || HasFlags(EMetricAttribute::IsPairwise)
                      || HasFlags(EMetricAttribute::IsUserDefined)
                      || HasFlags(EMetricAttribute::IsCombination)
                      || HasFlags(EMetricAttribute::IsMultiRegression),
                      "no type (regression, classification, ranking) for [" + ToString(loss) + "]");
        }

        bool HasFlags(EMetricAttributes flags) const {
            return Flags.HasFlags(flags);
        }

        bool MissesFlags(EMetricAttributes flags) const {
            return (~Flags).HasFlags(flags);
        }

        ERankingType GetRankingType() const {
            CB_ENSURE(HasFlags(EMetricAttribute::IsGroupwise) || HasFlags(EMetricAttribute::IsPairwise),
                      "[" + ToString(Loss) + "] metric does not have ranking type since it's not ranking");
            return RankingType.GetRef();
        }

        ELossFunction GetLoss() const {
            return Loss;
        }

        virtual ~IMetricInfo() = default;

    private:
        const ELossFunction Loss;
        const EMetricAttributes Flags;
        const TMaybe<ERankingType> RankingType;
    };
}

#define MakeRegister(name, /*registrees*/...)                           \
    static const TMap<ELossFunction, THolder<IMetricInfo>> name = []() { \
        TMap<ELossFunction, THolder<IMetricInfo>> reg;                  \
        (void) (/*registrees*/__VA_ARGS__);                             \
        return reg;                                                     \
    }();

#define Registree(loss, flags)                                          \
    reg.insert({ELossFunction::loss, [&]() {                            \
        CB_ENSURE(!reg.contains(ELossFunction::loss), "Loss " + ToString(ELossFunction::loss) + " redefined"); \
        class T##loss : public IMetricInfo {                            \
        public:                                                         \
            T##loss() : IMetricInfo(ELossFunction::loss, flags){}       \
        };                                                              \
        return MakeHolder<T##loss>();                                   \
    }()})

#define RankingRegistree(loss, rankingType, flags)                      \
    reg.insert({ELossFunction::loss, [&]() {                            \
        CB_ENSURE(!reg.contains(ELossFunction::loss), "Loss " + ToString(ELossFunction::loss) + " redefined"); \
        class T##loss : public IMetricInfo {                            \
        public:                                                         \
            T##loss() : IMetricInfo(ELossFunction::loss, rankingType, flags){} \
        };                                                              \
        return MakeHolder<T##loss>();                                   \
    }()})

MakeRegister(LossInfos,
    Registree(Logloss,
        EMetricAttribute::IsBinaryClassCompatible
    ),
    Registree(CrossEntropy,
        EMetricAttribute::IsBinaryClassCompatible
    ),
    Registree(CtrFactor,
        EMetricAttribute::IsBinaryClassCompatible
    ),
    Registree(MultiRMSE,
        EMetricAttribute::IsMultiRegression
    ),
    Registree(RMSE,
        EMetricAttribute::IsRegression
    ),
    Registree(Lq,
        EMetricAttribute::IsRegression
    ),
    Registree(MAE,
        EMetricAttribute::IsRegression
    ),
    Registree(Quantile,
        EMetricAttribute::IsRegression
    ),
    Registree(Expectile,
        EMetricAttribute::IsRegression
    ),
    Registree(LogLinQuantile,
        EMetricAttribute::IsRegression
    ),
    Registree(MAPE,
        EMetricAttribute::IsRegression
    ),
    Registree(Poisson,
        EMetricAttribute::IsRegression
    ),
    Registree(MSLE,
        EMetricAttribute::IsRegression
    ),
    Registree(MedianAbsoluteError,
        EMetricAttribute::IsRegression
    ),
    Registree(SMAPE,
        EMetricAttribute::IsRegression
    ),
    Registree(Huber,
        EMetricAttribute::IsRegression
    ),
    Registree(MultiClass,
        EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(MultiClassOneVsAll,
        EMetricAttribute::IsMultiClassCompatible
    ),
    RankingRegistree(PairLogit, ERankingType::CrossEntropy,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
        | EMetricAttribute::IsPairwise
    ),
    RankingRegistree(PairLogitPairwise, ERankingType::CrossEntropy,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
        | EMetricAttribute::IsPairwise
    ),
    RankingRegistree(YetiRank, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(YetiRankPairwise, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(QueryRMSE, ERankingType::AbsoluteValue,
        EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(QuerySoftMax, ERankingType::CrossEntropy,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(QueryCrossEntropy, ERankingType::CrossEntropy,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(StochasticFilter, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(StochasticRank, ERankingType::Order,
        EMetricAttribute::IsGroupwise
    ),
    Registree(PythonUserDefinedPerObject,
        EMetricAttribute::IsUserDefined
    ),
    Registree(PythonUserDefinedMultiRegression,
        EMetricAttribute::IsUserDefined
    ),
    Registree(UserPerObjMetric,
        EMetricAttribute::IsUserDefined
    ),
    Registree(UserQuerywiseMetric,
        EMetricAttribute::IsUserDefined
    ),
    Registree(R2,
        EMetricAttribute::IsRegression
    ),
    Registree(NumErrors,
        EMetricAttribute::IsRegression
    ),
    Registree(FairLoss,
        EMetricAttribute::IsRegression
    ),
    Registree(AUC,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(PRAUC,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(Accuracy,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(BalancedAccuracy,
        EMetricAttribute::IsBinaryClassCompatible
    ),
    Registree(BalancedErrorRate,
        EMetricAttribute::IsBinaryClassCompatible
    ),
    Registree(BrierScore,
        EMetricAttribute::IsBinaryClassCompatible
    ),
    Registree(Precision,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(Recall,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(F1,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(TotalF1,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(MCC,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(ZeroOneLoss,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(HammingLoss,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(HingeLoss,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(Kappa,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(WKappa,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(LogLikelihoodOfPrediction,
        EMetricAttribute::IsBinaryClassCompatible
    ),
    Registree(NormalizedGini,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsMultiClassCompatible
    ),
    Registree(Combination,
        EMetricAttribute::IsCombination
    ),
    RankingRegistree(PairAccuracy, ERankingType::CrossEntropy,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
        | EMetricAttribute::IsPairwise
    ),
    RankingRegistree(AverageGain, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(QueryAverage, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(PFound, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(PrecisionAt, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(RecallAt, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(MAP, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(NDCG, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(DCG, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    RankingRegistree(FilteredDCG, ERankingType::Order,
        EMetricAttribute::IsBinaryClassCompatible
        | EMetricAttribute::IsGroupwise
    ),
    Registree(Tweedie,
        EMetricAttribute::IsRegression
    )
)

const IMetricInfo *GetInfo(ELossFunction loss) {
    CB_ENSURE(LossInfos.contains(loss), "No description for [" + ToString(loss) + "]");
    return LossInfos.at(loss).Get();
}

bool IsPlainOnlyModeLoss(ELossFunction loss) {
    return (
        loss == ELossFunction::YetiRankPairwise ||
        loss == ELossFunction::PairLogitPairwise ||
        loss == ELossFunction::QueryCrossEntropy
    );
}

bool IsPairwiseScoring(ELossFunction loss) {
    return (
        loss == ELossFunction::YetiRankPairwise ||
        loss == ELossFunction::PairLogitPairwise ||
        loss == ELossFunction::QueryCrossEntropy
    );
}

bool IsGpuPlainDocParallelOnlyMode(ELossFunction loss) {
    return (
        loss == ELossFunction::YetiRankPairwise ||
        loss == ELossFunction::PairLogitPairwise ||
        loss == ELossFunction::QueryCrossEntropy ||
        loss == ELossFunction::MultiClass ||
        loss == ELossFunction::MultiClassOneVsAll
    );
}

bool IsYetiRankLossFunction(ELossFunction loss) {
    return (
        loss == ELossFunction::YetiRank ||
        loss == ELossFunction::YetiRankPairwise
    );
}

bool IsPairLogit(ELossFunction loss) {
    return (
        loss == ELossFunction::PairLogit ||
        loss == ELossFunction::PairLogitPairwise
    );
}


bool UsesPairsForCalculation(ELossFunction loss) {
    return IsYetiRankLossFunction(loss) || IsPairLogit(loss);
}

bool ShouldSkipCalcOnTrainByDefault(ELossFunction loss) {
    return (
        loss == ELossFunction::MedianAbsoluteError ||
        loss == ELossFunction::YetiRank ||
        loss == ELossFunction::YetiRankPairwise ||
        loss == ELossFunction::AUC ||
        loss == ELossFunction::PFound ||
        loss == ELossFunction::NDCG ||
        loss == ELossFunction::DCG ||
        loss == ELossFunction::FilteredDCG ||
        loss == ELossFunction::NormalizedGini
    );
}

bool ShouldBinarizeLabel(ELossFunction loss) {
    return loss == ELossFunction::Logloss;
}

bool IsCvStratifiedObjective(ELossFunction loss) {
    return (
        loss == ELossFunction::Logloss ||
        loss == ELossFunction::MultiClass ||
        loss == ELossFunction::MultiClassOneVsAll
    );
}

static const TVector<ELossFunction> RegressionObjectives = {
    ELossFunction::RMSE,
    ELossFunction::MAE,
    ELossFunction::Quantile,
    ELossFunction::LogLinQuantile,
    ELossFunction::Expectile,
    ELossFunction::MAPE,
    ELossFunction::Poisson,
    ELossFunction::Lq,
    ELossFunction::Huber,
    ELossFunction::Tweedie
};

static const TVector<ELossFunction> MultiRegressionObjectives = {
    ELossFunction::MultiRMSE,
    ELossFunction::PythonUserDefinedMultiRegression
};

static const TVector<ELossFunction> ClassificationObjectives = {
    ELossFunction::Logloss,
    ELossFunction::CrossEntropy,
    ELossFunction::MultiClass,
    ELossFunction::MultiClassOneVsAll
};

static const TVector<ELossFunction> RankingObjectives = {
    ELossFunction::PairLogit,
    ELossFunction::PairLogitPairwise,
    ELossFunction::YetiRank,
    ELossFunction::YetiRankPairwise,
    ELossFunction::QueryRMSE,
    ELossFunction::QuerySoftMax,
    ELossFunction::QueryCrossEntropy,
    ELossFunction::StochasticFilter,
    ELossFunction::StochasticRank,
    ELossFunction::UserPerObjMetric,
    ELossFunction::UserQuerywiseMetric,
    ELossFunction::Combination
};

static const TVector<ELossFunction> Objectives = []() {
    TVector<ELossFunction> objectives;
    TVector<const TVector<ELossFunction>*> objectiveLists = {
        &RegressionObjectives,
        &MultiRegressionObjectives,
        &ClassificationObjectives,
        &RankingObjectives
    };
    for (auto objectiveList : objectiveLists) {
        for (auto objective : *objectiveList) {
            objectives.push_back(objective);
        }
    }
    return objectives;
}();

TConstArrayRef<ELossFunction> GetAllObjectives() {
    return Objectives;
}

ERankingType GetRankingType(ELossFunction loss) {
    CB_ENSURE(IsRankingMetric(loss),
              "[" + ToString(loss) + "] metric does not have ranking type since it's not ranking");
    return GetInfo(loss)->GetRankingType();
}

static bool IsFromAucFamily(ELossFunction loss) {
    return loss == ELossFunction::AUC
        || loss == ELossFunction::NormalizedGini;
}

bool IsClassificationOnlyMetric(ELossFunction loss) {
    return IsClassificationMetric(loss) && !IsRegressionMetric(loss)
        && !IsRankingMetric(loss) && !IsFromAucFamily(loss);
}

bool IsBinaryClassCompatibleMetric(ELossFunction loss) {
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsBinaryClassCompatible);
}

bool IsMultiClassCompatibleMetric(ELossFunction loss) {
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsMultiClassCompatible);
}

bool IsMultiClassCompatibleMetric(TStringBuf lossFunction) {
    auto loss = ParseLossType(lossFunction);
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsMultiClassCompatible);
}

bool IsClassificationMetric(ELossFunction loss) {
    auto info = GetInfo(loss);
    return info->HasFlags(EMetricAttribute::IsBinaryClassCompatible)
        || info->HasFlags(EMetricAttribute::IsMultiClassCompatible);
}

bool IsBinaryClassOnlyMetric(ELossFunction loss) {
    auto info = GetInfo(loss);
    return IsClassificationOnlyMetric(loss)
        && info->HasFlags(EMetricAttribute::IsBinaryClassCompatible)
        && info->MissesFlags(EMetricAttribute::IsMultiClassCompatible);
}

bool IsMultiClassOnlyMetric(ELossFunction loss) {
    auto info = GetInfo(loss);
    return IsClassificationOnlyMetric(loss)
        && info->HasFlags(EMetricAttribute::IsMultiClassCompatible)
        && info->MissesFlags(EMetricAttribute::IsBinaryClassCompatible);
}

bool IsClassificationObjective(ELossFunction loss) {
    return IsIn(ClassificationObjectives, loss);
}

bool IsRegressionObjective(ELossFunction loss) {
    return IsIn(RegressionObjectives, loss);
}

bool IsMultiRegressionObjective(ELossFunction loss) {
    return IsIn(MultiRegressionObjectives, loss);
}

bool IsMultiRegressionObjective(TStringBuf loss) {
    return IsMultiRegressionObjective(ParseLossType(loss));
}

bool IsMultiRegressionMetric(ELossFunction loss) {
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsMultiRegression);
}

bool IsRegressionMetric(ELossFunction loss) {
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsRegression);
}

bool IsGroupwiseMetric(ELossFunction loss) {
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsGroupwise);
}

bool IsPairwiseMetric(ELossFunction loss) {
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsPairwise);
}

bool IsRankingMetric(ELossFunction loss) {
    auto info = GetInfo(loss);
    return info->HasFlags(EMetricAttribute::IsPairwise)
        || info->HasFlags(EMetricAttribute::IsGroupwise);
}

bool IsUserDefined(ELossFunction loss) {
    return GetInfo(loss)->HasFlags(EMetricAttribute::IsUserDefined);
}

bool IsClassificationObjective(const TStringBuf lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsClassificationObjective(lossType);
}

bool IsCvStratifiedObjective(const TStringBuf lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsCvStratifiedObjective(lossType);
}

bool IsRegressionObjective(const TStringBuf lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsRegressionObjective(lossType);
}

bool IsGroupwiseMetric(TStringBuf metricName) {
    ELossFunction lossType = ParseLossType(metricName);
    return IsGroupwiseMetric(lossType);
}

bool IsPairwiseMetric(TStringBuf lossFunction) {
    const ELossFunction lossType = ParseLossType(lossFunction);
    return IsPairwiseMetric(lossType);
}

bool IsPlainMode(EBoostingType boostingType) {
    return (boostingType == EBoostingType::Plain);
}

bool IsSecondOrderScoreFunction(EScoreFunction function) {
    switch (function) {
    case EScoreFunction::NewtonL2:
    case EScoreFunction::NewtonCosine: {
        return true;
    }
    case EScoreFunction::Cosine:
    case EScoreFunction::SolarL2:
    case EScoreFunction::LOOL2:
    case EScoreFunction::L2: {
        return false;
    }
    default: {
        ythrow TCatBoostException() << "Unknown score function " << function;
    }
    }
    Y_UNREACHABLE();
}

bool AreZeroWeightsAfterBootstrap(EBootstrapType type) {
    switch (type) {
    case EBootstrapType::Bernoulli:
    case EBootstrapType::Poisson:
        return true;
    default:
        return false;
    }
}

bool IsEmbeddingFeatureEstimator(EFeatureCalcerType estimatorType) {
    return (
        estimatorType == EFeatureCalcerType::CosDistanceWithClassCenter ||
        estimatorType == EFeatureCalcerType::GaussianHomoscedasticModel ||
        estimatorType == EFeatureCalcerType::GaussianHeteroscedasticModel
    );
}

bool IsBuildingFullBinaryTree(EGrowPolicy growPolicy) {
    return (
        growPolicy == EGrowPolicy::SymmetricTree ||
        growPolicy == EGrowPolicy::Depthwise
    );
}

bool IsPlainOnlyModeScoreFunction(EScoreFunction scoreFunction) {
    return (
        scoreFunction != EScoreFunction::Cosine &&
        scoreFunction != EScoreFunction::NewtonCosine
    );
}

EFstrType AdjustFeatureImportanceType(EFstrType type, ELossFunction lossFunction) {
    if (type == EFstrType::InternalInteraction) {
        return EFstrType::Interaction;
    }
    if (type == EFstrType::InternalFeatureImportance || type == EFstrType::FeatureImportance) {
        return IsGroupwiseMetric(lossFunction)
           ? EFstrType::LossFunctionChange
           : EFstrType::PredictionValuesChange;
    }
    return type;
}

EFstrType AdjustFeatureImportanceType(EFstrType type, TStringBuf lossDescription) {
    switch (type) {
        case EFstrType::InternalInteraction: {
            return EFstrType::Interaction;
        }
        case EFstrType::FeatureImportance:
        case EFstrType::InternalFeatureImportance: {
            if (!lossDescription.empty()) {
                return AdjustFeatureImportanceType(type, ParseLossType(lossDescription));
            }
            CATBOOST_WARNING_LOG << "Optimized objective is not known, "
                                    "so use PredictionValuesChange for feature importance." << Endl;
            return EFstrType::PredictionValuesChange;
        }
        default:
            return type;
    }
}

bool IsInternalFeatureImportanceType(EFstrType type) {
    return (
        type == EFstrType::InternalFeatureImportance ||
        type == EFstrType::InternalInteraction
    );
}
