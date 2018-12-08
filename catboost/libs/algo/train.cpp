#include "train.h"
#include <catboost/libs/helpers/query_info_helper.h>


using namespace NCB;


class TCrossEntropyError;
class TRMSEError;
class TQuantileError;
class TLogLinQuantileError;
class TMAPError;
class TPoissonError;
class TMultiClassError;
class TMultiClassOneVsAllError;
class TPairLogitError;
class TQueryRmseError;
class TQuerySoftMaxError;
class TCustomError;
class TUserDefinedPerObjectError;
class TUserDefinedQuerywiseError;
class TLqError;

TErrorTracker BuildErrorTracker(EMetricBestValue bestValueType, double bestPossibleValue, bool hasTest, TLearnContext* ctx) {
    const auto& odOptions = ctx->Params.BoostingOptions->OverfittingDetector;
    return CreateErrorTracker(odOptions, bestPossibleValue, bestValueType, hasTest);
}

template <typename TError>
void TrainOneIter(const TTrainingForCPUDataProviders& data, TLearnContext* ctx);

TTrainOneIterationFunc GetOneIterationFunc(ELossFunction lossFunction) {
    switch (lossFunction) {
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
            return TrainOneIter<TCrossEntropyError>;
            break;
        case ELossFunction::RMSE:
            return TrainOneIter<TRMSEError>;
            break;
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            return TrainOneIter<TQuantileError>;
            break;
        case ELossFunction::LogLinQuantile:
            return TrainOneIter<TLogLinQuantileError>;
            break;
        case ELossFunction::MAPE:
            return TrainOneIter<TMAPError>;
            break;
        case ELossFunction::Poisson:
            return TrainOneIter<TPoissonError>;
            break;
        case ELossFunction::MultiClass:
            return TrainOneIter<TMultiClassError>;
            break;
        case ELossFunction::MultiClassOneVsAll:
            return TrainOneIter<TMultiClassOneVsAllError>;
            break;
        case ELossFunction::PairLogit:
            return TrainOneIter<TPairLogitError>;
            break;
        case ELossFunction::PairLogitPairwise:
            return TrainOneIter<TPairLogitError>;
            break;
        case ELossFunction::QueryRMSE:
            return TrainOneIter<TQueryRmseError>;
            break;
        case ELossFunction::QuerySoftMax:
            return TrainOneIter<TQuerySoftMaxError>;
            break;
        case ELossFunction::YetiRank:
            return TrainOneIter<TPairLogitError>;
            break;
        case ELossFunction::YetiRankPairwise:
            return TrainOneIter<TPairLogitError>;
            break;
        case ELossFunction::Lq:
            return TrainOneIter<TLqError>;
            break;
        case ELossFunction::Custom:
            return TrainOneIter<TCustomError>;
            break;
        case ELossFunction::UserPerObjMetric:
            return TrainOneIter<TUserDefinedPerObjectError>;
            break;
        case ELossFunction::UserQuerywiseMetric:
            return TrainOneIter<TUserDefinedQuerywiseError>;
            break;
        default:
            CB_ENSURE(false, "provided error function is not supported");
    }
}
