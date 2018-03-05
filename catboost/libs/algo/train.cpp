#include "train.h"

class TLoglossError;
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

TErrorTracker BuildErrorTracker(EMetricBestValue bestValueType, float bestPossibleValue, bool hasTest, TLearnContext* ctx) {
    const auto& odOptions = ctx->Params.BoostingOptions->OverfittingDetector;
    return TErrorTracker(odOptions->OverfittingDetectorType,
                         bestValueType,
                         bestPossibleValue,
                         odOptions->AutoStopPValue,
                         odOptions->IterationsWait,
                         true,
                         hasTest);
}

template <typename TError>
void TrainOneIter(const TTrainData& learnData, const TTrainData* testData, TLearnContext* ctx);

TTrainOneIterationFunc GetOneIterationFunc(ELossFunction lossFunction) {
    switch (lossFunction) {
        case ELossFunction::Logloss:
            return TrainOneIter<TLoglossError>;
            break;
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
        case ELossFunction::QueryRMSE:
            return TrainOneIter<TQueryRmseError>;
            break;
        case ELossFunction::QuerySoftMax:
            return TrainOneIter<TQuerySoftMaxError>;
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

template <typename T>
TVector<T> Concat(const TVector<T>& a, const TVector<T>& b) {
    static_assert(std::is_pod<T>::value, "T must be a pod");
    TVector<T> res;
    res.insert(res.end(), a.begin(), a.end());
    res.insert(res.end(), b.begin(), b.end());
    return res;
}

template TVector<float> Concat<float>(const TVector<float>& a, const TVector<float>& b);

template <>
TVector<TQueryInfo> Concat(const TVector<TQueryInfo>& a, const TVector<TQueryInfo>& b) {
    TVector<TQueryInfo> res;
    res.insert(res.end(), a.begin(), a.end());
    res.insert(res.end(), b.begin(), b.end());
    return res;
}

template <>
TVector<TPair> Concat(const TVector<TPair>& a, const TVector<TPair>& b) {
    TVector<TPair> res;
    res.insert(res.end(), a.begin(), a.end());
    res.insert(res.end(), b.begin(), b.end());
    return res;
}

template <typename T>
TVector<TVector<T>> Concat(const TVector<TVector<T>>& a, const TVector<TVector<T>>& b) {
    static_assert(std::is_pod<T>::value, "T must be a pod");
    TVector<TVector<T>> res;
    if (b.size()) {
        Y_VERIFY(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            res.push_back(Concat(a[i], b[i]));
        }
    } else {
        return a;
    }
    return res;
}

template <>
TAllFeatures Concat<TAllFeatures>(const TAllFeatures& a, const TAllFeatures& b) {
    TAllFeatures res;
    res.FloatHistograms = Concat(a.FloatHistograms, b.FloatHistograms);
    res.CatFeaturesRemapped = Concat(a.CatFeaturesRemapped, b.CatFeaturesRemapped);
    Y_VERIFY(a.OneHotValues.size() == b.OneHotValues.size() || b.OneHotValues.empty());
    res.OneHotValues = a.OneHotValues;
    if (!b.OneHotValues.empty()) {
        for (size_t c = 0; c < a.OneHotValues.size(); ++c) {
            for (size_t i = 0; i < std::min(a.OneHotValues[c].size(), b.OneHotValues[c].size()); ++i) {
                Y_VERIFY(a.OneHotValues[c][i] == b.OneHotValues[c][i]);
            }
            res.OneHotValues[c].insert(res.OneHotValues[c].end(), b.OneHotValues[c].begin() + a.OneHotValues[c].size(), b.OneHotValues[c].end());
        }
    }
    Y_VERIFY(a.IsOneHot == b.IsOneHot || b.IsOneHot.empty());
    res.IsOneHot = a.IsOneHot;
    return res;
}

template <>
TTrainData Concat<TTrainData>(const TTrainData& a, const TTrainData& b) {
    TTrainData res;
    res.LearnSampleCount = a.Target.ysize();
    res.LearnQueryCount = a.QueryInfo.ysize();
    res.LearnPairsCount = a.Pairs.ysize();
    res.AllFeatures = Concat(a.AllFeatures, b.AllFeatures);
    res.Baseline = Concat(a.Baseline, b.Baseline);
    res.Target = Concat(a.Target, b.Target);
    res.Weights = Concat(a.Weights, b.Weights);
    res.QueryId = Concat(a.QueryId, b.QueryId);
    res.QueryInfo = Concat(a.QueryInfo, b.QueryInfo);
    for (size_t i = res.LearnQueryCount; i < res.QueryInfo.size(); ++i) {
        res.QueryInfo[i].Begin += res.LearnSampleCount;
        res.QueryInfo[i].End += res.LearnSampleCount;
    }
    res.Pairs = Concat(a.Pairs, b.Pairs);
    for (size_t i = res.LearnPairsCount; i < res.Pairs.size(); ++i) {
        res.Pairs[i].WinnerId += res.LearnSampleCount;
        res.Pairs[i].LoserId += res.LearnSampleCount;
    }
    return res;
}
