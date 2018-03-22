#pragma once

#include <catboost/libs/algo/calc_score_cache.h>
#include <catboost/libs/algo/fold.h>
#include <catboost/libs/algo/online_predictor.h>
#include <catboost/libs/algo/score_calcer.h>
#include <catboost/libs/algo/target_classifier.h>
#include <catboost/libs/algo/dataset.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/restrictions.h>

#include <library/binsaver/bin_saver.h>
#include <library/par/par.h>
#include <library/par/par_util.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/singleton.h>

#define SHARED_ID_TRAIN_DATA                (0xd66d480)

namespace NCatboostDistributed {
template<typename TData>
struct TEnvelope : public IObjectBase {
    OBJECT_NOCOPY_METHODS(TEnvelope);
public:
    TEnvelope() = default;
    explicit TEnvelope(const TData& data)
    : Data(data)
    {
    }
    TData Data;
    SAVELOAD(Data);
};

using TStats5D = TVector<TVector<TStats3D>>; // [cand][subCand][bodyTail & approxDim][leaf][bucket]
using TStats4D = TVector<TStats3D>; // [subCand][bodyTail & approxDim][leaf][bucket]
using TIsLeafEmpty = TVector<bool>;
using TSums = TVector<TSum>;

struct TTrainData : public IObjectBase {
    OBJECT_NOCOPY_METHODS(TTrainData);
public:
    TTrainData() = default;
    TTrainData(const ::TDataset& trainData,
        const TVector<TTargetClassifier>& targetClassifiers,
        const TVector<int>& splitCounts,
        ui64 randomSeed,
        int approxDimension,
        bool storeExpApprox,
        bool hasPairwiseWeights)
    : TrainData(trainData)
    , TargetClassifiers(targetClassifiers)
    , SplitCounts(splitCounts)
    , RandomSeed(randomSeed)
    , ApproxDimension(approxDimension)
    , StoreExpApprox(storeExpApprox)
    , HasPairwiseWeights(hasPairwiseWeights)
    {
    }
    ::TDataset TrainData;
    TVector<TTargetClassifier> TargetClassifiers;
    TVector<int> SplitCounts;
    ui64 RandomSeed;
    int ApproxDimension;
    bool StoreExpApprox;
    bool HasPairwiseWeights;

    SAVELOAD(TrainData, TargetClassifiers, SplitCounts, RandomSeed, ApproxDimension, StoreExpApprox, HasPairwiseWeights);
};

struct TLocalTensorSearchData {
    Y_DECLARE_SINGLETON_FRIEND();
    // part of TLearnContext used by GreedyTensorSearch
    TCalcScoreFold SampledDocs;
    TCalcScoreFold SmallestSplitSideDocs;
    TBucketStatsCache PrevTreeLevelStats;
    THolder<TRestorableFastRng64> Rand;
    // data used by CalcScore, SetPermutedIndices, CalcApprox, CalcWeightedDerivatives
    TFold PlainFold;
    int Depth;
    TVector<TIndexType> Indices;

    TVector<double> LeafValues;
    TVector<TVector<double>> ApproxDeltas; // 2D because only plain boosting is supported
    TSums Buckets;
    int GradientIteration;

    NCatboostOptions::TCatBoostOptions Params;
    TMaybe<TCustomObjectiveDescriptor> Objective;
    TLocalTensorSearchData()
    : Params(ETaskType::CPU)
    {
    }
    inline static TLocalTensorSearchData& GetRef() {
        return *Singleton<TLocalTensorSearchData>();
    }
};
} // NCatboostDistributed
