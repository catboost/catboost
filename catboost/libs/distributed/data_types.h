#pragma once

#include <catboost/libs/algo/calc_score_cache.h>
#include <catboost/libs/algo/fold.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/pairwise_scoring.h>
#include <catboost/libs/algo/score_calcers.h>
#include <catboost/libs/algo/target_classifier.h>
#include <catboost/libs/algo_helpers/online_predictor.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/options/restrictions.h>

#include <library/binsaver/bin_saver.h>
#include <library/par/par.h>
#include <library/par/par_util.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/singleton.h>

#define SHARED_ID_TRAIN_DATA                (0xd66d480)


namespace NCatboostDistributed {

    struct TUnusedInitializedParam {
        char Zero = 0;
    };

    template <typename TData>
    struct TEnvelope : public IObjectBase {
        TData Data;

    public:
        TEnvelope() = default;
        explicit TEnvelope(const TData& data)
            : Data(data)
        {
        }

        SAVELOAD(Data);
        OBJECT_NOCOPY_METHODS(TEnvelope);
    };

    template <typename TData>
    TEnvelope<TData> MakeEnvelope(const TData& data) {
        return TEnvelope<TData>(data);
    }

    using TStats5D = TVector<TVector<TStats3D>>; // [cand][subCand][bodyTail & approxDim][leaf][bucket]
    using TStats4D = TVector<TStats3D>; // [subCand][bodyTail & approxDim][leaf][bucket]
    using TIsLeafEmpty = TVector<bool>;
    using TSums = TVector<TSum>;
    using TMultiSums = TVector<TSumMulti>;

    using TWorkerPairwiseStats = TVector<TVector<TPairwiseStats>>; // [cand][subCand]

    struct TTrainData : public IObjectBase {
        NCB::TTrainingForCPUDataProviderPtr TrainData;

    public:
        TTrainData() = default;
        TTrainData(NCB::TTrainingForCPUDataProviderPtr trainData)
        : TrainData(trainData)
        {
        }

        int operator&(IBinSaver& binSaver) {
            NCB::AddWithShared(&binSaver, &TrainData);
            return 0;
        }

        OBJECT_NOCOPY_METHODS(TTrainData);
    };

    struct TPlainFoldBuilderParams {
        TVector<TTargetClassifier> TargetClassifiers;
        ui64 RandomSeed;
        int ApproxDimension;
        TString TrainParams;
        ui32 AllDocCount;
        double SumAllWeights;
        EHessianType HessianType;

    public:
        TPlainFoldBuilderParams() = default;

        SAVELOAD(
            TargetClassifiers,
            RandomSeed,
            ApproxDimension,
            TrainParams,
            AllDocCount,
            SumAllWeights,
            HessianType);
    };

    struct TDatasetLoaderParams {
        NCatboostOptions::TPoolLoadParams PoolLoadOptions;
        TString TrainOptions;
        NCB::EObjectsOrder ObjectsOrder;
        NCB::TObjectsGrouping ObjectsGrouping;
        NCB::TFeaturesLayout FeaturesLayout;
        ui64 RandomSeed;

    public:
        TDatasetLoaderParams() = default;

        SAVELOAD(
            PoolLoadOptions,
            TrainOptions,
            ObjectsOrder,
            ObjectsGrouping,
            FeaturesLayout,
            RandomSeed);
    };

    struct TLocalTensorSearchData {
        // part of TLearnContext used by GreedyTensorSearch
        TCalcScoreFold SampledDocs;
        TCalcScoreFold SmallestSplitSideDocs;
        TBucketStatsCache PrevTreeLevelStats;
        THolder<TRestorableFastRng64> Rand;

        // data used by CalcScore, SetPermutedIndices, CalcApprox, CalcWeightedDerivatives
        THolder<TLearnProgress> Progress;
        int Depth;
        TVector<TIndexType> Indices;

        bool StoreExpApprox;
        bool UseTreeLevelCaching;
        TVector<TVector<double>> ApproxDeltas; // 2D because only plain boosting is supported
        TSums Buckets;
        TMultiSums MultiBuckets;
        TArray2D<double> PairwiseBuckets;
        int GradientIteration;

        ui32 AllDocCount;
        double SumAllWeights;
        EHessianType HessianType = EHessianType::Symmetric;

        NCatboostOptions::TCatBoostOptions Params;

        NCB::TTrainingForCPUDataProviderPtr TrainData;
        TVector<TString> ClassNamesFromDataset;

        TFlatPairsInfo FlatPairs;

    public:
        TLocalTensorSearchData()
            : Params(ETaskType::CPU)
        {
        }
        Y_DECLARE_SINGLETON_FRIEND();

        inline static TLocalTensorSearchData& GetRef() {
            return *Singleton<TLocalTensorSearchData>();
        }
    };

} // NCatboostDistributed
