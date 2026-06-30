#pragma once

#include <catboost/private/libs/algo/calc_score_cache.h>
#include <catboost/private/libs/algo/fold.h>
#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo/online_ctr.h>
#include <catboost/private/libs/algo/pairwise_scoring.h>
#include <catboost/private/libs/algo/score_calcers.h>
#include <catboost/private/libs/algo/target_classifier.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/load_options.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/containers/2d_array/2d_array.h>
#include <library/cpp/json/json_value.h>
#include <library/cpp/par/par.h>
#include <library/cpp/par/par_util.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/singleton.h>

#define SHARED_ID_TRAIN_DATA                (0xd66d480)


namespace NCatboostDistributed {

    struct TUnusedInitializedParam {
        char Zero = 0;
    };

    using TStats5D = TVector<TVector<TStats3D>>; // [cand][subCand][bodyTail & approxDim][leaf][bucket]
    using TStats4D = TVector<TStats3D>; // [subCand][bodyTail & approxDim][leaf][bucket]
    using TIsLeafEmpty = TVector<bool>;
    using TSums = TVector<TSum>;
    using TMultiSums = TVector<TSumMulti>;

    using TWorkerPairwiseStats = TVector<TVector<TPairwiseStats>>; // [cand][subCand]

    struct TTrainData : public IObjectBase {
        NCB::TTrainingDataProviders TrainData;

    public:
        TTrainData() = default;
        TTrainData(NCB::TTrainingDataProviders trainData)
        : TrainData(std::move(trainData))
        {
        }

        SAVELOAD_OVERRIDE_WITHOUT_BASE(TrainData);

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
        NCB::TObjectsGrouping LearnObjectsGrouping;
        TVector<NCB::TObjectsGrouping> TestObjectsGroupings;
        NCB::TFeaturesLayout FeaturesLayout;
        TLabelConverter LabelConverter;
        ui64 RandomSeed;

    public:
        SAVELOAD(
            PoolLoadOptions,
            TrainOptions,
            ObjectsOrder,
            LearnObjectsGrouping,
            TestObjectsGroupings,
            FeaturesLayout,
            LabelConverter,
            RandomSeed);
    };

    struct TApproxReconstructorParams {
        TMaybe<int> BestIteration;
        TVector<std::variant<TSplitTree, TNonSymmetricTreeStructure>> TreeStruct;
        TVector<TVector<TVector<double>>> LeafValues;

    public:
        SAVELOAD(BestIteration, TreeStruct, LeafValues);
    };

    struct TApproxGetterParams {
        bool ReturnLearnApprox;
        bool ReturnTestApprox;
        bool ReturnBestTestApprox;

    public:
        SAVELOAD(ReturnLearnApprox, ReturnTestApprox, ReturnBestTestApprox);
    };

    struct TApproxesResult {
        TVector<TVector<double>> LearnApprox;         //       [dim][docIdx]
        TVector<TVector<TVector<double>>> TestApprox; // [test][dim][docIdx]
        TVector<TVector<double>> BestTestApprox;      //       [dim][docIdx]

    public:
        SAVELOAD(LearnApprox, TestApprox, BestTestApprox);
    };

    struct TErrorCalcerParams {
        bool CalcOnlyBacktrackingObjective;
        bool CalcAllMetrics;
        bool CalcErrorTrackerMetric;

    public:
        SAVELOAD(CalcOnlyBacktrackingObjective, CalcAllMetrics, CalcErrorTrackerMetric);
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

        // Starting point for gradient walker
        TVector<TVector<double>> BacktrackingStart;

        // data used by Exact approx calcer
        TVector<TVector<TVector<std::pair<double, double>>>> ExactDiff; // [dim][leaf][]
        TVector<TVector<TMinMax<int>>> SplitBounds; // [dim][leaf]
        TVector<TVector<double>> LastPivot; // [dim][leaf]
        TVector<TVector<int>> LastPartitionPoint; // [dim][leaf]
        TVector<TVector<double>> CollectedLeftSumWeight; // [dim][leaf]
        TVector<TVector<double>> LastSplitLeftSumWeight; // [dim][leaf]

        ui32 AllDocCount;
        double SumAllWeights;
        EHessianType HessianType = EHessianType::Symmetric;

        NCatboostOptions::TCatBoostOptions Params;

        NCB::TTrainingDataProviders TrainData;
        TMaybe<NCB::TPrecomputedOnlineCtrData> PrecomputedSingleOnlineCtrDataForSingleFold;

        TVector<NJson::TJsonValue> ClassLabelsFromDataset;

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
