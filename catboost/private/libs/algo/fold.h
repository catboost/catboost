#pragma once

#include "online_ctr.h"
#include "projection.h"
#include "target_classifier.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/data_types/query.h>
#include <catboost/libs/helpers/clear_array.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/model/online_ctr.h>
#include <catboost/private/libs/options/binarization_options.h>
#include <catboost/private/libs/options/defaults_helper.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/random/shuffle.h>

#include <tuple>


struct TRestorableFastRng64;

namespace NPar {
    class TLocalExecutor;
}


class TFold {
public:
    struct TBodyTail {
    public:
        TBodyTail() : TBodyTail(0, 0, 0, 0, 0) {
        }

        TBodyTail(
            int bodyQueryFinish,
            int tailQueryFinish,
            int bodyFinish,
            int tailFinish,
            double bodySumWeight)
            : BodyQueryFinish(bodyQueryFinish)
            , TailQueryFinish(tailQueryFinish)
            , BodyFinish(bodyFinish)
            , TailFinish(tailFinish)
            , BodySumWeight(bodySumWeight) {
        }

        int GetBodyDocCount() const { return BodyFinish; }

    public:
        TVector<TVector<double>> Approx;  // [dim][]
        TVector<TVector<double>> WeightedDerivatives;  // [dim][]
        // TODO(annaveronika): make a single vector<vector> for all BodyTail
        TVector<TVector<double>> SampleWeightedDerivatives;  // [dim][]
        TVector<float> PairwiseWeights;  // [dim][]
        TVector<float> SamplePairwiseWeights;  // [dim][]

        const int BodyQueryFinish;
        const int TailQueryFinish;
        const int BodyFinish;
        const int TailFinish;
        const double BodySumWeight;
    };

public:
    TOnlineCTRHash& GetCtrs(const TProjection& proj) {
        return proj.HasSingleFeature() ? OnlineSingleCtrs : OnlineCTR;
    }

    const TOnlineCTRHash& GetCtrs(const TProjection& proj) const {
        return proj.HasSingleFeature() ? OnlineSingleCtrs : OnlineCTR;
    }

    TOnlineCTR& GetCtrRef(const TProjection& proj) {
        return GetCtrs(proj)[proj];
    }

    const TOnlineCTR& GetCtr(const TProjection& proj) const {
        return GetCtrs(proj).at(proj);
    }

    void DropEmptyCTRs();

    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&> GetAllCtrs() const {
        return std::tie(OnlineSingleCtrs, OnlineCTR);
    }

    const NCB::TEstimatedForCPUObjectsDataProviders& GetOnlineEstimatedFeatures() const {
        return OnlineEstimatedFeatures;
    }

    NCB::TEstimatedForCPUObjectsDataProviders& GetOnlineEstimatedFeatures() {
        return OnlineEstimatedFeatures;
    }

    const NCB::TQuantizedEstimatedFeaturesInfo& GetOnlineEstimatedFeaturesInfo() const {
        return OnlineEstimatedFeatures.QuantizedEstimatedFeaturesInfo;
    }

    template <typename T>
    void AssignPermuted(TConstArrayRef<T> source, TVector<T>* dest) const {
        *dest = NCB::GetSubset<T>(source, LearnPermutation->GetObjectsIndexing());
    }

    template <typename T>
    void AssignPermutedIfDefined(NCB::TMaybeData<TConstArrayRef<T>> source, TVector<T>* dest) const {
        if (source) {
            AssignPermuted(*source, dest);
        }
    }

    int GetApproxDimension() const {
        return BodyTailArr[0].Approx.ysize();
    }

    void TrimOnlineCTR(size_t maxOnlineCTRFeatures) {
        if (OnlineCTR.size() > maxOnlineCTRFeatures) {
            OnlineCTR.clear();
        }
    }

    const TVector<float>& GetLearnWeights() const { return LearnWeights; }

    void SaveApproxes(IOutputStream* s) const;
    void LoadApproxes(IInputStream* s);

    static TFold BuildDynamicFold(
        const NCB::TTrainingDataProviders& data,
        const TVector<TTargetClassifier>& targetClassifiers,
        bool shuffle,
        ui32 permuteBlockSize,
        int approxDimension,
        double multiplier,
        bool storeExpApproxes,
        bool hasPairwiseWeights,
        TMaybe<double> startingApprox,
        const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
        NCB::TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo, // can be nullptr
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );

    static TFold BuildPlainFold(
        const NCB::TTrainingDataProviders& data,
        const TVector<TTargetClassifier>& targetClassifiers,
        bool shuffle,
        ui32 permuteBlockSize,
        int approxDimension,
        bool storeExpApproxes,
        bool hasPairwiseWeights,
        TMaybe<double> startingApprox,
        const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
        NCB::TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo, // can be nullptr
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    );

    double GetSumWeight() const { return SumWeight; }
    ui32 GetLearnSampleCount() const { return LearnPermutation->GetSubsetGrouping()->GetObjectCount(); }

    TConstArrayRef<ui32> GetLearnPermutationArray() const {
        return LearnPermutation->GetObjectsIndexing().Get<NCB::TIndexedSubset<ui32>>();
    }

    TConstArrayRef<ui32> GetLearnPermutationOfflineEstimatedFeaturesSubset() const {
        return GetLearnPermutationArray();
    }

private:
    void AssignTarget(
        NCB::TMaybeData<TConstArrayRef<TConstArrayRef<float>>> target,
        const TVector<TTargetClassifier>& targetClassifiers,
        NPar::TLocalExecutor* localExecutor
    );

    void SetWeights(TConstArrayRef<float> weights, ui32 learnSampleCount);

    void InitOnlineEstimatedFeatures(
        const NCatboostOptions::TBinarizationOptions& quantizationOptions,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        const NCB::TTrainingDataProviders& data,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand
    );

public:
    TVector<TQueryInfo> LearnQueriesInfo;

    /*
     * TMaybe is used because of delayed initialization and lack of default constructor in
     * TObjectsGroupingSubset
     */
    TMaybe<NCB::TObjectsGroupingSubset> LearnPermutation; // use for non-features data

    /* indexing in features buckets arrays, always TIndexedSubset
     * initialized to some default value because TArraySubsetIndexing has no default constructor
     */
    NCB::TFeaturesArraySubsetIndexing LearnPermutationFeaturesSubset
        = NCB::TFeaturesArraySubsetIndexing(NCB::TIndexedSubset<ui32>());

    /* begin of subset of data in features buckets arrays, used only for permutation block index calculation
     * if (PermutationBlockSize != 1) && (PermutationBlockSize != learnSampleCount))
     */
    ui32 FeaturesSubsetBegin;

    TVector<TBodyTail> BodyTailArr;
    TVector<TVector<float>> LearnTarget;
    TVector<float> SampleWeights; // Resulting bootstrapped weights of documents.
    TVector<TVector<int>> LearnTargetClass;
    TVector<int> TargetClassesCount;
    ui32 PermutationBlockSize = FoldPermutationBlockSizeNotSet;

private:
    TVector<float> LearnWeights;  // Initial document weights. Empty if no weights present.
    double SumWeight;

    TOnlineCTRHash OnlineSingleCtrs;
    TOnlineCTRHash OnlineCTR;

    NCB::TEstimatedForCPUObjectsDataProviders OnlineEstimatedFeatures;
};

