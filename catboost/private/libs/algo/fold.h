#pragma once

#include "online_ctr.h"
#include "projection.h"
#include "target_classifier.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/data_types/query.h>
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
    class ILocalExecutor;
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
    TOwnedOnlineCtr* GetOwnedCtrs(const TProjection& proj) {
        return (proj.HasSingleFeature() ? OwnedOnlineSingleCtrs : OwnedOnlineCtrs);
    }

    void ClearCtrDataForProjectionIfOwned(const TProjection& proj) {
        auto* ownedCtr = GetOwnedCtrs(proj);
        if (ownedCtr) {
            ownedCtr->Data.at(proj).Feature.clear();
        }
    }

    const TOnlineCtrBase& GetCtrs(const TProjection& proj) const {
        return *(proj.HasSingleFeature() ? OnlineSingleCtrs : OnlineCtrs);
    }


    void DropEmptyCTRs();

    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&> GetAllCtrs() const {
        return std::tie(*OnlineSingleCtrs, *OnlineCtrs);
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
        if (OwnedOnlineCtrs && OwnedOnlineCtrs->Data.size() > maxOnlineCTRFeatures) {
            OwnedOnlineCtrs->Data.clear();
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
        const TMaybe<TVector<double>>& startingApprox,
        const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
        NCB::TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo, // can be nullptr
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* localExecutor
    );

    static TFold BuildPlainFold(
        const NCB::TTrainingDataProviders& data,
        const TVector<TTargetClassifier>& targetClassifiers,
        bool shuffle,
        ui32 permuteBlockSize,
        int approxDimension,
        bool storeExpApproxes,
        bool hasPairwiseWeights,
        const TMaybe<TVector<double>>& startingApprox,
        const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
        NCB::TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo, // can be nullptr
        TIntrusivePtr<TPrecomputedOnlineCtr> precomputedSingleOnlineCtrs, // can be empty
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* localExecutor
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
        NPar::ILocalExecutor* localExecutor
    );

    void SetWeights(TConstArrayRef<float> weights, ui32 learnSampleCount);

    void InitOnlineEstimatedFeatures(
        const NCatboostOptions::TBinarizationOptions& quantizationOptions,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        const NCB::TTrainingDataProviders& data,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand
    );

    void InitOnlineCtrs(
        const NCB::TTrainingDataProviders& data,
        TIntrusivePtr<TPrecomputedOnlineCtr> precomputedSingleOnlineCtrs = nullptr
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

    TIntrusivePtr<TOnlineCtrBase> OnlineSingleCtrs;
    TIntrusivePtr<TOnlineCtrBase> OnlineCtrs;

    // point to OnlineSingleCtrs & OnlineCtrs if they are of type TOwnedOnlineCtrStorage, null otherwise
    TOwnedOnlineCtr* OwnedOnlineSingleCtrs = nullptr;
    TOwnedOnlineCtr* OwnedOnlineCtrs = nullptr;

    NCB::TEstimatedForCPUObjectsDataProviders OnlineEstimatedFeatures;
};

