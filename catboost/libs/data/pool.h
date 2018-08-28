#pragma once

#include "quantized_features.h"

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/pool_builder/pool_builder.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/string/cast.h>
#include <util/random/fast.h>
#include <util/generic/algorithm.h>
#include <util/generic/is_in.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>
#include <util/ysaveload.h>
#include <util/generic/hash.h>


struct TDocumentStorage {
    TVector<TVector<float>> Factors; // [factorIdx][docIdx]
    TVector<TVector<double>> Baseline; // [dim][docIdx]
    TVector<TString> Label; // [docIdx] used only as buffer for processing labels at the end of pool reading with converting target policy MakeClassNames
    TVector<float> Target; // [docIdx] stores processed numeric target
    TVector<float> Weight; // [docIdx]
    TVector<TString> Id; // [docIdx]
    TVector<TGroupId> QueryId; // [docIdx]
    TVector<TSubgroupId> SubgroupId; // [docIdx]
    TVector<ui64> Timestamp; // [docIdx]

    inline int GetBaselineDimension() const {
        return Baseline.ysize();
    }

    /// @return the number of non-constant factors (`Factors` stores only non-constant factors).
    inline int GetEffectiveFactorCount() const {
        return Factors.ysize();
    }

    inline size_t GetDocCount() const {
        return Target.size();
    }

    bool operator==(const TDocumentStorage& other) const {
        if (Factors.ysize() != other.Factors.ysize()) {
            return false;
        }
        if (Factors.ysize() > 0 && Factors[0].ysize() != other.Factors[0].ysize()) {
            return false;
        }
        bool areFactorsEqual = true;
        for (size_t i = 0; i < Factors.size(); ++i) {
            for (size_t j = 0; j < Factors[i].size(); ++j) {
                areFactorsEqual &= (ConvertFloatCatFeatureToIntHash(Factors[i][j]) == ConvertFloatCatFeatureToIntHash(other.Factors[i][j]));
            }
        }
        return areFactorsEqual && (
            std::tie(Baseline, Target, Weight, Id, QueryId, SubgroupId, Timestamp) ==
            std::tie(other.Baseline, other.Target, other.Weight, other.Id, other.QueryId, other.SubgroupId, other.Timestamp)
        );
    }

    bool operator!=(const TDocumentStorage& other) const {
        return !(*this == other);
    }

    inline void Swap(TDocumentStorage& other) {
        Factors.swap(other.Factors);
        Baseline.swap(other.Baseline);
        Target.swap(other.Target);
        Weight.swap(other.Weight);
        Id.swap(other.Id);
        QueryId.swap(other.QueryId);
        SubgroupId.swap(other.SubgroupId);
        Timestamp.swap(other.Timestamp);
    }

    inline void AssignDoc(int destinationIdx, const TDocumentStorage& sourceDocs, int sourceIdx) {
        Y_ASSERT(GetEffectiveFactorCount() == sourceDocs.GetEffectiveFactorCount());
        Y_ASSERT(GetBaselineDimension() == sourceDocs.GetBaselineDimension());
        for (int factorIdx = 0; factorIdx < GetEffectiveFactorCount(); ++factorIdx) {
            Factors[factorIdx][destinationIdx] = sourceDocs.Factors[factorIdx][sourceIdx];
        }
        for (int dim = 0; dim < GetBaselineDimension(); ++dim) {
            Baseline[dim][destinationIdx] = sourceDocs.Baseline[dim][sourceIdx];
        }
        Target[destinationIdx] = sourceDocs.Target[sourceIdx];
        Weight[destinationIdx] = sourceDocs.Weight[sourceIdx];
        Id[destinationIdx] = sourceDocs.Id[sourceIdx];
        if (!sourceDocs.QueryId.empty()) {
            QueryId[destinationIdx] = sourceDocs.QueryId[sourceIdx];
        }
        if (!sourceDocs.SubgroupId.empty()) {
            SubgroupId[destinationIdx] = sourceDocs.SubgroupId[sourceIdx];
        }
        Timestamp[destinationIdx] = sourceDocs.Timestamp[sourceIdx];
    }

    inline void Resize(int docCount, int featureCount, int approxDim = 0, bool hasQueryId = false, bool hasSubgroupId = false) {
        Factors.resize(featureCount);
        for (auto& factor : Factors) {
            factor.resize(docCount);
        }
        Baseline.resize(approxDim);
        for (auto& dim : Baseline) {
            dim.resize(docCount);
        }
        Target.resize(docCount);
        Label.resize(docCount);
        Weight.resize(docCount, 1.0f);
        Id.resize(docCount);
        for (int ind = 0; ind < docCount; ++ ind) {
            Id[ind] = ToString(ind);
        }
        if (hasQueryId) {
            QueryId.resize(docCount);
        }
        if (hasSubgroupId) {
            SubgroupId.resize(docCount);
        }
        Timestamp.resize(docCount);
    }

    inline void Clear() {
        for (auto& factor : Factors) {
            factor.clear();
            factor.shrink_to_fit();
        }
        for (auto& dim : Baseline) {
            dim.clear();
            dim.shrink_to_fit();
        }
        Target.clear();
        Target.shrink_to_fit();
        Weight.clear();
        Weight.shrink_to_fit();
        Id.clear();
        Id.shrink_to_fit();
        QueryId.clear();
        QueryId.shrink_to_fit();
        SubgroupId.clear();
        SubgroupId.shrink_to_fit();
        Timestamp.clear();
        Timestamp.shrink_to_fit();
    }
};

struct TPool {
    mutable TDocumentStorage Docs; // allow freeing Factors[i] and Baseline[i] as Docs are binarized, to reduce memory footprint
    mutable TAllFeatures QuantizedFeatures; // TODO(akhropov): Temporary solution until MLTOOLS-140 is implemented
    TVector<TFloatFeature> FloatFeatures;
    TVector<int> CatFeatures;
    TVector<TString> FeatureId;
    THashMap<int, TString> CatFeaturesHashToString;
    TVector<TPair> Pairs;
    TPoolMetaInfo MetaInfo;

    int GetFactorCount() const {
        Y_ASSERT(Docs.GetEffectiveFactorCount() == 0 || QuantizedFeatures.FloatHistograms.ysize() + QuantizedFeatures.CatFeaturesRemapped.ysize() == 0);
        return Docs.GetEffectiveFactorCount() + QuantizedFeatures.FloatHistograms.ysize() + QuantizedFeatures.CatFeaturesRemapped.ysize();
    }

    bool IsQuantized() const {
        return !QuantizedFeatures.FloatHistograms.empty() || !QuantizedFeatures.CatFeaturesRemapped.empty();
    }

    void Swap(TPool& other) {
        Docs.Swap(other.Docs);
        CatFeatures.swap(other.CatFeatures);
        FeatureId.swap(other.FeatureId);
        CatFeaturesHashToString.swap(other.CatFeaturesHashToString);
        Pairs.swap(other.Pairs);
        MetaInfo.Swap(other.MetaInfo);
    }

    bool operator==(const TPool& other) const {
        return (
            std::tie(Docs, CatFeatures, FeatureId, CatFeaturesHashToString, Pairs) ==
            std::tie(other.Docs, other.CatFeatures, other.FeatureId, other.CatFeaturesHashToString, other.Pairs)
        );
    }

    void SetCatFeatureHashWithBackMapUpdate(size_t factorIdx, size_t docIdx, TStringBuf catFeatureString) {
        Y_ASSERT(IsIn(CatFeatures, factorIdx));

        int hashVal = CalcCatFeatureHash(catFeatureString);
        Docs.Factors[factorIdx][docIdx] = ConvertCatFeatureHashToFloat(hashVal);

        THashMap<int, TString>::insert_ctx insertCtx;
        if (!CatFeaturesHashToString.has(hashVal, insertCtx)) {
            CatFeaturesHashToString.emplace_direct(insertCtx, hashVal, catFeatureString);
        }
    }

    bool IsTrivialWeights() const {
        return !MetaInfo.HasWeights || AllOf(Docs.Weight, [](float weight) { return weight == 1.0f; });
    }
};

struct TTrainPools {
    TPool Learn;
    TVector<TPool> Test;
};

// const and ptrs because compatibility with current code needed
struct TClearablePoolPtrs {
    // cannot use a reference here because it will make it hard to use in Cython
    TPool* Learn = nullptr;
    bool AllowClearLearn = false;

    /* TODO(akhropov): should not be const because can be possibly cleared,
       Allowed by 'mutable TDocumentStorage' that has to be refactored
    */
    TVector<const TPool*> Test;
    bool AllowClearTest = false;

public:
    // needed for Cython
    TClearablePoolPtrs() = default;

    TClearablePoolPtrs(
        TPool& learn,
        const TVector<const TPool*>& test,
        bool allowClearLearn = false,
        bool allowClearTest = false
    )
        : Learn(&learn)
        , AllowClearLearn(allowClearLearn)
        , Test(test)
        , AllowClearTest(allowClearTest)
    {}

    TClearablePoolPtrs(
        TTrainPools& trainPools,
        bool allowClearLearn = false,
        bool allowClearTest = false
    )
        : Learn(&trainPools.Learn)
        , AllowClearLearn(allowClearLearn)
        , AllowClearTest(allowClearTest)
    {
        for (const auto& testPool : trainPools.Test) {
            Test.push_back(&testPool);
        }
    }
};


THolder<TPool> SlicePool(const TPool& pool, const TVector<size_t>& rowIndices);

inline int GetDocCount(const TVector<const TPool*>& testPoolPtrs) {
    int result = 0;
    for (const TPool* testPool : testPoolPtrs) {
        result += testPool->Docs.GetDocCount();
    }
    return result;
}

void ApplyPermutation(const TVector<ui64>& permutation, TPool* pool, NPar::TLocalExecutor* localExecutor);
void ApplyPermutationToPairs(const TVector<ui64>& permutation, TVector<TPair>* pairs);
