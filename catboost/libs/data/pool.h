#pragma once

#include "pair.h"

#include <catboost/libs/helpers/exception.h>

#include <util/random/fast.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>
#include <util/ysaveload.h>
#include <util/generic/hash.h>

struct TDocInfo {
    float Target = 0;
    float Weight = 1;
    TVector<float> Factors;
    TVector<double> Baseline;
    TString Id;

    void Swap(TDocInfo& other) {
        DoSwap(Target, other.Target);
        DoSwap(Weight, other.Weight);
        Factors.swap(other.Factors);
        Baseline.swap(other.Baseline);
        DoSwap(Id, other.Id);
    }
};

struct TDocumentStorage {
    TVector<TVector<float>> Factors; // [factorIdx][docIdx]
    TVector<TVector<double>> Baseline; // [dim][docIdx]
    TVector<float> Target; // [docIdx]
    TVector<float> Weight; // [docIdx]
    TVector<TString> Id; // [docIdx]
    TVector<ui32> QueryId; // [docIdx]

    inline int GetBaselineDimension() const {
        return Baseline.ysize();
    }

    inline int GetFactorsCount() const {
        return Factors.ysize();
    }

    inline size_t GetDocCount() const {
        return Target.size();
    }

    inline void Swap(TDocumentStorage& other) {
        Factors.swap(other.Factors);
        Baseline.swap(other.Baseline);
        Target.swap(other.Target);
        Weight.swap(other.Weight);
        Id.swap(other.Id);
        QueryId.swap(other.QueryId);
    }

    inline void SwapDoc(size_t doc1Idx, size_t doc2Idx) {
        for (int factorIdx = 0; factorIdx < GetFactorsCount(); ++factorIdx) {
            DoSwap(Factors[factorIdx][doc1Idx], Factors[factorIdx][doc2Idx]);
        }
        for (int dim = 0; dim < GetBaselineDimension(); ++dim) {
            DoSwap(Baseline[dim][doc1Idx], Baseline[dim][doc2Idx]);
        }
        DoSwap(Target[doc1Idx], Target[doc2Idx]);
        DoSwap(Weight[doc1Idx], Weight[doc2Idx]);
        DoSwap(Id[doc1Idx], Id[doc2Idx]);
        DoSwap(QueryId[doc1Idx], QueryId[doc2Idx]);
    }

    inline void AssignDoc(int destinationIdx, const TDocumentStorage& sourceDocs, int sourceIdx) {
        Y_ASSERT(GetFactorsCount() == sourceDocs.GetFactorsCount());
        Y_ASSERT(GetBaselineDimension() == sourceDocs.GetBaselineDimension());
        for (int factorIdx = 0; factorIdx < GetFactorsCount(); ++factorIdx) {
            Factors[factorIdx][destinationIdx] = sourceDocs.Factors[factorIdx][sourceIdx];
        }
        for (int dim = 0; dim < GetBaselineDimension(); ++dim) {
            Baseline[dim][destinationIdx] = sourceDocs.Baseline[dim][sourceIdx];
        }
        Target[destinationIdx] = sourceDocs.Target[sourceIdx];
        Weight[destinationIdx] = sourceDocs.Weight[sourceIdx];
        Id[destinationIdx] = sourceDocs.Id[sourceIdx];
        QueryId[destinationIdx] = sourceDocs.QueryId[sourceIdx];
    }

    inline void Resize(int docCount, int featureCount, int approxDim) {
        Factors.resize(featureCount);
        for (auto& factor : Factors) {
            factor.resize(docCount);
        }
        Baseline.resize(approxDim);
        for (auto& dim : Baseline) {
            dim.resize(docCount);
        }
        Target.resize(docCount);
        Weight.resize(docCount, 1.0f);
        Id.resize(docCount);
        QueryId.resize(docCount);
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
    }

    inline void Append(const TDocumentStorage& documents) {
        if (!documents.Factors.empty()) {
            Y_ASSERT(GetFactorsCount() == documents.GetFactorsCount());
            for (int factorIdx = 0; factorIdx < GetFactorsCount(); ++factorIdx) {
                Factors[factorIdx].insert(Factors[factorIdx].end(), documents.Factors[factorIdx].begin(), documents.Factors[factorIdx].end());
            }
        }
        if (!documents.Baseline.empty()) {
            Y_ASSERT(GetBaselineDimension() == documents.GetBaselineDimension());
            for (int dim = 0; dim < GetBaselineDimension(); ++dim) {
                Baseline[dim].insert(Baseline[dim].end(), documents.Baseline[dim].begin(), documents.Baseline[dim].end());
            }
        }
        Target.insert(Target.end(), documents.Target.begin(), documents.Target.end());
        Weight.insert(Weight.end(), documents.Weight.begin(), documents.Weight.end());
        Id.insert(Id.end(), documents.Id.begin(), documents.Id.end());
        QueryId.insert(QueryId.end(), documents.QueryId.begin(), documents.QueryId.end());
    }
};

struct TPool {
    mutable TDocumentStorage Docs; // allow freeing Factors[i] and Baseline[i] as Docs are processed by PrepareAllFeatures and PrepareAllFeaturesFromPermutedDocs to reduce memory footprint
    TVector<int> CatFeatures;
    TVector<TString> FeatureId;
    yhash<int, TString> CatFeaturesHashToString;
    TVector<TPair> Pairs;
};
