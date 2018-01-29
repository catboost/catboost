#pragma once

#include "hash.h"
#include "online_ctr.h"
#include "features.h"
#include "ctr_value_table.h"
#include <util/generic/array_ref.h>


class ICtrProvider : public TThrRefBase {
public:
    virtual ~ICtrProvider() {
    }

    virtual bool HasNeededCtrs(const TVector<TModelCtr>& neededCtrs) const = 0;

    virtual void CalcCtrs(
        const TVector<TModelCtr>& neededCtrs,
        const TConstArrayRef<ui8>& binarizedFeatures, // vector of binarized float & one hot features
        const TConstArrayRef<int>& hashedCatFeatures,
        size_t docCount,
        TArrayRef<float> result) = 0;

    virtual void SetupBinFeatureIndexes(
        const TVector<TFloatFeature>& floatFeatures,
        const TVector<TOneHotFeature>& oheFeatures,
        const TVector<TCatFeature>& catFeatures) = 0;

    virtual void AddCtrCalcerData(TCtrValueTable&& valueTable) = 0;
    virtual bool IsSerializable() const {
        return false;
    }

    virtual void Save(IOutputStream* ) const {
        throw yexception() << "Serialization not allowed";
    };

    virtual void Load(IInputStream* ) {
        throw yexception() << "Deserialization not allowed";
    };

    // can use this later for complex model deserialization logic
    virtual TString ModelPartIdentifier() const = 0;
};

// slow reference realization
inline ui64 CalcHash(
    const TConstArrayRef<ui8>& binarizedFeatures,
    const TConstArrayRef<int>& hashedCatFeatures,
    const TConstArrayRef<int>& transposedCatFeatureIndexes,
    const TConstArrayRef<int>& binarizedFeatureIndexes) {
    ui64 result = 0;
    for (const int featureIdx : transposedCatFeatureIndexes) {
        result = CalcHash(result, (ui64)hashedCatFeatures[featureIdx]);
    }
    for (const auto& index : binarizedFeatureIndexes) {
        result = CalcHash(result, (ui64)binarizedFeatures[index]);
    }
    return result;
}

struct TBinFeatureIndexValue {
    ui32 BinIndex = 0;
    bool CheckValueEqual = 0;
    ui8 Value = 0;
    TBinFeatureIndexValue() = default;
    TBinFeatureIndexValue(ui32 binIndex, bool checkValueEqual, ui8 value)
        : BinIndex(binIndex)
        , CheckValueEqual(checkValueEqual)
        , Value(value)
    {}
};

inline void CalcHashes(
    const TConstArrayRef<ui8>& binarizedFeatures,
    const TConstArrayRef<int>& hashedCatFeatures,
    const TConstArrayRef<int>& transposedCatFeatureIndexes,
    const TConstArrayRef<TBinFeatureIndexValue>& binarizedFeatureIndexes,
    size_t docCount,
    TVector<ui64>* result) {
    result->resize(docCount);
    std::fill(result->begin(), result->end(), 0);
    ui64* ptr = result->data();
    for (const int featureIdx : transposedCatFeatureIndexes) {
        auto valPtr = &hashedCatFeatures[featureIdx * docCount];
        for (size_t i = 0; i < docCount; ++i) {
            ptr[i] = CalcHash(ptr[i], (ui64)valPtr[i]);
        }
    }
    for (const auto& binFeatureIndex : binarizedFeatureIndexes) {
        const ui8* binFPtr = &binarizedFeatures[binFeatureIndex.BinIndex * docCount];
        if (!binFeatureIndex.CheckValueEqual) {
            for (size_t i = 0; i < docCount; ++i) {
                ptr[i] = CalcHash(ptr[i], (ui64)(binFPtr[i] >= binFeatureIndex.Value));
            }
        } else {
            for (size_t i = 0; i < docCount; ++i) {
                ptr[i] = CalcHash(ptr[i], (ui64)(binFPtr[i] == binFeatureIndex.Value));
            }
        }
    }
}

template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
inline void CalcHashes(
    const TFeatureCombination& featureCombination,
    const TFloatFeatureAccessor& floatFeatureAccessor,
    const TCatFeatureAccessor& catFeatureAccessor,
    size_t docCount,
    TVector<ui64>* result) {
    result->resize(docCount);
    std::fill(result->begin(), result->end(), 0);
    ui64* ptr = result->data();
    for (auto catFeatureIndex : featureCombination.CatFeatures) {
        for (size_t docId = 0; docId < docCount; ++docId) {
            ptr[docId] = CalcHash(ptr[docId],
                                  (ui64)catFeatureAccessor(catFeatureIndex, docId));
        }
    }
    for (const auto& floatFeature : featureCombination.BinFeatures) {
        for (size_t docId = 0; docId < docCount; ++docId) {
            ptr[docId] = CalcHash(ptr[docId],
                                  (ui64)(floatFeatureAccessor(floatFeature.FloatFeature, docId) > floatFeature.Split));
        }
    }
    for (auto oheFeature : featureCombination.OneHotFeatures) {
        for (size_t docId = 0; docId < docCount; ++docId) {
            ptr[docId] = CalcHash(ptr[docId],
                                  (ui64)(catFeatureAccessor(oheFeature.CatFeatureIdx, docId) == oheFeature.Value));
        }
    }
};
