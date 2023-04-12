#pragma once

#include "hash.h"
#include "online_ctr.h"
#include "features.h"
#include "ctr_value_table.h"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/stream/fwd.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <algorithm>


class ICtrProvider : public TThrRefBase {
public:
    virtual ~ICtrProvider() {
    }

    virtual bool HasNeededCtrs(TConstArrayRef<TModelCtr> neededCtrs) const = 0;

    virtual void CalcCtrs(
        const TConstArrayRef<TModelCtr> neededCtrs,
        const TConstArrayRef<ui8> binarizedFeatures, // vector of binarized float & one hot features
        const TConstArrayRef<ui32> hashedCatFeatures,
        size_t docCount,
        TArrayRef<float> result) = 0;

    virtual void SetupBinFeatureIndexes(
        const TConstArrayRef<TFloatFeature> floatFeatures,
        const TConstArrayRef<TOneHotFeature> oheFeatures,
        const TConstArrayRef<TCatFeature> catFeatures) = 0;

    virtual void AddCtrCalcerData(TCtrValueTable&& valueTable) = 0;
    virtual bool IsSerializable() const {
        return false;
    }

    virtual void DropUnusedTables(TConstArrayRef<TModelCtrBase> usedModelCtrBase) = 0;

    virtual void Save(IOutputStream* ) const {
        CB_ENSURE(false, "Serialization not allowed");
    };

    virtual void Load(IInputStream* ) {
        CB_ENSURE(false, "Deserialization not allowed");
    };

    // can use this later for complex model deserialization logic
    virtual TString ModelPartIdentifier() const = 0;

    virtual TIntrusivePtr<ICtrProvider> Clone() const {
        CB_ENSURE(false, "Cloning not supported");
    }
};

// slow reference realization
inline ui64 CalcHash(
    const TConstArrayRef<ui8>& binarizedFeatures,
    const TConstArrayRef<ui32>& hashedCatFeatures,
    const TConstArrayRef<int>& transposedCatFeatureIndexes,
    const TConstArrayRef<int>& binarizedFeatureIndexes) {
    ui64 result = 0;
    for (const int featureIdx : transposedCatFeatureIndexes) {
        result = CalcHash(result, (ui64)(int)hashedCatFeatures[featureIdx]);
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

public:
    TBinFeatureIndexValue() = default;
    TBinFeatureIndexValue(ui32 binIndex, bool checkValueEqual, ui8 value)
        : BinIndex(binIndex)
        , CheckValueEqual(checkValueEqual)
        , Value(value)
    {}
};

inline void CalcHashes(
    const TConstArrayRef<ui8>& binarizedFeatures,
    const TConstArrayRef<ui32>& hashedCatFeatures,
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
            ptr[i] = CalcHash(ptr[i], (ui64)(int)valPtr[i]);
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

enum class ECtrTableMergePolicy {
    FailIfCtrIntersects,
    LeaveMostDiversifiedTable,
    IntersectingCountersAverage,
    KeepAllTables
};

TIntrusivePtr<ICtrProvider> MergeCtrProvidersData(
    const TVector<TIntrusivePtr<ICtrProvider>>& providers,
    ECtrTableMergePolicy mergePolicy);
