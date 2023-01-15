#pragma once

#include "projection.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/model/online_ctr.h>

#include <util/generic/maybe.h>
#include <util/system/types.h>

#include <functional>


class TCtrValueTable;
class TFold;
class TLearnContext;

namespace NCB {
    template <class TSize>
    class TArraySubsetIndexing;
}

namespace NPar {
    class TLocalExecutor;
}


const int SIMPLE_CLASSES_COUNT = 2;


struct TOnlineCTR {
    TVector<TArray2D<TVector<ui8>>> Feature; // Feature[ctrIdx][classIdx][priorIdx][docIdx]
    size_t UniqueValuesCount = 0;

    // Counter ctrs could have more values than other types when counter_calc_method == Full
    size_t CounterUniqueValuesCount = 0;

public:
    size_t GetMaxUniqueValueCount() const {
        return Max(UniqueValuesCount, CounterUniqueValuesCount);
    }
    size_t GetUniqueValueCountForType(ECtrType type) const {
        if (ECtrType::Counter == type) {
            return CounterUniqueValuesCount;
        } else {
            return UniqueValuesCount;
        }
    }
};

using TOnlineCTRHash = THashMap<TProjection, TOnlineCTR>;

inline ui8 CalcCTR(float countInClass, int totalCount, float prior, float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;
}

void CalcNormalization(const TVector<float>& priors, TVector<float>* shift, TVector<float>* norm);


void ComputeOnlineCTRs(
    const NCB::TTrainingDataProviders& data,
    const TFold& fold,
    const TProjection& proj,
    const TLearnContext* ctx,
    TOnlineCTR* dst
);


struct TDatasetDataForFinalCtrs {
    NCB::TTrainingDataProviders Data;

    TMaybe<const NCB::TArraySubsetIndexing<ui32>*> LearnPermutation;

    // permuted according to LearnPermutation if it is defined
    TMaybe<TConstArrayRef<float>> Targets;

    // class data needed only if any of used ctrs need target classifier

    // permuted according to LearnPermutation if it is defined
    TMaybe<const TVector<TVector<int>>*> LearnTargetClass; // [targetBorderClassifierIdx][objectIdx]
    TMaybe<const TVector<int>*> TargetClassesCount; // [targetBorderClassifierIdx]
};

void CalcFinalCtrsAndSaveToModel(
    ui64 cpuRamLimit,
    const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjectionMap,
    const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
    const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtrs,
    ECounterCalc counterCalcMethod,
    const TVector<TModelCtrBase>& usedCtrBases,
    std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback,
    NPar::TLocalExecutor* localExecutor
);
