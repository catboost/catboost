#pragma once

#include "index_hash_calcer.h"
#include "projection.h"
#include "target_classifier.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/quantized_features_info.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/online_ctr.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/maybe.h>
#include <util/system/types.h>

#include <functional>

struct TFold;

const int SIMPLE_CLASSES_COUNT = 2;


struct TOnlineCTR {
    TVector<TArray2D<TVector<ui8>>> Feature; // Feature[ctrIdx][classIdx][priorIdx][docIdx]
    size_t UniqueValuesCount = 0;
    size_t CounterUniqueValuesCount = 0; // Counter ctrs could have more values than other types when  counter_calc_method == Full

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

class TLearnContext;

void ComputeOnlineCTRs(const NCB::TTrainingForCPUDataProviders& data,
                       const TFold& fold,
                       const TProjection& proj,
                       const TLearnContext* ctx,
                       TOnlineCTR* dst);

class TCtrValueTable;


struct TDatasetDataForFinalCtrs {
    NCB::TTrainingForCPUDataProviders Data;

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
    NPar::TLocalExecutor& localExecutor,
    const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjectionMap,
    const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
    const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtrs,
    ECounterCalc counterCalcMethod,
    const TVector<TModelCtrBase>& usedCtrBases,
    std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback
);
