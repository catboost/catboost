#pragma once

#include "projection.h"
#include "split.h"

#include <catboost/libs/data/columns.h>
#include <catboost/libs/data/ctrs.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/model/target_classifier.h>

#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/algo_helpers/scratch_cache.h>

#include <library/cpp/containers/2d_array/2d_array.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <functional>



class TCtrHelper;
class TCtrValueTable;
class TFold;
class TLearnContext;

namespace NCB {
    template <class TSize>
    class TArraySubsetIndexing;
}

namespace NCatboostOptions {
    class TCatFeatureParams;
}

namespace NPar {
    class ILocalExecutor;
}


const int SIMPLE_CLASSES_COUNT = 2;


class TOnlineCtrBase : public TThrRefBase {
public:
    virtual ~TOnlineCtrBase() = default;

    virtual NCB::TOnlineCtrUniqValuesCounts GetUniqValuesCounts(const TProjection& projection) const = 0;

    /* BorderCount in ctr is not used here
     * datasetIdx means 0 - learn, 1 - eval #0, 2 - eval #1 ...
     */
    virtual TConstArrayRef<ui8> GetData(const TCtr& ctr, ui32 datasetIdx) const = 0;
};

struct IOnlineCtrProjectionDataWriter {
    virtual ~IOnlineCtrProjectionDataWriter() = default;

    virtual void SetUniqValuesCounts(const NCB::TOnlineCtrUniqValuesCounts& uniqValuesCounts) = 0;

    virtual void AllocateData(size_t ctrCount) = 0;

    /* call after AllocateData has been called
      it must be thread-safe to call concurrently for different ctrIdx
     */
    virtual void AllocateCtrData(size_t ctrIdx, size_t targetBorderCount, size_t priorCount) = 0;

    virtual TArrayRef<ui8> GetDataBuffer(int ctrIdx, int targetBorderIdx, int priorIdx, int datasetIdx) = 0;
};


struct TOnlineCtrPerProjectionData {
    NCB::TOnlineCtrUniqValuesCounts UniqValuesCounts;
    TVector<TArray2D<TVector<ui8>>> Feature; // Feature[ctrIdx][targetBorderIdx][priorIdx][docIdx]
};


class TOwnedOnlineCtr final : public TOnlineCtrBase {
public:
    THashMap<TProjection, TOnlineCtrPerProjectionData> Data;
    TVector<NCB::TIndexRange<size_t>> DatasetsObjectRanges;

public:
    NCB::TOnlineCtrUniqValuesCounts GetUniqValuesCounts(const TProjection& projection) const override {
        return Data.at(projection).UniqValuesCounts;
    }

    TConstArrayRef<ui8> GetData(const TCtr& ctr, ui32 datasetIdx) const override {
        const ui8* allDataPtr = Data.at(
            ctr.Projection
        ).Feature[ctr.CtrIdx][ctr.TargetBorderIdx][ctr.PriorIdx].data();
        return TConstArrayRef<ui8>(
            allDataPtr + DatasetsObjectRanges[datasetIdx].Begin,
            allDataPtr + DatasetsObjectRanges[datasetIdx].End
        );
    }

    void EnsureProjectionInData(const TProjection& projection) {
        Data[projection];
    }

    void DropEmptyData();
};


struct TPrecomputedOnlineCtr final : public TOnlineCtrBase {
public:
    NCB::TPrecomputedOnlineCtrData Data;

public:
    NCB::TOnlineCtrUniqValuesCounts GetUniqValuesCounts(const TProjection& projection) const override {
        Y_ASSERT(projection.IsSingleCatFeature());
        return Data.Meta.ValuesCounts.at(projection.CatFeatures[0]);
    }

    TConstArrayRef<ui8> GetData(const TCtr& ctr, ui32 datasetIdx) const override;
};


inline ui8 CalcCTR(float countInClass, int totalCount, float prior, float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;
}

void CalcNormalization(const TVector<float>& priors, TVector<float>* shift, TVector<float>* norm);


void ComputeOnlineCTRs(
    const NCB::TTrainingDataProviders& data,
    const TProjection& proj,
    const TCtrHelper& ctrHelper,
    const NCB::TFeaturesArraySubsetIndexing& foldLearnPermutationFeaturesSubset,
    const TVector<TVector<int>>& foldLearnTargetClass,
    const TVector<int>& foldTargetClassesCount,
    const NCatboostOptions::TCatFeatureParams& catFeatureParams,
    NPar::ILocalExecutor* localExecutor,
    NCB::TScratchCache* scratchCache,
    IOnlineCtrProjectionDataWriter* writer
);

void ComputeOnlineCTRs(
    const NCB::TTrainingDataProviders& data,
    const TFold& fold,
    const TProjection& proj,
    TLearnContext* ctx,
    TOwnedOnlineCtr* onlineCtrStorage
);


struct TDatasetDataForFinalCtrs {
    NCB::TTrainingDataProviders Data;

    TMaybe<const NCB::TArraySubsetIndexing<ui32>*> LearnPermutation;

    // permuted according to LearnPermutation if it is defined
    TMaybe<TVector<TConstArrayRef<float>>> Targets;

    // class data needed only if any of used ctrs need target classifier

    // permuted according to LearnPermutation if it is defined
    TMaybe<const TVector<TVector<int>>*> LearnTargetClass; // [targetBorderClassifierIdx][objectIdx]
    TMaybe<const TVector<int>*> TargetClassesCount; // [targetBorderClassifierIdx]
    TMaybe<const TVector<TTargetClassifier>*> TargetClassifiers; // [targetBorderClassifierIdx]
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
    NPar::ILocalExecutor* localExecutor
);
