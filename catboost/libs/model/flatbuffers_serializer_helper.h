#pragma once

#include "online_ctr.h"
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/flatbuffers/model.fbs.h>
#include <catboost/libs/options/enums.h>
#include <util/generic/hash.h>


class TModelPartsCachingSerializer
{
public:
    flatbuffers::FlatBufferBuilder FlatbufBuilder;

#define GENERATE_OFFSET_HELPER(TNativeType, TFlatbuffersType)\
    public:\
    flatbuffers::Offset<TFlatbuffersType> GetOffset(const TNativeType& value) {\
        if (OffsetsFor##TNativeType.contains(value)) {\
            return OffsetsFor##TNativeType.at(value);\
        }\
        auto result = value.FBSerialize(*this);\
        OffsetsFor##TNativeType[value] = result;\
        return result;\
    }\
private:\
    THashMap<TNativeType, flatbuffers::Offset<TFlatbuffersType>> OffsetsFor##TNativeType;

    GENERATE_OFFSET_HELPER(TModelCtrBase, NCatBoostFbs::TModelCtrBase)
    GENERATE_OFFSET_HELPER(TModelCtr, NCatBoostFbs::TModelCtr)
    GENERATE_OFFSET_HELPER(TFeatureCombination, NCatBoostFbs::TFeatureCombination)
#undef GENERATE_OFFSET_HELPER
};


inline static ENanMode NanValueTreatmentToNanMode(NCatBoostFbs::ENanValueTreatment nanValueTreatment) {
    switch (nanValueTreatment) {
        case NCatBoostFbs::ENanValueTreatment_AsFalse:
            return ENanMode::Min;
        case NCatBoostFbs::ENanValueTreatment_AsTrue:
            return ENanMode::Max;
        case NCatBoostFbs::ENanValueTreatment_AsIs:
            return ENanMode::Forbidden;
        default:
            ythrow TCatBoostException() <<  "Unknown ENanValueTreatment value";
    }
}

inline static NCatBoostFbs::ENanValueTreatment NanModeToNanValueTreatment(ENanMode nanMode) {
    switch (nanMode) {
        case ENanMode::Min:
            return NCatBoostFbs::ENanValueTreatment_AsFalse;
        case ENanMode::Max:
            return NCatBoostFbs::ENanValueTreatment_AsTrue;
        case ENanMode::Forbidden:
            return NCatBoostFbs::ENanValueTreatment_AsIs;
        default:
            ythrow TCatBoostException() <<  "Unknown ENanMode value " << nanMode;
    }
}
