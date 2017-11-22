#pragma once

#include "online_ctr.h"
#include <catboost/libs/model/flatbuffers/model.fbs.h>
#include <util/generic/hash.h>


class TModelPartsCachingSerializer
{
public:
    flatbuffers::FlatBufferBuilder FlatbufBuilder;

#define GENERATE_OFFSET_HELPER(TNativeType, TFlatbuffersType)\
    public:\
    flatbuffers::Offset<TFlatbuffersType> GetOffset(const TNativeType& value) {\
        if (OffsetsFor##TNativeType.has(value)) {\
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
