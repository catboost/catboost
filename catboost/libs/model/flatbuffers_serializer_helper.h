#pragma once

#include "features.h"
#include "online_ctr.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/flatbuffers/model.fbs.h>
#include <catboost/private/libs/options/enums.h>
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


inline static ENanMode NanValueTreatmentToNanMode(TFloatFeature::ENanValueTreatment nanValueTreatment) {
    switch (nanValueTreatment) {
        case TFloatFeature::ENanValueTreatment::AsFalse:
            return ENanMode::Min;
        case TFloatFeature::ENanValueTreatment::AsTrue:
            return ENanMode::Max;
        case TFloatFeature::ENanValueTreatment::AsIs:
            return ENanMode::Forbidden;
        default:
            ythrow TCatBoostException() <<  "Unknown ENanValueTreatment value";
    }
}

inline static TFloatFeature::ENanValueTreatment NanModeToNanValueTreatment(ENanMode nanMode) {
    switch (nanMode) {
        case ENanMode::Min:
            return TFloatFeature::ENanValueTreatment::AsFalse;
        case ENanMode::Max:
            return TFloatFeature::ENanValueTreatment::AsTrue;
        case ENanMode::Forbidden:
            return TFloatFeature::ENanValueTreatment::AsIs;
        default:
            ythrow TCatBoostException() <<  "Unknown ENanMode value " << nanMode;
    }
}

inline static NCatBoostFbs::ENanValueTreatment NanModeToFbsEnumValue(TFloatFeature::ENanValueTreatment nanTreatment) {
    switch (nanTreatment) {
        case TFloatFeature::ENanValueTreatment::AsFalse:
            return NCatBoostFbs::ENanValueTreatment_AsFalse;
        case TFloatFeature::ENanValueTreatment::AsTrue:
            return NCatBoostFbs::ENanValueTreatment_AsTrue;
        case TFloatFeature::ENanValueTreatment::AsIs:
            return NCatBoostFbs::ENanValueTreatment_AsIs;
        default:
            ythrow TCatBoostException() <<  "Unknown ENanValueTreatment value " << nanTreatment;
    }
}

inline static TFloatFeature::ENanValueTreatment NanModeFromFbsEnumValue(NCatBoostFbs::ENanValueTreatment nanTreatment) {
    switch (nanTreatment) {
        case NCatBoostFbs::ENanValueTreatment_AsFalse:
            return TFloatFeature::ENanValueTreatment::AsFalse;
        case NCatBoostFbs::ENanValueTreatment_AsTrue:
            return TFloatFeature::ENanValueTreatment::AsTrue;
        case NCatBoostFbs::ENanValueTreatment_AsIs:
            return TFloatFeature::ENanValueTreatment::AsIs;
        default:
            ythrow TCatBoostException() <<  "Unknown NCatBoostFbs::ENanValueTreatment value " << (int)nanTreatment;
    }
}

inline static NCatBoostFbs::ESourceFeatureType SourceFeatureTypeToFbsEnumValue(EEstimatedSourceFeatureType sourceFeatureType) {
    switch (sourceFeatureType) {
        case EEstimatedSourceFeatureType::Text:
            return NCatBoostFbs::ESourceFeatureType_Text;
        case EEstimatedSourceFeatureType::Embedding:
            return NCatBoostFbs::ESourceFeatureType_Embedding;
        default:
            ythrow TCatBoostException() <<  "Unknown EEstimatedSourceFeatureType value " << (int)sourceFeatureType;
    }
}

inline static EEstimatedSourceFeatureType SourceFeatureTypeFromFbsEnumValue(NCatBoostFbs::ESourceFeatureType sourceFeatureType) {
    switch (sourceFeatureType) {
        case NCatBoostFbs::ESourceFeatureType_Text:
            return EEstimatedSourceFeatureType::Text;
        case NCatBoostFbs::ESourceFeatureType_Embedding:
            return EEstimatedSourceFeatureType::Embedding;
        default:
            ythrow TCatBoostException() <<  "Unknown NCatBoostFbs::ESourceFeatureType value " << (int)sourceFeatureType;
    }
}
