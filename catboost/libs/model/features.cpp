#include "features.h"

#include <catboost/libs/model/flatbuffers/features.fbs.h>

#include "flatbuffers_serializer_helper.h"


flatbuffers::Offset<NCatBoostFbs::TCtrFeature> TCtrFeature::FBSerialize(TModelPartsCachingSerializer& serializer) const {
    return NCatBoostFbs::CreateTCtrFeatureDirect(
        serializer.FlatbufBuilder,
        serializer.GetOffset(Ctr),
        &Borders
    );
}

void TCtrFeature::FBDeserialize(const NCatBoostFbs::TCtrFeature* fbObj) {
    if (fbObj == nullptr) {
        return;
    }
    Ctr.FBDeserialize(fbObj->Ctr());
    if (fbObj->Borders() && fbObj->Borders()->size() != 0) {
        Borders.assign(fbObj->Borders()->begin(), fbObj->Borders()->end());
    }
}

flatbuffers::Offset<NCatBoostFbs::TFloatFeature> TFloatFeature::FBSerialize(
    flatbuffers::FlatBufferBuilder& builder
) const {
    return NCatBoostFbs::CreateTFloatFeatureDirect(
        builder,
        HasNans,
        Position.Index,
        Position.FlatIndex,
        &Borders,
        FeatureId.empty() ? nullptr : FeatureId.data(),
        NanModeToFbsEnumValue(NanValueTreatment)
    );
}

void TFloatFeature::FBDeserialize(const NCatBoostFbs::TFloatFeature* fbObj) {
    if (fbObj == nullptr) {
        return;
    }
    HasNans = fbObj->HasNans();
    Position.Index = fbObj->Index();
    Position.FlatIndex = fbObj->FlatIndex();
    NanValueTreatment = NanModeFromFbsEnumValue(fbObj->NanValueTreatment());
    if (fbObj->Borders()) {
        Borders.assign(fbObj->Borders()->begin(), fbObj->Borders()->end());
    }
    if (fbObj->FeatureId()) {
        FeatureId.assign(fbObj->FeatureId()->data(), fbObj->FeatureId()->size());
    }
}

flatbuffers::Offset<NCatBoostFbs::TCatFeature> TCatFeature::FBSerialize(
    flatbuffers::FlatBufferBuilder& builder
) const {
    return NCatBoostFbs::CreateTCatFeatureDirect(
        builder,
        Position.Index,
        Position.FlatIndex,
        FeatureId.empty() ? nullptr : FeatureId.data(),
        IsUsedInModel
    );
}

void TCatFeature::FBDeserialize(const NCatBoostFbs::TCatFeature* fbObj) {
    Position.Index = fbObj->Index();
    Position.FlatIndex = fbObj->FlatIndex();
    if (fbObj->FeatureId()) {
        FeatureId.assign(fbObj->FeatureId()->data(), fbObj->FeatureId()->size());
    }
    IsUsedInModel = fbObj->UsedInModel();
}

flatbuffers::Offset<NCatBoostFbs::TOneHotFeature> TOneHotFeature::FBSerialize(
    flatbuffers::FlatBufferBuilder& builder
) const {
    std::vector<flatbuffers::Offset<flatbuffers::String>> vectorOfStringOffsets;
    if (!StringValues.empty()) {
        for (auto strValue : StringValues) {
            vectorOfStringOffsets.push_back(builder.CreateString(strValue.data(), strValue.size()));
        }
    }
    return NCatBoostFbs::CreateTOneHotFeatureDirect(
        builder,
        CatFeatureIndex,
        &Values,
        vectorOfStringOffsets.empty()? nullptr : &vectorOfStringOffsets
    );
}

void TOneHotFeature::FBDeserialize(const NCatBoostFbs::TOneHotFeature* fbObj) {
    if (fbObj == nullptr) {
        return;
    }
    CatFeatureIndex = fbObj->Index();
    if (fbObj->Values()) {
        Values.assign(fbObj->Values()->begin(), fbObj->Values()->end());
    }
    if (fbObj->StringValues()) {
        StringValues.resize(fbObj->StringValues()->size());
        for (size_t i = 0; i < StringValues.size(); ++i) {
            auto fbString = fbObj->StringValues()->Get(i);
            StringValues[i].assign(fbString->data(), fbString->size());
        }
    }
}

flatbuffers::Offset<NCatBoostFbs::TTextFeature> TTextFeature::FBSerialize(
    flatbuffers::FlatBufferBuilder& builder
) const {
    return NCatBoostFbs::CreateTTextFeatureDirect(
        builder,
        Position.Index,
        Position.FlatIndex,
        FeatureId.empty() ? nullptr : FeatureId.data(),
        IsUsedInModel
    );
}

flatbuffers::Offset<NCatBoostFbs::TEmbeddingFeature> TEmbeddingFeature::FBSerialize(
        flatbuffers::FlatBufferBuilder& builder
) const {
    return NCatBoostFbs::CreateTEmbeddingFeatureDirect(
        builder,
        Position.Index,
        Position.FlatIndex,
        FeatureId.empty() ? nullptr : FeatureId.data(),
        Dimension,
        IsUsedInModel
    );
}

void TTextFeature::FBDeserialize(const NCatBoostFbs::TTextFeature* fbObj) {
    Position.Index = fbObj->Index();
    Position.FlatIndex = fbObj->FlatIndex();
    if (fbObj->FeatureId()) {
        FeatureId.assign(fbObj->FeatureId()->data(), fbObj->FeatureId()->size());
    }
    IsUsedInModel = fbObj->UsedInModel();
}

void TEmbeddingFeature::FBDeserialize(const NCatBoostFbs::TEmbeddingFeature* fbObj) {
    Position.Index = fbObj->Index();
    Position.FlatIndex = fbObj->FlatIndex();
    if (fbObj->FeatureId()) {
        FeatureId.assign(fbObj->FeatureId()->data(), fbObj->FeatureId()->size());
    }
    Dimension = fbObj->Dimension();
    IsUsedInModel = fbObj->UsedInModel();
}

flatbuffers::Offset<NCatBoostFbs::TEstimatedFeature> TEstimatedFeature::FBSerialize(
    flatbuffers::FlatBufferBuilder& builder
) const {
    const auto calcerFbsGuid = CreateFbsGuid(ModelEstimatedFeature.CalcerId);
    return NCatBoostFbs::CreateTEstimatedFeatureDirect(
        builder,
        ModelEstimatedFeature.SourceFeatureId,
        &calcerFbsGuid,
        ModelEstimatedFeature.LocalId,
        &Borders,
        SourceFeatureTypeToFbsEnumValue(ModelEstimatedFeature.SourceFeatureType)
    );
}

void TEstimatedFeature::FBDeserialize(const NCatBoostFbs::TEstimatedFeature* fbObj) {
    ModelEstimatedFeature.SourceFeatureId = fbObj->SourceFeatureIndex();
    ModelEstimatedFeature.CalcerId = GuidFromFbs(fbObj->CalcerId());
    ModelEstimatedFeature.LocalId = fbObj->LocalIndex();
    if (fbObj->Borders()) {
        Borders.assign(fbObj->Borders()->begin(), fbObj->Borders()->end());
    }
    ModelEstimatedFeature.SourceFeatureType = SourceFeatureTypeFromFbsEnumValue(fbObj->SourceFeatureType());
}
