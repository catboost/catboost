#pragma once

#include "online_ctr.h"

#include <catboost/libs/model/flatbuffers/features.fbs.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/ysaveload.h>

#include <tuple>


struct TFloatFeature {
    bool HasNans = false;
    int FeatureIndex = -1;
    int FlatFeatureIndex = -1;
    TVector<float> Borders;
    TString FeatureId;
    NCatBoostFbs::ENanValueTreatment NanValueTreatment = NCatBoostFbs::ENanValueTreatment_AsIs;

public:
    TFloatFeature() = default;
    TFloatFeature(
        bool hasNans,
        int featureIndex,
        int flatFeatureIndex,
        const TVector<float>& borders,
        const TString& featureId = ""
    )
        : HasNans(hasNans)
        , FeatureIndex(featureIndex)
        , FlatFeatureIndex(flatFeatureIndex)
        , Borders(borders)
        , FeatureId(featureId)
    {}

    bool operator==(const TFloatFeature& other) const {
        return std::tie(HasNans, FeatureIndex, FlatFeatureIndex, Borders, FeatureId) ==
               std::tie(other.HasNans, other.FeatureIndex, other.FlatFeatureIndex, other.Borders, other.FeatureId);
    }
    bool operator!=(const TFloatFeature& other) const {
        return !(*this == other);
    }

    bool UsedInModel() const {
        return !Borders.empty();
    }

    flatbuffers::Offset<NCatBoostFbs::TFloatFeature> FBSerialize(flatbuffers::FlatBufferBuilder& builder) const {
        return NCatBoostFbs::CreateTFloatFeatureDirect(
            builder,
            HasNans,
            FeatureIndex,
            FlatFeatureIndex,
            &Borders,
            FeatureId.empty() ? nullptr : FeatureId.data(),
            NanValueTreatment
        );
    }

    void FBDeserialize(const NCatBoostFbs::TFloatFeature* fbObj) {
        if (fbObj == nullptr) {
            return;
        }
        HasNans = fbObj->HasNans();
        FeatureIndex = fbObj->Index();
        FlatFeatureIndex = fbObj->FlatIndex();
        NanValueTreatment = fbObj->NanValueTreatment();
        if (fbObj->Borders()) {
            Borders.assign(fbObj->Borders()->begin(), fbObj->Borders()->end());
        }
        if (fbObj->FeatureId()) {
            FeatureId.assign(fbObj->FeatureId()->data(), fbObj->FeatureId()->Length());
        }
    }

    Y_SAVELOAD_DEFINE(HasNans, FeatureIndex, FlatFeatureIndex, Borders, FeatureId);
};

inline TVector<int> CountSplits(const TVector<TFloatFeature>& floatFeatures) {
    TVector<int> result;
    for (int i = 0; i < floatFeatures.ysize(); ++i) {
        result.push_back(floatFeatures[i].Borders.ysize());
    }
    return result;
}

struct TCatFeature {
    bool UsedInModel = true;
    int FeatureIndex = -1;
    int FlatFeatureIndex = -1;
    TString FeatureId;

public:
    bool operator==(const TCatFeature& other) const {
        return std::tie(FeatureIndex, FlatFeatureIndex, FeatureId) ==
               std::tie(other.FeatureIndex, other.FlatFeatureIndex, other.FeatureId);
    }
    bool operator!=(const TCatFeature& other) const {
        return !(*this == other);
    }

    flatbuffers::Offset<NCatBoostFbs::TCatFeature> FBSerialize(flatbuffers::FlatBufferBuilder& builder) const {
        return NCatBoostFbs::CreateTCatFeatureDirect(
            builder,
            FeatureIndex,
            FlatFeatureIndex,
            FeatureId.empty() ? nullptr : FeatureId.data(),
            UsedInModel
        );
    }
    void FBDeserialize(const NCatBoostFbs::TCatFeature* fbObj) {
        FeatureIndex = fbObj->Index();
        FlatFeatureIndex = fbObj->FlatIndex();
        if (fbObj->FeatureId()) {
            FeatureId.assign(fbObj->FeatureId()->data(), fbObj->FeatureId()->size());
        }
        UsedInModel = fbObj->UsedInModel();
    }
    Y_SAVELOAD_DEFINE(FeatureIndex, FlatFeatureIndex, FeatureId);
};

struct TOneHotFeature {
    int CatFeatureIndex = -1;
    TVector<int> Values;
    TVector<TString> StringValues;

public:
    bool operator==(const TOneHotFeature& other) const {
        return std::tie(CatFeatureIndex, Values) == std::tie(other.CatFeatureIndex, other.Values);
    }
    bool operator!=(const TOneHotFeature& other) const {
        return !(*this == other);
    }

    flatbuffers::Offset<NCatBoostFbs::TOneHotFeature> FBSerialize(
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
    void FBDeserialize(const NCatBoostFbs::TOneHotFeature* fbObj) {
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
    Y_SAVELOAD_DEFINE(CatFeatureIndex, Values);
};

class TModelPartsCachingSerializer;

struct TCtrFeature {
    TModelCtr Ctr;
    TVector<float> Borders;

public:
    bool operator==(const TCtrFeature& other) const {
        return std::tie(Ctr, Borders) == std::tie(other.Ctr, other.Borders);
    }
    bool operator!=(const TCtrFeature& other) const {
        return !(*this == other);
    }
    flatbuffers::Offset<NCatBoostFbs::TCtrFeature> FBSerialize(TModelPartsCachingSerializer& serializer) const;
    void FBDeserialize(const NCatBoostFbs::TCtrFeature* fbObj) {
        if (fbObj == nullptr) {
            return;
        }
        Ctr.FBDeserialize(fbObj->Ctr());
        if (fbObj->Borders() && fbObj->Borders()->size() != 0) {
            Borders.assign(fbObj->Borders()->begin(), fbObj->Borders()->end());
        }
    }
    Y_SAVELOAD_DEFINE(Ctr, Borders);
};
