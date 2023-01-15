#pragma once

#include "fwd.h"

#include "online_ctr.h"

#include <catboost/libs/helpers/guid.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/ysaveload.h>

#include <tuple>

struct TFeaturePosition {
    int Index = -1;
    int FlatIndex = -1;
public:
    TFeaturePosition() = default;

    TFeaturePosition(int index, int flatIndex)
        : Index(index)
        , FlatIndex(flatIndex)
    {}

    bool operator==(const TFeaturePosition& other) const {
        return std::tie(Index, FlatIndex) ==
               std::tie(other.Index, other.FlatIndex);
    }
    bool operator!=(const TFeaturePosition& other) const {
        return !(*this == other);
    }

    Y_SAVELOAD_DEFINE(Index, FlatIndex);
};

struct TFloatFeature {
    enum class ENanValueTreatment {
        AsIs,
        AsFalse,
        AsTrue
    };
public:
    bool HasNans = false;
    TFeaturePosition Position;
    TVector<float> Borders;
    TString FeatureId;
    ENanValueTreatment NanValueTreatment = ENanValueTreatment::AsIs;

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
        , Position(featureIndex, flatFeatureIndex)
        , Borders(borders)
        , FeatureId(featureId)
    {}

    bool operator==(const TFloatFeature& other) const {
        return std::tie(HasNans, Position, Borders, FeatureId) ==
               std::tie(other.HasNans, other.Position, other.Borders, other.FeatureId);
    }
    bool operator!=(const TFloatFeature& other) const {
        return !(*this == other);
    }

    bool UsedInModel() const {
        return !Borders.empty();
    }

    flatbuffers::Offset<NCatBoostFbs::TFloatFeature> FBSerialize(flatbuffers::FlatBufferBuilder& builder) const;

    void FBDeserialize(const NCatBoostFbs::TFloatFeature* fbObj);

    Y_SAVELOAD_DEFINE(HasNans, Position, Borders, FeatureId);
};

inline TVector<int> CountSplits(const TVector<TFloatFeature>& floatFeatures) {
    TVector<int> result;
    for (int i = 0; i < floatFeatures.ysize(); ++i) {
        result.push_back(floatFeatures[i].Borders.ysize());
    }
    return result;
}

struct TCatFeature {
public:
    TFeaturePosition Position;
    TString FeatureId;
public:
    TCatFeature() = default;

    TCatFeature(
        bool usedInModel,
        int featureIndex,
        int flatFeatureIndex,
        TString featureId
    )
        : Position(featureIndex, flatFeatureIndex)
        , FeatureId(featureId)
        , IsUsedInModel(usedInModel)
    {}

    bool operator==(const TCatFeature& other) const {
        return std::tie(Position, FeatureId) ==
               std::tie(other.Position, other.FeatureId);
    }
    bool operator!=(const TCatFeature& other) const {
        return !(*this == other);
    }

    bool UsedInModel() const {
        return IsUsedInModel;
    }

    void SetUsedInModel(bool isUsedInModel) {
        IsUsedInModel = isUsedInModel;
    }

    flatbuffers::Offset<NCatBoostFbs::TCatFeature> FBSerialize(flatbuffers::FlatBufferBuilder& builder) const;
    void FBDeserialize(const NCatBoostFbs::TCatFeature* fbObj);
    Y_SAVELOAD_DEFINE(IsUsedInModel, Position, FeatureId);

private:
    bool IsUsedInModel = true;
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
    ) const;
    void FBDeserialize(const NCatBoostFbs::TOneHotFeature* fbObj);
    Y_SAVELOAD_DEFINE(CatFeatureIndex, Values);
};

struct TTextFeature {
public:
    TFeaturePosition Position;
    TString FeatureId;
public:
    TTextFeature() = default;

    TTextFeature(
        bool usedInModel,
        int featureIndex,
        int flatFeatureIndex,
        TString featureId
    )
        : Position(featureIndex, flatFeatureIndex)
        , FeatureId(std::move(featureId))
        , IsUsedInModel(usedInModel)
    {}

    bool operator==(const TTextFeature& other) const {
        return std::tie(Position, FeatureId) ==
            std::tie(other.Position, other.FeatureId);
    }
    bool operator!=(const TTextFeature& other) const {
        return !(*this == other);
    }

    bool UsedInModel() const {
        return IsUsedInModel;
    };

    void SetUsedInModel(bool isUsedInModel) {
        IsUsedInModel = isUsedInModel;
    }

    flatbuffers::Offset<NCatBoostFbs::TTextFeature> FBSerialize(
        flatbuffers::FlatBufferBuilder& builder
    ) const;
    void FBDeserialize(const NCatBoostFbs::TTextFeature* fbObj);
    Y_SAVELOAD_DEFINE(IsUsedInModel, Position, FeatureId);

private:
    bool IsUsedInModel = true;
};

struct TEstimatedFeature {
    int SourceFeatureIndex = -1;
    NCB::TGuid CalcerId;
    int LocalIndex = -1;
    TVector<float> Borders;

public:
    TEstimatedFeature() = default;

    TEstimatedFeature(
        int sourceFeatureIndex,
        const NCB::TGuid& calcerId,
        int localIndex
    )
        : SourceFeatureIndex(sourceFeatureIndex)
        , CalcerId(calcerId)
        , LocalIndex(localIndex)
    {}

    bool operator<(const TEstimatedFeature& other) const {
        return std::tie(
            SourceFeatureIndex,
            CalcerId,
            LocalIndex) <
               std::tie(
                   other.SourceFeatureIndex,
                   other.CalcerId,
                   other.LocalIndex);
    }

    bool operator==(const TEstimatedFeature& other) const {
        return std::tie(
            SourceFeatureIndex,
            CalcerId,
            LocalIndex) ==
            std::tie(
                other.SourceFeatureIndex,
                other.CalcerId,
                other.LocalIndex);
    }

    bool operator!=(const TEstimatedFeature& other) const {
        return !(*this == other);
    }

    flatbuffers::Offset<NCatBoostFbs::TEstimatedFeature> FBSerialize(
        flatbuffers::FlatBufferBuilder& builder
    ) const;
    void FBDeserialize(const NCatBoostFbs::TEstimatedFeature* fbObj);
    Y_SAVELOAD_DEFINE(SourceFeatureIndex, CalcerId, LocalIndex, Borders);
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
    bool operator<(const TCtrFeature& other) const {
        return std::tie(Ctr, Borders) < std::tie(other.Ctr, other.Borders);
    }
    flatbuffers::Offset<NCatBoostFbs::TCtrFeature> FBSerialize(TModelPartsCachingSerializer& serializer) const;
    void FBDeserialize(const NCatBoostFbs::TCtrFeature* fbObj);
    Y_SAVELOAD_DEFINE(Ctr, Borders);
};
