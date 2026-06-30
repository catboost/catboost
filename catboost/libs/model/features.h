#pragma once

#include "fwd.h"

#include "model_estimated_features.h"
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

struct TFeatureBase {
public:
    TFeaturePosition Position;
    TString FeatureId;
public:
    TFeatureBase() = default;
    TFeatureBase(TFeaturePosition position, TString featureId)
        : Position(std::move(position))
        , FeatureId(std::move(featureId))
    {}
};

struct TFloatFeature : public TFeatureBase {
    enum class ENanValueTreatment {
        AsIs,
        AsFalse,
        AsTrue
    };
public:
    bool HasNans = false;

    TVector<float> Borders;

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
        : TFeatureBase(TFeaturePosition(featureIndex, flatFeatureIndex), featureId)
        , HasNans(hasNans)
        , Borders(borders)
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

struct TCatFeature : public TFeatureBase {
public:
    TCatFeature() = default;

    TCatFeature(
        bool usedInModel,
        int featureIndex,
        int flatFeatureIndex,
        TString featureId
    )
        : TFeatureBase(TFeaturePosition(featureIndex, flatFeatureIndex), featureId)
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

struct TTextFeature : public TFeatureBase {
public:
    TTextFeature() = default;

    TTextFeature(
        bool usedInModel,
        int featureIndex,
        int flatFeatureIndex,
        TString featureId
    )
        : TFeatureBase(TFeaturePosition(featureIndex, flatFeatureIndex), featureId)
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

struct TEmbeddingFeature : public TFeatureBase {
public:
    int Dimension = 0;
public:
    TEmbeddingFeature() = default;

    TEmbeddingFeature(
        bool usedInModel,
        int featureIndex,
        int flatFeatureIndex,
        TString featureId,
        int dimension
    )
        : TFeatureBase(TFeaturePosition(featureIndex, flatFeatureIndex), std::move(featureId))
        , Dimension(dimension)
        , IsUsedInModel(usedInModel)
    {}

    bool operator==(const TEmbeddingFeature& other) const {
        return std::tie(Position, FeatureId, Dimension) ==
            std::tie(other.Position, other.FeatureId, other.Dimension);
    }
    bool operator!=(const TEmbeddingFeature& other) const {
        return !(*this == other);
    }

    bool UsedInModel() const {
        return IsUsedInModel;
    };

    void SetUsedInModel(bool isUsedInModel) {
        IsUsedInModel = isUsedInModel;
    }

    flatbuffers::Offset<NCatBoostFbs::TEmbeddingFeature> FBSerialize(
        flatbuffers::FlatBufferBuilder& builder
    ) const;
    void FBDeserialize(const NCatBoostFbs::TEmbeddingFeature* fbObj);
    Y_SAVELOAD_DEFINE(IsUsedInModel, Position, FeatureId, Dimension);

private:
    bool IsUsedInModel = true;
};

struct TEstimatedFeature {
public:
    TModelEstimatedFeature ModelEstimatedFeature;
    TVector<float> Borders;

public:
    TEstimatedFeature() = default;

    TEstimatedFeature(
        int sourceFeatureId,
        NCB::TGuid calcerId,
        int localId,
        EEstimatedSourceFeatureType sourceFeatureType
    )
        : ModelEstimatedFeature(TModelEstimatedFeature(sourceFeatureId, calcerId, localId, sourceFeatureType))
    {}

    TEstimatedFeature(
        int sourceFeatureId,
        int localId,
        EEstimatedSourceFeatureType sourceFeatureType
    )
        : ModelEstimatedFeature(TModelEstimatedFeature(sourceFeatureId, localId, sourceFeatureType))
    {}

    TEstimatedFeature(
        const TModelEstimatedFeature& modelEstimatedFeature
    )
        : ModelEstimatedFeature(modelEstimatedFeature)
    {}

    TEstimatedFeature(
        const TModelEstimatedFeature& modelEstimatedFeature,
        const TVector<float>& borders
    )
        : ModelEstimatedFeature(modelEstimatedFeature)
        , Borders(borders)
    {}

    bool operator<(const TEstimatedFeature& other) const {
        return std::tie(ModelEstimatedFeature) < std::tie(other.ModelEstimatedFeature);
    }

    bool operator==(const TEstimatedFeature& other) const {
        return std::tie(ModelEstimatedFeature) == std::tie(other.ModelEstimatedFeature);
    }

    bool operator!=(const TEstimatedFeature& other) const {
        return !(*this == other);
    }

    flatbuffers::Offset<NCatBoostFbs::TEstimatedFeature> FBSerialize(
        flatbuffers::FlatBufferBuilder& builder
    ) const;
    void FBDeserialize(const NCatBoostFbs::TEstimatedFeature* fbObj);
    Y_SAVELOAD_DEFINE(ModelEstimatedFeature, Borders);
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
