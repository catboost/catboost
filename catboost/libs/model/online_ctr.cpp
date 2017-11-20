#include "online_ctr.h"
#include "flatbuffers_serializer_helper.h"

flatbuffers::Offset<NCatBoostFbs::TModelCtrBase> TModelCtrBase::FBSerialize(
    TModelPartsCachingSerializer& serializer) const {
    auto featureCombinationOffset = serializer.GetOffset(Projection);
    return NCatBoostFbs::CreateTModelCtrBase(
        serializer.FlatbufBuilder,
        featureCombinationOffset,
        static_cast<NCatBoostFbs::ECtrType>(CtrType)
    );
}

flatbuffers::Offset<NCatBoostFbs::TModelCtr> TModelCtr::FBSerialize(
    TModelPartsCachingSerializer& serializer) const {
    return NCatBoostFbs::CreateTModelCtr(
        serializer.FlatbufBuilder,
        serializer.GetOffset(Base),
        TargetBorderIdx,
        PriorNum,
        PriorDenom,
        Shift,
        Scale
    );
}

flatbuffers::Offset<NCatBoostFbs::TFeatureCombination> TFeatureCombination::FBSerialize(TModelPartsCachingSerializer& serializer) const {
    auto catFeatures = serializer.FlatbufBuilder.CreateVector(CatFeatures);
    auto floatSplits = serializer.FlatbufBuilder.CreateVectorOfStructs<NCatBoostFbs::TFloatSplit>(
        BinFeatures.size(),
        [this](size_t i, NCatBoostFbs::TFloatSplit* ptr){
            *ptr = NCatBoostFbs::TFloatSplit(BinFeatures[i].FloatFeature, BinFeatures[i].Split);
        }
    );
    auto oneHotSplits = serializer.FlatbufBuilder.CreateVectorOfStructs<NCatBoostFbs::TOneHotSplit>(
        OneHotFeatures.size(),
        [this](size_t i, NCatBoostFbs::TOneHotSplit* ptr){
            *ptr = NCatBoostFbs::TOneHotSplit(OneHotFeatures[i].CatFeatureIdx, OneHotFeatures[i].Value);
        }
    );
    return NCatBoostFbs::CreateTFeatureCombination(
        serializer.FlatbufBuilder,
        catFeatures,
        floatSplits,
        oneHotSplits
    );
}
