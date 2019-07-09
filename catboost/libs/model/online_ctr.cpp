#include "online_ctr.h"
#include "flatbuffers_serializer_helper.h"

#include <catboost/libs/model/flatbuffers/ctr_data.fbs.h>

flatbuffers::Offset<NCatBoostFbs::TModelCtrBase> TModelCtrBase::FBSerialize(
    TModelPartsCachingSerializer& serializer) const {
    auto featureCombinationOffset = serializer.GetOffset(Projection);
    return NCatBoostFbs::CreateTModelCtrBase(
        serializer.FlatbufBuilder,
        featureCombinationOffset,
        static_cast<NCatBoostFbs::ECtrType>(CtrType),
        TargetBorderClassifierIdx
    );
}

void TModelCtrBase::FBDeserialize(const NCatBoostFbs::TModelCtrBase* fbObj) {
    Projection.Clear();
    if (fbObj == nullptr) {
        return;
    }
    Projection.FBDeserialize(fbObj->FeatureCombination());
    CtrType = static_cast<ECtrType>(fbObj->CtrType());
    TargetBorderClassifierIdx = fbObj->TargetBorderClassifierIdx();
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

void TModelCtr::FBDeserialize(const NCatBoostFbs::TModelCtr* fbObj) {
    Base.FBDeserialize(fbObj->Base());
    TargetBorderIdx = fbObj->TargetBorderIdx();
    PriorNum = fbObj->PriorNum();
    PriorDenom = fbObj->PriorDenom();
    Shift = fbObj->Shift();
    Scale = fbObj->Scale();
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

void TFeatureCombination::FBDeserialize(const NCatBoostFbs::TFeatureCombination* fbObj) {
    Clear();
    if (fbObj == nullptr) {
        return;
    }
    if (fbObj->CatFeatures() && fbObj->CatFeatures()->size() != 0) {
        CatFeatures.assign(fbObj->CatFeatures()->begin(), fbObj->CatFeatures()->end());
    }
    if (fbObj->FloatSplits() && fbObj->FloatSplits()->size() != 0) {
        for (const auto fbSplit : *fbObj->FloatSplits()) {
            TFloatSplit split{fbSplit->Index(), fbSplit->Border()};
            BinFeatures.push_back(split);
        }
    }
    if (fbObj->OneHotSplits() && fbObj->OneHotSplits()->size() != 0) {
        for (const auto fbSplit : *fbObj->OneHotSplits()) {
            TOneHotSplit split{fbSplit->Index(), fbSplit->Value()};
            OneHotFeatures.push_back(split);
        }
    }
}
