#include "bow.h"

namespace NCB {
    TTextFeatureCalcerFactory::TRegistrator<TBagOfWordsCalcer>
        BagOfWordsRegistrator(EFeatureCalcerType::BoW);

    void TBagOfWordsCalcer::Compute(const NCB::TText& text, TOutputFloatIterator outputFeaturesIterator) const {
        TVector<float> features;
        features.yresize(NumTokens);

        for (ui32 tokenId = 0; tokenId < NumTokens; tokenId++, ++outputFeaturesIterator) {
            *outputFeaturesIterator = text.Has(tokenId);
        }
    }

    flatbuffers::Offset<NCatBoostFbs::TFeatureCalcer> TBagOfWordsCalcer::SaveParametersToFB(
        flatbuffers::FlatBufferBuilder& builder) const {
        using namespace NCatBoostFbs;

        auto bow = CreateTBoW(builder, NumTokens);

        return CreateTFeatureCalcer(builder, TAnyFeatureCalcer_TBoW, bow.Union());
    }

    void TBagOfWordsCalcer::LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer* calcer) {
        auto bow = calcer->FeatureCalcerImpl_as_TBoW();
        NumTokens = bow->NumTokens();
    }

}
