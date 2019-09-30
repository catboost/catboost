#include "bow.h"

namespace NCB {
    TTextFeatureCalcerFactory::TRegistrator<TBagOfWordsCalcer>
        BagOfWordsRegistrator(EFeatureCalcerType::BoW);

    void TBagOfWordsCalcer::Compute(const NCB::TText& text, TOutputFloatIterator outputFeaturesIterator) const {
        ForEachActiveFeature(
            [&text, &outputFeaturesIterator](ui32 featureId) {
                *outputFeaturesIterator = text.Has(featureId);
                ++outputFeaturesIterator;
            }
        );
    }

    flatbuffers::Offset<NCatBoostFbs::TFeatureCalcer> TBagOfWordsCalcer::SaveParametersToFB(
        flatbuffers::FlatBufferBuilder& builder) const {
        using namespace NCatBoostFbs;

        auto bow = CreateTBoW(builder, NumTokens);

        return CreateTFeatureCalcer(
            builder,
            ActiveFeatureIndicesToFB(builder),
            TAnyFeatureCalcer_TBoW,
            bow.Union()
        );
    }

    void TBagOfWordsCalcer::LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer* calcer) {
        auto bow = calcer->FeatureCalcerImpl_as_TBoW();
        NumTokens = bow->NumTokens();
    }

}
