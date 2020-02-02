#include "bow.h"

namespace NCB {
    TTextFeatureCalcerFactory::TRegistrator<TBagOfWordsCalcer>
        BagOfWordsRegistrator(EFeatureCalcerType::BoW);

    void TBagOfWordsCalcer::Compute(const NCB::TText& text, TOutputFloatIterator outputFeaturesIterator) const {
        auto textIterator = text.begin();
        for (ui32 activeFeatureId : GetActiveFeatureIndices()) {
            while (textIterator != text.end() && static_cast<ui32>(textIterator->Token()) < activeFeatureId) {
                ++textIterator;
            }

            if (textIterator == text.end() || static_cast<ui32>(textIterator->Token()) > activeFeatureId) {
                *outputFeaturesIterator = 0;
            } else {
                *outputFeaturesIterator = 1;
            }
            ++outputFeaturesIterator;
        }
    }

    TTextFeatureCalcer::TFeatureCalcerFbs TBagOfWordsCalcer::SaveParametersToFB(
        flatbuffers::FlatBufferBuilder& builder) const {
        using namespace NCatBoostFbs;

        auto bow = CreateTBoW(builder, NumTokens);
        return TFeatureCalcerFbs(TAnyFeatureCalcer_TBoW, bow.Union());
    }

    void TBagOfWordsCalcer::LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer* calcer) {
        auto bow = calcer->FeatureCalcerImpl_as_TBoW();
        NumTokens = bow->NumTokens();
    }

}
