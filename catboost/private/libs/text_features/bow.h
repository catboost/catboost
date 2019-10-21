#pragma once

#include "feature_calcer.h"

#include <catboost/private/libs/text_features/flatbuffers/feature_calcers.fbs.h>

namespace NCB {

    class TBagOfWordsCalcer final : public TTextFeatureCalcer {
    public:
        explicit TBagOfWordsCalcer(const TGuid& calcerId = CreateGuid(), ui32 numTokens = 1)
        : TTextFeatureCalcer(numTokens, calcerId)
        , NumTokens(numTokens)
        {}

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::BoW;
        }

        void Compute(const TText& text, TOutputFloatIterator outputFeaturesIterator) const override;

    protected:
        TTextFeatureCalcer::TFeatureCalcerFbs SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const override;
        void LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer*) override;

        void SaveLargeParameters(IOutputStream*) const override {}
        void LoadLargeParameters(IInputStream*) override {}

    private:
        ui32 NumTokens;
    };
}
