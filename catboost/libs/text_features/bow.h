#pragma once

#include "feature_calcer.h"

#include <catboost/libs/text_features/flatbuffers/feature_calcers.fbs.h>

namespace NCB {

    class TBagOfWordsCalcer final : public TTextFeatureCalcer {
    public:
        explicit TBagOfWordsCalcer(ui32 numTokens = 1)
        : NumTokens(numTokens)
        {}

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::BoW;
        }

        void Compute(const TText& text, TOutputFloatIterator outputFeaturesIterator) const override;

        ui32 FeatureCount() const override {
            return NumTokens;
        }

    protected:
        flatbuffers::Offset<NCatBoostFbs::TFeatureCalcer> SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const override;
        void LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer*) override;

        void SaveLargeParameters(IOutputStream*) const override {}
        void LoadLargeParameters(IInputStream*) override {}

    private:
        ui32 NumTokens;
    };
}
