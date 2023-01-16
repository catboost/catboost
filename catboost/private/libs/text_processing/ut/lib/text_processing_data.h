#pragma once

#include <catboost/private/libs/data_types/text.h>
#include <catboost/private/libs/text_processing/text_digitizers.h>

#include <util/generic/fwd.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    class TRuntimeTextOptions;
}

namespace NCBTest {
    using TTextFeature = TVector<TString>;
    using TTokenizedTextFeature = TVector<NCB::TText>;

    void CreateTextDataForTest(
        TVector<TTextFeature>* features,
        TMap<ui32, TTokenizedTextFeature>* tokenizedFeatures,
        TVector<ui32>* target,
        NCB::TTextDigitizers* textDigitizers,
        NCatboostOptions::TTextProcessingOptions* textProcessingOptions
    );

}
