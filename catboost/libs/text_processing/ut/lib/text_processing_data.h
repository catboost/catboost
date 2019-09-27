#pragma once

#include <catboost/libs/data_types/text.h>
#include <catboost/libs/text_processing/dictionary.h>
#include <catboost/libs/text_processing/tokenizer.h>

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
        TVector<TTokenizedTextFeature>* tokenizedFeatures,
        TVector<NCB::TDictionaryPtr>* dictionaries,
        NCB::TTokenizerPtr* tokenizer,
        TVector<ui32>* target,
        NCatboostOptions::TRuntimeTextOptions* textProcessingOptions
    );

}
