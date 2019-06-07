#pragma once

#include "enums.h"
#include "option.h"

#include <library/json/json_value.h>
#include <library/text_processing/dictionary/options.h>
#include <util/generic/vector.h>
#include <util/generic/strbuf.h>
#include <util/generic/map.h>
#include <utility>


namespace NCatboostOptions {
    class TTextProcessingOptions {
    public:
        TTextProcessingOptions();
        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextProcessingOptions& rhs) const;
        bool operator!=(const TTextProcessingOptions& rhs) const;

    public:
        TOption<NTextProcessing::NDictionary::TDictionaryOptions> DictionaryOptions;
        TOption<NTextProcessing::NDictionary::TDictionaryBuilderOptions> DictionaryBuilderOptions;
        TOption<ETokenizerType> TokenizerType;
    };

    std::pair<TStringBuf, NJson::TJsonValue> ParsePerTextFeatureProcessing(TStringBuf description);

    class TTextProcessingOptionCollection {
    public:
        TTextProcessingOptionCollection();

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextProcessingOptionCollection& rhs) const;
        bool operator!=(const TTextProcessingOptionCollection& rhs) const;

        const TTextProcessingOptions& GetFeatureTextProcessing(ui32 featureId) const;

    private:
        const TTextProcessingOptions DefaultTextProcessing = TTextProcessingOptions();
        TOption<TMap<ui32, TTextProcessingOptions>> FeatureIdToTextProcessing;
    };

    class TTextFeatureOptions {
    public:
        TTextFeatureOptions();
        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextFeatureOptions& rhs) const;
        bool operator!=(const TTextFeatureOptions& rhs) const;

    public:
        TOption<TVector<EFeatureEstimatorType>> FeatureEstimators;
    };
}
