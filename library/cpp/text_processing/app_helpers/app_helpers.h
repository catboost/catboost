#pragma once

#include <library/cpp/text_processing/dictionary/bpe_dictionary.h>
#include <library/cpp/text_processing/tokenizer/options.h>

namespace NTextProcessing::NDictionary {

    TIntrusivePtr<TDictionary> BuildDictionary(
        const TString& inputFilePath,
        const TDictionaryBuilderOptions& dictionaryBuilderOptions,
        const TDictionaryOptions& dictionaryOptions,
        const NTokenizer::TTokenizerOptions& tokenizerOptions,
        bool useTokenizer,
        bool verbose
    );

    TIntrusivePtr<TBpeDictionary> BuildBpe(
        const TString& inputFilePath,
        const TDictionaryBuilderOptions& dictionaryBuilderOptions,
        const TDictionaryOptions& dictionaryOptions,
        const TBpeDictionaryOptions& bpeOptions,
        const NTokenizer::TTokenizerOptions& tokenizerOptions,
        bool useTokenizer,
        bool verbose
    );

    void ApplyDictionaryToFile(
        const TString& inputPath,
        const TVector<TIntrusivePtr<IDictionary>>& dictionaries,
        const TString& outputPath,
        const TString& outputDelimiter,
        const NTokenizer::TTokenizerOptions& tokenizerOptions,
        bool useTokenizer,
        bool verbose
    );

}
