#include "app_helpers.h"

#include <library/cpp/text_processing/dictionary/bpe_builder.h>
#include <library/cpp/text_processing/dictionary/dictionary_builder.h>
#include <library/cpp/text_processing/dictionary/options.h>
#include <library/cpp/text_processing/tokenizer/tokenizer.h>
#include <library/cpp/containers/flat_hash/flat_hash.h>

#include <util/generic/xrange.h>
#include <util/string/join.h>
#include <util/stream/file.h>
#include <util/stream/format.h>
#include <util/stream/length.h>
#include <util/system/fstat.h>
#include <util/system/hp_timer.h>

using NTextProcessing::NDictionary::EContextLevel;
using NTextProcessing::NDictionary::ETokenLevelType;
using NTextProcessing::NDictionary::EEndOfWordTokenPolicy;
using NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy;
using NTextProcessing::NDictionary::EDictionaryType;
using NTextProcessing::NDictionary::IDictionary;
using NTextProcessing::NDictionary::TBpeDictionary;
using NTextProcessing::NDictionary::TBpeDictionaryOptions;
using NTextProcessing::NDictionary::TBpeDictionaryBuilder;
using NTextProcessing::NDictionary::TDictionaryBuilder;
using NTextProcessing::NDictionary::TDictionaryBuilderOptions;
using NTextProcessing::NDictionary::TDictionaryOptions;
using NTextProcessing::NDictionary::TDictionary;
using NTextProcessing::NDictionary::TTokenId;
using NTextProcessing::NTokenizer::ETokenType;
using NTextProcessing::NTokenizer::TTokenizer;
using NTextProcessing::NTokenizer::TTokenizerOptions;

constexpr double REPORT_PROGRESS_INTERVAL_SECONDS = 1.0;

template <typename TVisitor>
static void ApplyFuncTotokenizedText(
    const TString& inputPath,
    const TTokenizerOptions& tokenizerOptions,
    bool useTokenizer,
    bool verbose,
    const TVisitor& visitor
) {
    TTokenizer tokenizer(tokenizerOptions);
    TVector<TString> tokens;


    ui32 dataSize = 0;
    if (verbose) {
        Y_ENSURE(inputPath != "-", "verbose parameter isn't supported for stdin input.");
        dataSize = GetFileLength(inputPath);
    }

    TFileInput input(inputPath);
    TCountingInput countingInput(&input);
    TString line;
    THPTimer watch;
    double lastReportProgressTime = watch.Passed();
    while (countingInput.ReadLine(line)) {
        if (useTokenizer) {
            tokenizer.Tokenize(line, &tokens);
            visitor(tokens);
        } else {
            visitor({line});
        }

        if (verbose) {
            const double passedTime = watch.Passed();
            if (passedTime - lastReportProgressTime > REPORT_PROGRESS_INTERVAL_SECONDS) {
                const double progress = (double)(countingInput.Counter()) / dataSize;
                const double leftTime = passedTime / progress - passedTime;
                Cerr << "[" << Prec(progress * 100, PREC_POINT_DIGITS, 1) << "%]\t";
                Cerr << "Time passed: " << HumanReadable(TDuration::Seconds(passedTime)) << "\t";
                Cerr << "Time left: " << HumanReadable(TDuration::Seconds(leftTime)) << Endl;
                lastReportProgressTime = passedTime;
            }
        }
    }

    const double passedTime = watch.Passed();
    if (verbose) {
        Cerr << "Time passed: " << HumanReadable(TDuration::Seconds(passedTime)) << "\n";
    }
}

TIntrusivePtr<TDictionary> NTextProcessing::NDictionary::BuildDictionary(
    const TString& inputFilePath,
    const TDictionaryBuilderOptions& dictionaryBuilderOptions,
    const TDictionaryOptions& dictionaryOptions,
    const TTokenizerOptions& tokenizerOptions,
    bool useTokenizer,
    bool verbose
) {
    TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
    ApplyFuncTotokenizedText(
        inputFilePath,
        tokenizerOptions,
        useTokenizer,
        verbose,
        [&](const TVector<TString>& tokens) {
            dictionaryBuilder.Add(tokens, /*weight*/1);
        }
    );
    return dictionaryBuilder.FinishBuilding();
}

static TIntrusivePtr<TBpeDictionary> BuildBpeWord(
    const TDictionaryBuilderOptions& dictionaryBuilderOptions,
    const TDictionaryOptions& dictionaryOptions,
    const TBpeDictionaryOptions& bpeOptions,
    const TString& inputFilePath,
    const TTokenizerOptions& tokenizerOptions,
    bool useTokenizer,
    bool verbose
) {
    if (verbose) {
        Cerr << "Stage [1/2]: Dictionary building\n";
    }

    auto dictionary = BuildDictionary(
        inputFilePath,
        dictionaryBuilderOptions,
        dictionaryOptions,
        tokenizerOptions,
        useTokenizer,
        verbose
    );

    TBpeDictionaryBuilder bpeBuilder(bpeOptions.NumUnits, bpeOptions.SkipUnknown, dictionary);

    if (verbose) {
        Cerr << "Stage [2/2]: Bpe building\n";
    }

    ApplyFuncTotokenizedText(
        inputFilePath,
        tokenizerOptions,
        useTokenizer,
        verbose,
        [&](const TVector<TString>& tokens) {
            bpeBuilder.Add(tokens, /*weight*/1);
        }
    );

    return bpeBuilder.FinishBuilding();
}

static TIntrusivePtr<TBpeDictionary> BuildBpeLetter(
    const TDictionaryBuilderOptions& dictionaryBuilderOptions,
    const TDictionaryOptions& dictionaryOptions,
    const TBpeDictionaryOptions& bpeOptions,
    const TString& inputFilePath,
    const TTokenizerOptions& tokenizerOptions,
    bool useTokenizer,
    bool verbose
) {
    if (verbose) {
        Cerr << "Stage [1/2]: Dictionary building\n";
    }

    NFH::TFlatHashMap<TString, ui64> tokenCounts;
    TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);

    ApplyFuncTotokenizedText(
        inputFilePath,
        tokenizerOptions,
        useTokenizer,
        verbose,
        [&](const TVector<TString>& tokens) {
            dictionaryBuilder.Add(tokens, /*weight*/1);
            for (const auto& token : tokens) {
                ++tokenCounts[token];
            }
        }
    );

    auto dictionary = dictionaryBuilder.FinishBuilding();

    if (verbose) {
        Cerr << "Stage [2/2]: Bpe building\n";
    }

    TBpeDictionaryBuilder bpeBuilder(
        bpeOptions.NumUnits,
        bpeOptions.SkipUnknown,
        dictionary);

    for (const auto& [token, count] : tokenCounts) {
        bpeBuilder.Add(TVector<TStringBuf>({token}), count);
    }

    return bpeBuilder.FinishBuilding();
}

TIntrusivePtr<TBpeDictionary> NTextProcessing::NDictionary::BuildBpe(
    const TString& inputFilePath,
    const TDictionaryBuilderOptions& dictionaryBuilderOptions,
    const TDictionaryOptions& dictionaryOptions,
    const TBpeDictionaryOptions& bpeOptions,
    const TTokenizerOptions& tokenizerOptions,
    bool useTokenizer,
    bool verbose
) {
    if (dictionaryOptions.TokenLevelType == ETokenLevelType::Word || bpeOptions.ContextLevel == EContextLevel::Sentence) {
        return BuildBpeWord(
            dictionaryBuilderOptions,
            dictionaryOptions,
            bpeOptions,
            inputFilePath,
            tokenizerOptions,
            useTokenizer,
            verbose
        );
    } else {
        Y_ASSERT(dictionaryOptions.TokenLevelType == ETokenLevelType::Letter);
        return BuildBpeLetter(
            dictionaryBuilderOptions,
            dictionaryOptions,
            bpeOptions,
            inputFilePath,
            tokenizerOptions,
            useTokenizer,
            verbose
        );
    }
}

void NTextProcessing::NDictionary::ApplyDictionaryToFile(
    const TString& inputPath,
    const TVector<TIntrusivePtr<IDictionary>>& dictionaries,
    const TString& outputPath,
    const TString& outputDelimiter,
    const TTokenizerOptions& tokenizerOptions,
    bool useTokenizer,
    bool verbose
) {
    TFileOutput output(outputPath);
    TVector<TTokenId> tokenIds;
    ApplyFuncTotokenizedText(
        inputPath,
        tokenizerOptions,
        useTokenizer,
        verbose,
        [&](const TVector<TString>& tokens) {
            for (auto i : xrange(dictionaries.size())) {
                dictionaries[i]->Apply(tokens, &tokenIds);
                output << JoinSeq(outputDelimiter, tokenIds);
                if (i == dictionaries.size() - 1) {
                    output << '\n';
                    continue;
                }
                output << outputDelimiter;
            }
        }
    );
}
