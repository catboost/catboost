#include <library/cpp/text_processing/tokenizer/tokenizer.h>

#include <library/cpp/unittest/registar.h>

#include <util/generic/xrange.h>

using NTextProcessing::NTokenizer::TTokenizerOptions;
using NTextProcessing::NTokenizer::TTokenizer;
using NTextProcessing::NTokenizer::ETokenType;
using NTextProcessing::NTokenizer::ESeparatorType;
using NTextProcessing::NTokenizer::ETokenProcessPolicy;
using NTextProcessing::NTokenizer::ESubTokensPolicy;
using NTextProcessing::NTokenizer::TokenizerOptionsToJson;
using NTextProcessing::NTokenizer::JsonToTokenizerOptions;

static void AssertTokensEqual(const TVector<TString>& canonicalTokens, const TVector<TString>& tokens) {
    for (auto i : xrange(canonicalTokens.size())) {
        UNIT_ASSERT_VALUES_EQUAL(canonicalTokens[i], tokens[i]);
    }
}

Y_UNIT_TEST_SUITE(TokenizerTests) {

    Y_UNIT_TEST(TokenizerOptionsSaveLoadTest) {

        TTokenizerOptions options;
        options.Lowercasing = true;
        options.Lemmatizing = true;
        options.NumberProcessPolicy = ETokenProcessPolicy::Replace;
        options.SubTokensPolicy = ESubTokensPolicy::SeveralTokens;
        options.SeparatorType = ESeparatorType::ByDelimiter;
        options.Delimiter = "test";
        options.TokenTypes = {ETokenType::Unknown};
        options.Languages = {LANG_UNK};

        TTokenizerOptions saveLoadOptions = JsonToTokenizerOptions(TokenizerOptionsToJson(options));

        UNIT_ASSERT_VALUES_EQUAL(options.Lowercasing, saveLoadOptions.Lowercasing);
        UNIT_ASSERT_VALUES_EQUAL(options.Lemmatizing, saveLoadOptions.Lemmatizing);
        UNIT_ASSERT_VALUES_EQUAL(options.NumberProcessPolicy, saveLoadOptions.NumberProcessPolicy);
        UNIT_ASSERT_VALUES_EQUAL(options.SubTokensPolicy, saveLoadOptions.SubTokensPolicy);
        UNIT_ASSERT_VALUES_EQUAL(options.SeparatorType, saveLoadOptions.SeparatorType);
        UNIT_ASSERT_VALUES_EQUAL(options.Delimiter, saveLoadOptions.Delimiter);
        UNIT_ASSERT_VALUES_EQUAL(options.TokenTypes, saveLoadOptions.TokenTypes);
        UNIT_ASSERT_VALUES_EQUAL(options.Languages, saveLoadOptions.Languages);

    }

    Y_UNIT_TEST(TokenizerMainTest) {

        TTokenizerOptions options;
        options.SeparatorType = ESeparatorType::BySense;
        options.TokenTypes = {ETokenType::Word};
        options.Lowercasing = true;
        options.Lemmatizing = true;
        TTokenizer tokenizer(options);

        const TVector<TString> canonicalTokens = {"i", "love", "catboost"};
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i love catboost"));
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i loves catboost"));
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("I LoVe cATbOoSt"));
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i ,love    catboost!!!!"));
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i ,love 43 4  catboost!!!!"));

    }

    Y_UNIT_TEST(TokenizerNumbersTest) {

        TTokenizerOptions options;
        options.SeparatorType = ESeparatorType::BySense;
        options.TokenTypes = {ETokenType::Number};
        TTokenizer tokenizer(options);

        const TVector<TString> canonicalTokens = {"43", "4"};
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i 43 love catboost 4"));
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i ,love 43 4  catboost!!!!"));

    }

    Y_UNIT_TEST(TokenizerLowercasingTest) {

        TTokenizerOptions options;
        options.SeparatorType = ESeparatorType::BySense;
        options.TokenTypes = {ETokenType::Word};
        options.Lowercasing = false;
        options.Lemmatizing = false;
        TTokenizer tokenizer(options);

        const TVector<TString> canonicalTokens = {"I", "LoVe", "cATbOoSt"};
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("I LoVe cATbOoSt"));

    }

    Y_UNIT_TEST(TokenizerUnknownTest) {

        TString string = "i love cat6oost";

        TTokenizerOptions firstOptions;
        firstOptions.SeparatorType = ESeparatorType::BySense;
        firstOptions.TokenTypes = {ETokenType::Word};
        TTokenizer firstTokenizer(firstOptions);
        UNIT_ASSERT_VALUES_EQUAL(2, firstTokenizer.Tokenize(string).size());

        TTokenizerOptions secondOptions;
        secondOptions.SeparatorType = ESeparatorType::BySense;
        secondOptions.TokenTypes = {ETokenType::Word, ETokenType::Unknown};
        TTokenizer secondTokenizer(secondOptions);
        UNIT_ASSERT_VALUES_EQUAL(3, secondTokenizer.Tokenize(string).size());
        TVector<TString> tokens;
        TVector<ETokenType> tokenTypes;
        secondTokenizer.Tokenize(string, &tokens, &tokenTypes);
        UNIT_ASSERT_VALUES_EQUAL(ETokenType::Unknown, tokenTypes[2]);

    }

    Y_UNIT_TEST(TokenizerPerDelimiterTest) {

        TTokenizerOptions options;
        options.SeparatorType = ESeparatorType::ByDelimiter;
        options.Delimiter = ";6";
        options.Lowercasing = false;
        options.Lemmatizing = false;
        TTokenizer tokenizer(options);

        const TVector<TString> canonicalTokens = {"i", "love", "catboost"};
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i;6love;6catboost"));

    }

    Y_UNIT_TEST(TokenizerSubTokensTest) {

        {
            TTokenizerOptions options;
            options.SeparatorType = ESeparatorType::BySense;
            options.TokenTypes = {ETokenType::Word};
            TTokenizer tokenizer(options);
            const TVector<TString> canonicalTokens = {"А", "давай-ка", "сделаем", "это"};
            AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("А давай-ка сделаем это!"));
        }

        {
            TTokenizerOptions options;
            options.SeparatorType = ESeparatorType::BySense;
            options.TokenTypes = {ETokenType::Word};
            options.SubTokensPolicy = ESubTokensPolicy::SeveralTokens;
            TTokenizer tokenizer(options);
            const TVector<TString> canonicalTokens = {"А", "давай", "ка", "сделаем", "это"};
            AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("А давай-ка сделаем это!"));
        }

    }

    Y_UNIT_TEST(TokenizerLangest) {

        {
            TTokenizerOptions options;
            options.Lemmatizing = true;
            options.Languages = {LANG_RUS};
            options.SeparatorType = ESeparatorType::BySense;
            options.TokenTypes = {ETokenType::Word};
            TTokenizer tokenizer(options);
            AssertTokensEqual({"яблоко", "apples"}, tokenizer.Tokenize("яблоки apples"));
        }

        {
            TTokenizerOptions options;
            options.Lemmatizing = true;
            options.Languages = {LANG_ENG};
            options.SeparatorType = ESeparatorType::BySense;
            options.TokenTypes = {ETokenType::Word};
            TTokenizer tokenizer(options);
            AssertTokensEqual({"яблоки", "apple"}, tokenizer.Tokenize("яблоки apples"));
        }

    }

    Y_UNIT_TEST(TokenizerNumberProcessPolicyTest) {
        TVector<TTokenizerOptions> options(3);

        options[0].SeparatorType = ESeparatorType::BySense;
        options[0].TokenTypes = {ETokenType::Word, ETokenType::Number};
        options[0].NumberProcessPolicy = ETokenProcessPolicy::Replace;
        options[1].SeparatorType = ESeparatorType::ByDelimiter;
        options[1].NumberProcessPolicy = ETokenProcessPolicy::Replace;
        options[2].SeparatorType = ESeparatorType::ByDelimiter;
        options[2].NumberProcessPolicy = ETokenProcessPolicy::Replace;
        options[2].NumberToken = "lol";

        for (const auto& option : options) {
            TTokenizer tokenizer(option);
            AssertTokensEqual({"He", "has", option.NumberToken, "apples"}, tokenizer.Tokenize("He has 649 apples"));
        }

    }

    Y_UNIT_TEST(TokenizerSplitBySetTest) {

        TTokenizerOptions options;
        options.SeparatorType = ESeparatorType::ByDelimiter;
        options.SplitBySet = true;
        options.Delimiter = ",;";
        TTokenizer tokenizer(options);

        const TVector<TString> canonicalTokens = {"i", "love", "catboost"};
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i;love,catboost"));

    }

    Y_UNIT_TEST(TokenizerSkipEmptyTest) {

        TTokenizerOptions options;
        options.SeparatorType = ESeparatorType::ByDelimiter;
        options.SkipEmpty = true;
        options.Delimiter = "\t";
        TTokenizer tokenizer(options);

        const TVector<TString> canonicalTokens = {"i", "love", "catboost"};
        AssertTokensEqual(canonicalTokens, tokenizer.Tokenize("i\tlove\t\tcatboost"));

    }

}
