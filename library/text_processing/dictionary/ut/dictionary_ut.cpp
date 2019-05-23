#include <library/text_processing/dictionary/dictionary_builder.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/unittest/registar.h>

using NTextProcessing::NDictionary::IDictionary;
using NTextProcessing::NDictionary::TDictionaryOptions;
using NTextProcessing::NDictionary::TDictionaryBuilderOptions;
using NTextProcessing::NDictionary::TDictionaryBuilder;
using NTextProcessing::NDictionary::ETokenLevelType;
using NTextProcessing::NDictionary::TTokenId;
using NTextProcessing::NDictionary::EUnknownTokenPolicy;

Y_UNIT_TEST_SUITE(DictionaryTests) {

    Y_UNIT_TEST(DictionaryMainTest) {

        TVector<TString> firstSentence = {"i", "like", "apples"};
        TVector<TString> secondSentence = {"i", "don't", "like", "pears"};

        TDictionaryOptions dictionaryOptions;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(firstSentence);
        const auto dictionary = dictionaryBuilder.FinishBuilding();
        UNIT_ASSERT_VALUES_EQUAL(dictionary->Size(), 3);

        UNIT_ASSERT_VALUES_EQUAL(dictionary->Apply(secondSentence[1]), dictionary->GetUnknownTokenId());

        TVector<TTokenId> tokenIdsWithOutUnknownTokens;
        dictionary->Apply(secondSentence, &tokenIdsWithOutUnknownTokens, EUnknownTokenPolicy::Skip);
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithOutUnknownTokens.size(), 2);

        TVector<TTokenId> tokenIdsWithUnknownTokens;
        dictionary->Apply(secondSentence, &tokenIdsWithUnknownTokens, EUnknownTokenPolicy::Insert);
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens.size(), 4);

        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[0], dictionary->Apply(firstSentence[0]));
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[1], dictionary->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[2], dictionary->Apply(firstSentence[1]));
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[3], dictionary->GetUnknownTokenId());

    }

    Y_UNIT_TEST(DictionaryOptionsTest) {

        TVector<TString> tokens = {"a", "a", "a", "b", "b", "c"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;

        {
            TDictionaryBuilderOptions dictionaryBuilderOptions;
            dictionaryBuilderOptions.OccurrenceLowerBound = 2;

            TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
            dictionaryBuilder.Add(tokens);
            const auto dictionary = dictionaryBuilder.FinishBuilding();
            UNIT_ASSERT_VALUES_EQUAL(dictionary->Size(), 2);

            UNIT_ASSERT_VALUES_UNEQUAL(dictionary->Apply(tokens[0]), dictionary->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_UNEQUAL(dictionary->Apply(tokens[3]), dictionary->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_EQUAL(dictionary->Apply(tokens[5]), dictionary->GetUnknownTokenId());
        }

        {
            TDictionaryBuilderOptions dictionaryBuilderOptions;
            dictionaryBuilderOptions.MaxDictionarySize = 1;
            dictionaryBuilderOptions.OccurrenceLowerBound = 0;

            TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
            dictionaryBuilder.Add(tokens);
            const auto dictionary = dictionaryBuilder.FinishBuilding();
            UNIT_ASSERT_VALUES_EQUAL(dictionary->Size(), 1);

            UNIT_ASSERT_VALUES_UNEQUAL(dictionary->Apply(tokens[0]), dictionary->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_EQUAL(dictionary->Apply(tokens[3]), dictionary->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_EQUAL(dictionary->Apply(tokens[5]), dictionary->GetUnknownTokenId());
        }
    }

    Y_UNIT_TEST(DictionaryLetterTrigramTest) {

        TVector<TString> tokens = {"i", "love", "catboost"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 3;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Letter;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        const auto dictionary = dictionaryBuilder.FinishBuilding();

        UNIT_ASSERT_VALUES_UNEQUAL(dictionary->Apply("cat"), dictionary->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_EQUAL(dictionary->Apply("cot"), dictionary->GetUnknownTokenId());
    }

    Y_UNIT_TEST(DictionaryWordSkipBigramTest) {

        TVector<TString> tokens = {"aaa", "aa", "a", "bb", "b", "c"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 2;
        dictionaryOptions.SkipStep = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        const auto dictionary = dictionaryBuilder.FinishBuilding();

        auto applyFunc = [&](const TVector<TString>& token) -> TTokenId {
            TVector<TTokenId> tokenIds;
            dictionary->Apply(token, &tokenIds, EUnknownTokenPolicy::Insert);
            return tokenIds[0];
        };

        UNIT_ASSERT_VALUES_UNEQUAL(applyFunc({"aaa", "", "a"}), dictionary->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_UNEQUAL(applyFunc({"a", "", "b"}), dictionary->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_EQUAL(applyFunc({"b", "", "c"}), dictionary->GetUnknownTokenId());
    }

    Y_UNIT_TEST(DictionarySaveLoadTest) {

        TVector<TString> tokens = {"aaa", "aa", "a", "bb", "b", "c"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 2;
        dictionaryOptions.SkipStep = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        const auto dictionaryPtr = dictionaryBuilder.FinishBuilding();

        TStringStream serializedDictionary;
        dictionaryPtr->Save(&serializedDictionary);
        const auto newDictionaryPtr = IDictionary::Load(&serializedDictionary);

        auto applyFunc = [&](const TVector<TString>& token) -> TTokenId {
            TVector<TTokenId> tokenIds;
            newDictionaryPtr->Apply(token, &tokenIds, EUnknownTokenPolicy::Insert);
            return tokenIds[0];
        };

        UNIT_ASSERT_VALUES_UNEQUAL(applyFunc({"aaa", "", "a"}), newDictionaryPtr->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_UNEQUAL(applyFunc({"a", "", "b"}), newDictionaryPtr->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_EQUAL(applyFunc({"b", "", "c"}), newDictionaryPtr->GetUnknownTokenId());
    }

    Y_UNIT_TEST(DictionaryTopTokensTest) {

        TVector<TString> tokens = {"a", "a", "a", "b", "b", "c", "c", "c", "c"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;

        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        const auto dictionaryPtr = dictionaryBuilder.FinishBuilding();
        UNIT_ASSERT_VALUES_EQUAL(dictionaryPtr->Size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(dictionaryPtr->GetTopTokens()[0], "c");
        UNIT_ASSERT_VALUES_EQUAL(dictionaryPtr->GetTopTokens()[1], "a");
        UNIT_ASSERT_VALUES_EQUAL(dictionaryPtr->GetTopTokens()[2], "b");

        TStringStream serializedDictionary;
        dictionaryPtr->Save(&serializedDictionary);
        const auto newDictionaryPtr = IDictionary::Load(&serializedDictionary);
        UNIT_ASSERT_VALUES_EQUAL(newDictionaryPtr->Size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(newDictionaryPtr->GetTopTokens()[0], "c");
        UNIT_ASSERT_VALUES_EQUAL(newDictionaryPtr->GetTopTokens()[1], "a");
        UNIT_ASSERT_VALUES_EQUAL(newDictionaryPtr->GetTopTokens()[2], "b");
    }

    Y_UNIT_TEST(DictionaryClearStatsDataTest) {

        TVector<TString> tokens = {"a", "a", "a", "b", "b", "c", "c", "c", "c"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;

        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        const auto dictionaryPtr = dictionaryBuilder.FinishBuilding();
        UNIT_ASSERT_VALUES_EQUAL(dictionaryPtr->Size(), 3);

        dictionaryPtr->ClearStatsData();
        UNIT_ASSERT_EXCEPTION(dictionaryPtr->GetTopTokens(), yexception);

        TStringStream serializedDictionary;
        dictionaryPtr->Save(&serializedDictionary);
        const auto newDictionaryPtr = IDictionary::Load(&serializedDictionary);
        UNIT_ASSERT_VALUES_EQUAL(newDictionaryPtr->Size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(newDictionaryPtr->GetTopTokens().size(), 3);
        UNIT_ASSERT_EXCEPTION(newDictionaryPtr->GetCount(1), yexception);
    }

    Y_UNIT_TEST(DictionaryGetTokenByTokenIdTest) {

        TVector<TString> tokens = {"a", "a", "a", "b", "b", "c", "c", "c", "c"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        const auto dictionaryPtr = dictionaryBuilder.FinishBuilding();
        UNIT_ASSERT_VALUES_EQUAL(tokens[0], dictionaryPtr->GetToken(dictionaryPtr->Apply(tokens[0])));
        UNIT_ASSERT_VALUES_EQUAL(tokens[3], dictionaryPtr->GetToken(dictionaryPtr->Apply(tokens[3])));
        UNIT_ASSERT_VALUES_EQUAL(tokens[5], dictionaryPtr->GetToken(dictionaryPtr->Apply(tokens[5])));
        UNIT_ASSERT_EXCEPTION(dictionaryPtr->GetToken(1000), yexception);

    }

    Y_UNIT_TEST(DictionaryGetCountByTokenIdTest) {

        TVector<TString> tokens = {"a", "a", "a", "b", "b", "c", "c", "c", "c"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        const auto dictionaryPtr = dictionaryBuilder.FinishBuilding();
        UNIT_ASSERT_VALUES_EQUAL(3, dictionaryPtr->GetCount(dictionaryPtr->Apply(tokens[0])));
        UNIT_ASSERT_VALUES_EQUAL(2, dictionaryPtr->GetCount(dictionaryPtr->Apply(tokens[3])));
        UNIT_ASSERT_VALUES_EQUAL(4, dictionaryPtr->GetCount(dictionaryPtr->Apply(tokens[5])));
        UNIT_ASSERT_EXCEPTION(dictionaryPtr->GetCount(1000), yexception);

    }

    Y_UNIT_TEST(DictionarySaveFailOnTokenWithNewLineSymbolTest) {

        TVector<TString> tokens = {"i", "l\nke", "apples"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        TStringStream serializedDictionary;
        UNIT_ASSERT_EXCEPTION(dictionaryBuilder.FinishBuilding()->Save(&serializedDictionary), yexception);

    }

    Y_UNIT_TEST(DictionarySaveFailOnTokenWithSpaceSymbolTest) {

        TVector<TString> tokens = {"i", "l ke", "apples"};

        TDictionaryOptions dictionaryOptions;
        dictionaryOptions.GramOrder = 2;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(tokens);
        TStringStream serializedDictionary;
        UNIT_ASSERT_EXCEPTION(dictionaryBuilder.FinishBuilding()->Save(&serializedDictionary), yexception);

    }

}
