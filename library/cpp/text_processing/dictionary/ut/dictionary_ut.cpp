#include <library/cpp/text_processing/dictionary/bpe_builder.h>
#include <library/cpp/text_processing/dictionary/dictionary_builder.h>
#include <library/cpp/text_processing/dictionary/frequency_based_dictionary.h>
#include <library/cpp/text_processing/dictionary/mmap_frequency_based_dictionary.h>

#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/memory/blob.h>

using NTextProcessing::NDictionary::IDictionary;
using NTextProcessing::NDictionary::TBpeDictionary;
using NTextProcessing::NDictionary::TBpeDictionaryBuilder;
using NTextProcessing::NDictionary::TDictionary;
using NTextProcessing::NDictionary::TMMapBpeDictionary;
using NTextProcessing::NDictionary::TMMapDictionary;
using NTextProcessing::NDictionary::TDictionaryOptions;
using NTextProcessing::NDictionary::TDictionaryBuilderOptions;
using NTextProcessing::NDictionary::TDictionaryBuilder;
using NTextProcessing::NDictionary::ETokenLevelType;
using NTextProcessing::NDictionary::TTokenId;
using NTextProcessing::NDictionary::EUnknownTokenPolicy;

static auto GetApplyToAllDictsFunc(TDictionaryBuilder dictionaryBuilder, TBlob* blob, TVector<TIntrusivePtr<IDictionary>>* dicts) {
    auto dictionary = dictionaryBuilder.FinishBuilding();
    auto mmapDictionary = MakeIntrusive<TMMapDictionary>(dictionary);

    TStringStream stream;
    dictionary->Save(&stream);
    auto restoredDictionary = MakeIntrusive<TDictionary>();
    restoredDictionary->Load(&stream);

    mmapDictionary->Save(&stream);
    auto restoredMmapDictionary = MakeIntrusive<TMMapDictionary>();
    restoredMmapDictionary->Load(&stream);

    restoredMmapDictionary->Save(&stream);
    *blob = TBlob::FromStream(stream);
    auto restoredFromMemoryMmapDictionary = MakeIntrusive<TMMapDictionary>(blob->Data(), blob->Size());

    dicts->emplace_back(dictionary);
    dicts->emplace_back(mmapDictionary);
    dicts->emplace_back(restoredDictionary);
    dicts->emplace_back(restoredMmapDictionary);
    dicts->emplace_back(restoredFromMemoryMmapDictionary);

    return [=] (const std::function<void(IDictionary*)>& callback) {
        for (const auto& d : *dicts) {
            callback(d.Get());
        }
    };
}

static auto GetApplyToAllBpeDictsFunc(
    TIntrusivePtr<TDictionary> dictionary,
    TBpeDictionaryBuilder dictionaryBuilder,
    TBlob* blob,
    TVector<TIntrusivePtr<IDictionary>>* dicts
) {
    auto bpeDictionary = dictionaryBuilder.FinishBuilding();
    auto mmapDictionary = MakeIntrusive<TMMapBpeDictionary>(bpeDictionary);

    TStringStream stream;
    bpeDictionary->Save(&stream);
    auto restoredDictionary = MakeIntrusive<TBpeDictionary>(dictionary);
    restoredDictionary->Load(&stream);

    mmapDictionary->Save(&stream);
    auto restoredMmapDictionary = MakeIntrusive<TMMapBpeDictionary>(MakeIntrusive<TMMapDictionary>(dictionary));
    restoredMmapDictionary->Load(&stream);

    restoredMmapDictionary->Save(&stream);
    *blob = TBlob::FromStream(stream);
    auto restoredFromMemoryMmapDictionary = MakeIntrusive<TMMapBpeDictionary>(
        MakeIntrusive<TMMapDictionary>(dictionary),
        blob->Data(),
        blob->Size()
    );

    dicts->emplace_back(bpeDictionary);
    dicts->emplace_back(mmapDictionary);
    dicts->emplace_back(restoredDictionary);
    dicts->emplace_back(restoredMmapDictionary);
    dicts->emplace_back(restoredFromMemoryMmapDictionary);

    return [=] (const std::function<void(IDictionary*)>& callback) {
        for (const auto& d : *dicts) {
            callback(d.Get());
        }
    };
}

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

        TVector<TIntrusivePtr<IDictionary>> dicts;
        TBlob blob;
        auto checkForAll = GetApplyToAllDictsFunc(std::move(dictionaryBuilder), &blob, &dicts);

        checkForAll([](IDictionary* d){
            UNIT_ASSERT_VALUES_EQUAL(d->Size(), 3);
        });

        const auto unknownTokenId = dicts[0]->GetUnknownTokenId();
        checkForAll([&](IDictionary* d){
            UNIT_ASSERT_VALUES_EQUAL(d->Apply(secondSentence[1]), unknownTokenId);
        });

        TVector<TTokenId> tokenIdsWithOutUnknownTokens;
        dicts[0]->Apply(secondSentence, &tokenIdsWithOutUnknownTokens, EUnknownTokenPolicy::Skip);
        checkForAll([&](IDictionary* d){
            TVector<TTokenId> tokenIds;
            d->Apply(secondSentence, &tokenIds, EUnknownTokenPolicy::Skip);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithOutUnknownTokens, tokenIds);
        });

        TVector<TTokenId> tokenIdsWithUnknownTokens;
        dicts[0]->Apply(secondSentence, &tokenIdsWithUnknownTokens, EUnknownTokenPolicy::Insert);
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[0], dicts[0]->Apply(firstSentence[0]));
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[1], dicts[0]->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[2], dicts[0]->Apply(firstSentence[1]));
        UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens[3], dicts[0]->GetUnknownTokenId());
        checkForAll([&](IDictionary* d){
            TVector<TTokenId> tokenIds;
            d->Apply(secondSentence, &tokenIds, EUnknownTokenPolicy::Insert);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds.size(), 4);
            UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens, tokenIds);
        });

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

            TVector<TIntrusivePtr<IDictionary>> dicts;
            TBlob blob;
            auto checkForAll = GetApplyToAllDictsFunc(std::move(dictionaryBuilder), &blob, &dicts);

            checkForAll([](IDictionary* d){
                UNIT_ASSERT_VALUES_EQUAL(d->Size(), 2);
            });

            UNIT_ASSERT_VALUES_UNEQUAL(dicts[0]->Apply(tokens[0]), dicts[0]->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_UNEQUAL(dicts[0]->Apply(tokens[3]), dicts[0]->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_EQUAL(dicts[0]->Apply(tokens[5]), dicts[0]->GetUnknownTokenId());

            checkForAll([&](IDictionary* d){
                UNIT_ASSERT_VALUES_EQUAL(d->Apply(tokens[0]), dicts[0]->Apply(tokens[0]));
                UNIT_ASSERT_VALUES_EQUAL(d->Apply(tokens[3]), dicts[0]->Apply(tokens[3]));
                UNIT_ASSERT_VALUES_EQUAL(d->Apply(tokens[5]), dicts[0]->Apply(tokens[5]));
            });
        }

        {
            TDictionaryBuilderOptions dictionaryBuilderOptions;
            dictionaryBuilderOptions.MaxDictionarySize = 1;
            dictionaryBuilderOptions.OccurrenceLowerBound = 0;

            TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
            dictionaryBuilder.Add(tokens);

            TVector<TIntrusivePtr<IDictionary>> dicts;
            TBlob blob;
            auto checkForAll = GetApplyToAllDictsFunc(std::move(dictionaryBuilder), &blob, &dicts);

            checkForAll([](IDictionary* d){
                UNIT_ASSERT_VALUES_EQUAL(d->Size(), 1);
            });

            UNIT_ASSERT_VALUES_UNEQUAL(dicts[0]->Apply(tokens[0]), dicts[0]->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_EQUAL(dicts[0]->Apply(tokens[3]), dicts[0]->GetUnknownTokenId());
            UNIT_ASSERT_VALUES_EQUAL(dicts[0]->Apply(tokens[5]), dicts[0]->GetUnknownTokenId());

            checkForAll([&](IDictionary* d){
                UNIT_ASSERT_VALUES_EQUAL(d->Apply(tokens[0]), dicts[0]->Apply(tokens[0]));
                UNIT_ASSERT_VALUES_EQUAL(d->Apply(tokens[3]), dicts[0]->Apply(tokens[3]));
                UNIT_ASSERT_VALUES_EQUAL(d->Apply(tokens[5]), dicts[0]->Apply(tokens[5]));
            });
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

        TVector<TIntrusivePtr<IDictionary>> dicts;
        TBlob blob;
        auto checkForAll = GetApplyToAllDictsFunc(std::move(dictionaryBuilder), &blob, &dicts);

        UNIT_ASSERT_VALUES_UNEQUAL(dicts[0]->Apply("cat"), dicts[0]->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_EQUAL(dicts[0]->Apply("cot"), dicts[0]->GetUnknownTokenId());

        checkForAll([&](IDictionary* d){
            UNIT_ASSERT_VALUES_EQUAL(d->Apply("cat"), dicts[0]->Apply("cat"));
            UNIT_ASSERT_VALUES_EQUAL(d->Apply("cot"), dicts[0]->Apply("cot"));
        });

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

        TVector<TIntrusivePtr<IDictionary>> dicts;
        TBlob blob;
        auto checkForAll = GetApplyToAllDictsFunc(std::move(dictionaryBuilder), &blob, &dicts);

        auto applyFunc = [](const TVector<TString>& token, IDictionary* dictionary) -> TTokenId {
            TVector<TTokenId> tokenIds;
            dictionary->Apply(token, &tokenIds, EUnknownTokenPolicy::Insert);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds.size(), 1);
            return tokenIds[0];
        };

        UNIT_ASSERT_VALUES_UNEQUAL(applyFunc({"aaa", "", "a"}, dicts[0].Get()), dicts[0]->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_UNEQUAL(applyFunc({"a", "", "b"}, dicts[0].Get()), dicts[0]->GetUnknownTokenId());
        UNIT_ASSERT_VALUES_EQUAL(applyFunc({"b", "", "c"}, dicts[0].Get()), dicts[0]->GetUnknownTokenId());

        checkForAll([&](IDictionary* d){
            UNIT_ASSERT_VALUES_EQUAL(applyFunc({"aaa", "", "a"}, d), applyFunc({"aaa", "", "a"}, dicts[0].Get()));
            UNIT_ASSERT_VALUES_EQUAL(applyFunc({"a", "", "b"}, d), applyFunc({"a", "", "b"}, dicts[0].Get()));
            UNIT_ASSERT_VALUES_EQUAL(applyFunc({"b", "", "c"}, d), applyFunc({"b", "", "c"}, dicts[0].Get()));
        });

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

    Y_UNIT_TEST(BpeDictionaryMainTest) {

        TVector<TString> firstSentence = {"abc", "bcd", "bcd", "abc", "bcd"};
        TVector<TString> secondSentence = {"abc", "bcd", "abc", "cde"};

        TDictionaryOptions dictionaryOptions;
        TDictionaryBuilderOptions dictionaryBuilderOptions;
        dictionaryBuilderOptions.OccurrenceLowerBound = 0;
        dictionaryOptions.GramOrder = 1;
        dictionaryOptions.TokenLevelType = ETokenLevelType::Word;

        TDictionaryBuilder dictionaryBuilder(dictionaryBuilderOptions, dictionaryOptions);
        dictionaryBuilder.Add(firstSentence);
        auto dictionary = dictionaryBuilder.FinishBuilding();
        TBpeDictionaryBuilder bpeBuilder(/*numUnits=*/1, /*skipUnknown=*/false, dictionary);
        bpeBuilder.Add(firstSentence);

        TVector<TIntrusivePtr<IDictionary>> dicts;
        TBlob blob;
        auto checkForAll = GetApplyToAllBpeDictsFunc(dictionary, std::move(bpeBuilder), &blob, &dicts);

        checkForAll([](IDictionary* d){
            UNIT_ASSERT_VALUES_EQUAL(d->Size(), 3);
        });

        TVector<TTokenId> tokenIdsWithOutUnknownTokens;
        dicts[0]->Apply(secondSentence, &tokenIdsWithOutUnknownTokens);
        auto minUnusedTokenId = dicts[0]->GetMinUnusedTokenId();
        auto thirdTokenId = dictionary->Apply(secondSentence[2]);
        checkForAll([&](IDictionary* d){
            TVector<TTokenId> tokenIds;
            d->Apply(secondSentence, &tokenIds);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds[0], minUnusedTokenId - 1);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds[1], thirdTokenId);
            UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithOutUnknownTokens, tokenIds);
        });

        TVector<TTokenId> tokenIdsWithUnknownTokens;
        dicts[0]->Apply(secondSentence, &tokenIdsWithUnknownTokens, EUnknownTokenPolicy::Insert);
        auto unknownTokenId = dicts[0]->GetUnknownTokenId();
        checkForAll([&](IDictionary* d){
            TVector<TTokenId> tokenIds;
            d->Apply(secondSentence, &tokenIds, EUnknownTokenPolicy::Insert);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds.size(), 3);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds[0], minUnusedTokenId - 1);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds[1], thirdTokenId);
            UNIT_ASSERT_VALUES_EQUAL(tokenIds[2], unknownTokenId);
            UNIT_ASSERT_VALUES_EQUAL(tokenIdsWithUnknownTokens, tokenIds);
        });

    }
}
