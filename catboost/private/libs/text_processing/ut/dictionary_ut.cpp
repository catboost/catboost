#include <catboost/private/libs/text_processing/dictionary.h>

#include <library/cpp/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestDictionary) {
    Y_UNIT_TEST(TestBasicProperties) {
        using namespace NCB;
        using namespace NCatboostOptions;

        TVector<TString> text = {
            "a", "a", "a", "a", "b", "b", "c", "c", "c", "d"
        };

        TTokenizerPtr tokenizer = CreateTokenizer();
        TDictionaryBuilderOptions dictionaryBuilderOptions{2, -1};
        auto dictionary = CreateDictionary(
            TIterableTextFeature(text),
            NCatboostOptions::TTextColumnDictionaryOptions(
                "dictionary",
                TDictionaryOptions(),
                dictionaryBuilderOptions
            ),
            tokenizer
        );

        UNIT_ASSERT_EQUAL(0, dictionary->Apply("a"));
        UNIT_ASSERT_EQUAL(1, dictionary->Apply("c"));
        UNIT_ASSERT_EQUAL(2, dictionary->Apply("b"));
        UNIT_ASSERT_EQUAL(dictionary->GetUnknownTokenId(), dictionary->Apply("d"));

        TText textExample = TText(/* tokenIds */{
            0, 0,
            1, 1, 1,
            2
        });
        UNIT_ASSERT_VALUES_EQUAL(
            textExample,
            dictionary->Apply({"a", "a", "b", "c", "c", "c"})
        );

        TVector<TTokenId> topTokens = dictionary->GetTopTokens(10);
        UNIT_ASSERT_EQUAL(3, topTokens.size());
        UNIT_ASSERT_EQUAL(0, topTokens[0]);
        UNIT_ASSERT_EQUAL(1, topTokens[1]);
        UNIT_ASSERT_EQUAL(2, topTokens[2]);
    }
}
