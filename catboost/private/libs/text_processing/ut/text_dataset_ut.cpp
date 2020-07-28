#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/private/libs/text_processing/text_dataset.h>
#include <catboost/private/libs/text_processing/text_column_builder.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;

Y_UNIT_TEST_SUITE(TestTextDataset) {
    const TTokenizerPtr tokenizer = CreateTokenizer();
    Y_UNIT_TEST(TestNaiveTokenizer) {
        TTokensWithBuffer tokens;

        tokenizer->Tokenize("", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.View.size(), 0);

        tokenizer->Tokenize(" ", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.View.size(), 0);

        tokenizer->Tokenize("         ", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.View.size(), 0);

        tokenizer->Tokenize("a", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.View.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(tokens.View[0], "a");

        tokenizer->Tokenize("a   a  ", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.View.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(tokens.View[0], "a");
        UNIT_ASSERT_VALUES_EQUAL(tokens.View[1], "a");
    }

    Y_UNIT_TEST(TestTextDatasetBuilder) {
        TVector<TString> text = {
            "hi",
            "ha ha",
            "ho ho ho",
            "hi ha ho",
            ""
        };
        NCatboostOptions::TTextColumnDictionaryOptions options;
        NTextProcessing::NDictionary::TDictionaryBuilderOptions builderOptions;
        builderOptions.OccurrenceLowerBound = 1;
        options.DictionaryBuilderOptions.Set(builderOptions);

        TDictionaryPtr dictionary = CreateDictionary(TIterableTextFeature(text), options, tokenizer);
        TTextColumnBuilder textColumnBuilder(tokenizer, dictionary, text.size());
        for (ui32 i = 0; i < text.size(); i++) {
            TStringBuf line = text[i];
            textColumnBuilder.AddText(i, line);
        }
        TTextDataSetPtr dataSet = MakeIntrusive<TTextDataSet>(
            TTextColumn::CreateOwning(textColumnBuilder.Build()),
            dictionary
        );

        const TTokenId hiId = dictionary->Apply("hi");
        const TTokenId haId = dictionary->Apply("ha");
        const TTokenId hoId = dictionary->Apply("ho");

        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(0).Find(hiId)->Count(), 1);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(1).Find(haId)->Count(), 2);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(2).Find(hoId)->Count(), 3);

        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(3).Find(hiId)->Count(), 1);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(3).Find(haId)->Count(), 1);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(3).Find(hoId)->Count(), 1);

        const TText &lastText = dataSet->GetText(4);
        UNIT_ASSERT_EQUAL(lastText.Find(hiId), lastText.end());
        UNIT_ASSERT_EQUAL(lastText.Find(haId), lastText.end());
        UNIT_ASSERT_EQUAL(lastText.Find(hoId), lastText.end());
    }
}
