#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/private/libs/text_processing/text_dataset.h>
#include <catboost/private/libs/text_processing/text_column_builder.h>

#include <library/unittest/registar.h>


using namespace NCB;

Y_UNIT_TEST_SUITE(TestTextDataset) {
    const TTokenizerPtr tokenizer = CreateTokenizer();
    Y_UNIT_TEST(TestNaiveTokenizer) {
        TVector<TStringBuf> tokens;

        tokenizer->Tokenize("", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.size(), 0);

        tokenizer->Tokenize(" ", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.size(), 0);

        tokenizer->Tokenize("         ", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.size(), 0);

        tokenizer->Tokenize("a", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(tokens[0], "a");

        tokenizer->Tokenize("a   a  ", &tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(tokens[0], "a");
        UNIT_ASSERT_VALUES_EQUAL(tokens[1], "a");
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

        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(0).at(hiId), 1);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(1).at(haId), 2);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(2).at(hoId), 3);

        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(3).at(hiId), 1);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(3).at(haId), 1);
        UNIT_ASSERT_VALUES_EQUAL(dataSet->GetText(3).at(hoId), 1);

        const TText &lastText = dataSet->GetText(4);
        UNIT_ASSERT_EQUAL(lastText.find(hiId), lastText.end());
        UNIT_ASSERT_EQUAL(lastText.find(haId), lastText.end());
        UNIT_ASSERT_EQUAL(lastText.find(hoId), lastText.end());
    }
}
