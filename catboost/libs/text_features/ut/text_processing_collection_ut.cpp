#include <catboost/libs/text_features/text_processing_collection.h>

#include <catboost/libs/text_features/bow.h>
#include <catboost/libs/text_features/naive_bayesian.h>
#include <catboost/libs/text_processing/embedding.h>
#include <catboost/libs/text_processing/text_column_builder.h>
#include <catboost/libs/options/text_feature_options.h>

#include <library/unittest/registar.h>

using namespace NCB;
using namespace NCatboostOptions;

Y_UNIT_TEST_SUITE(TestTextProcessingCollection) {
    using TTextFeature = TVector<TString>;
    using TTokenizedTextFeature = TVector<TText>;

    static TIntrusivePtr<TMultinomialNaiveBayes> CreateBayes(
        const TTokenizedTextFeature& features,
        const TVector<ui32>& target,
        ui32 numClasses) {

        auto naiveBayes = MakeIntrusive<TMultinomialNaiveBayes>(numClasses);
        TNaiveBayesVisitor bayesVisitor;

        for (ui32 sampleId: xrange(features.size())) {
            bayesVisitor.Update(target[sampleId], features[sampleId], naiveBayes.Get());
        }

        return naiveBayes;
    }

    static TIntrusivePtr<TBagOfWordsCalcer> CreateBoW(const TDictionaryPtr& dictionaryPtr) {
        return MakeIntrusive<TBagOfWordsCalcer>(dictionaryPtr->Size());
    }

    struct TTextProcessingIdx {
    public:
        ui32 featureIdx;
        ui32 dictionaryIdx;
        ui32 calcerIdx;
    };

    static TVector<TStringBuf> ToStringBufArray(
        const TVector<TString>& stringArray,
        TVector<TStringBuf>* stringBufArray) {

        stringBufArray->yresize(stringArray.size());
        for (ui32 i : xrange(stringArray.size())) {
            (*stringBufArray)[i] = stringArray[i];
        }

        return *stringBufArray;
    }

    static void AssertApplyEqual(
        const TTextFeature& feature,
        const TTokenizedTextFeature& processedTextFeature,
        const TTextProcessingIdx index,
        const TTextFeatureCalcerPtr& calcer,
        const TTextProcessingCollection& collection) {

        const auto& collectionCalcer = collection.GetCalcer(index.calcerIdx);
        Y_ASSERT(calcer);
        Y_ASSERT(collectionCalcer);

        const ui32 calcerFeatureCount = calcer->FeatureCount();
        const ui64 docCount = feature.size();

        TVector<float> result;

        result.yresize(collection.NumberOfOutputFeatures(index.featureIdx) * docCount);
        TVector<TStringBuf> buffer;

        collection.CalcFeatures(
            ToStringBufArray(feature, &buffer),
            index.featureIdx,
            result);

        const ui32 calcerOffset = collection.GetCalcerFeatureOffset(
            index.featureIdx,
            index.dictionaryIdx,
            index.calcerIdx
        );

        for (ui32 docId : xrange(docCount)) {
            TVector<float> calcerResult = calcer->Compute(processedTextFeature[docId]);

            for (ui32 processedFeatureId : xrange(calcerFeatureCount)) {
                UNIT_ASSERT_EQUAL(
                    calcerResult[processedFeatureId],
                    result[(calcerOffset + processedFeatureId) * docCount + docId]
                );
            }
        }
    }

    static void AssertAllApplyEqual(
        const TVector<TTextFeature>& features,
        const TVector<TTokenizedTextFeature>& tokenizedTextFeatures,
        const TVector<TVector<ui32>>& perFeatureDictionaries,
        const TVector<TVector<ui32>>& perTokenizedFeatureCalcers,
        const TVector<TTextFeatureCalcerPtr>& calcers,
        const TTextProcessingCollection& collection) {

        ui32 tokenizedFeatureId = 0;
        for (ui32 featureId : xrange(features.size())) {
            for (ui32 dictionaryId : perFeatureDictionaries[featureId]) {

                for (ui32 calcerId : perTokenizedFeatureCalcers[tokenizedFeatureId]) {
                    AssertApplyEqual(
                        features[featureId],
                        tokenizedTextFeatures[tokenizedFeatureId],
                        TTextProcessingIdx{ featureId, dictionaryId, calcerId },
                        calcers[calcerId],
                        collection
                    );
                }

                tokenizedFeatureId++;
            }
        }
    }

    static void AssertApplyEqual(
        const TTextProcessingCollection& collection1,
        const TTextProcessingCollection& collection2,
        const TVector<TTextFeature>& features
        ) {

        const ui32 docCount = features[0].size();

        for (ui32 featureId: xrange(features.size())) {
            TVector<float> result1;
            result1.yresize(collection1.NumberOfOutputFeatures(featureId) * docCount);

            TVector<float> result2;
            result2.yresize(collection2.NumberOfOutputFeatures(featureId) * docCount);

            TVector<TStringBuf> buffer1;
            TVector<TStringBuf> buffer2;

            collection1.CalcFeatures(
                ToStringBufArray(features[featureId], &buffer1),
                featureId,
                result1
            );

            collection2.CalcFeatures(
                ToStringBufArray(features[featureId], &buffer2),
                featureId,
                result2
            );

            UNIT_ASSERT_VALUES_EQUAL(result1, result2);
        }
    }

    static TVector<TText> Preprocess(
        const TVector<TString>& feature,
        const TTokenizerPtr& tokenizer,
        const TDictionaryPtr& dictionary) {

        TTextColumnBuilder textColumnBuilder(tokenizer, dictionary, feature.size());
        for (ui32 index: xrange(feature.size())) {
            textColumnBuilder.AddText(index, feature[index]);
        }

        return textColumnBuilder.Build();
    }

    void CreateDataForTest(
        TVector<TTextFeature>* features,
        TVector<TTokenizedTextFeature>* tokenizedFeatures,
        TVector<TTextFeatureCalcerPtr>* calcers,
        TVector<TDictionaryPtr>* dictionaries,
        TTokenizerPtr* tokenizer,
        TVector<TVector<ui32>>* perFeatureDictionaries,
        TVector<TVector<ui32>>* perTokenizedFeatureCalcers
    ) {
        *tokenizer = CreateTokenizer();
        const TEmbeddingPtr embeddingPtr;
        const TTextProcessingOptions textProcessingOptions = TTextProcessingOptions();

        {
            TVector<TString> feature0 = {
                "a b a c a b a",
                "a a",
                "b b",
                "a a a c c"
            };

            TVector<TString> feature1 = {
                "e e f f g g h h",
                "f f",
                "g g",
                "h h"
            };
            TVector<TString> feature2 = {
                "i j k l m n",
                "m",
                "n n n",
                "l i i"
            };
            TVector<TString> feature3 = {
                "aabbaaccaba",
                "aa",
                "bb",
                "aaacc"
            };

            features->insert(features->end(), {feature0, feature1, feature2, feature3});
        }

        TVector<ui32> target = {1, 1, 0, 1};

        {
            TDictionaryPtr dictionary02 = CreateDictionary<TVector<TString>>(
                {(*features)[0], (*features)[2]},
                textProcessingOptions,
                *tokenizer
            );

            TDictionaryPtr dictionary1 = CreateDictionary(
                TIterableTextFeature((*features)[1]),
                textProcessingOptions,
                *tokenizer
            );

            using namespace NTextProcessing::NDictionary;

            TTextProcessingOptions letterGramOptions;
            TDictionaryOptions letterGramDictionaryOptions;
            letterGramDictionaryOptions.TokenLevelType = ETokenLevelType::Letter;
            letterGramDictionaryOptions.GramOrder = 1;
            letterGramOptions.DictionaryOptions.Set(letterGramDictionaryOptions);

            TDictionaryPtr dictionary3 = CreateDictionary(
                TIterableTextFeature((*features)[3]),
                letterGramOptions,
                *tokenizer
            );

            TTextProcessingOptions letterBiGramOptions;
            TDictionaryOptions letterBiGramDictionaryOptions;
            letterBiGramDictionaryOptions.TokenLevelType = ETokenLevelType::Letter;
            letterBiGramDictionaryOptions.GramOrder = 2;
            letterBiGramOptions.DictionaryOptions.Set(letterBiGramDictionaryOptions);
            TDictionaryPtr dictionary4 = CreateDictionary(
                TIterableTextFeature((*features)[3]),
                letterBiGramOptions,
                *tokenizer
            );

            dictionaries->insert(
                dictionaries->end(),
                {dictionary02, dictionary1, dictionary3, dictionary4}
            );
        }

        {
            tokenizedFeatures->emplace_back(Preprocess((*features)[0], *tokenizer, (*dictionaries)[0]));
            tokenizedFeatures->emplace_back(Preprocess((*features)[1], *tokenizer, (*dictionaries)[1]));
            tokenizedFeatures->emplace_back(Preprocess((*features)[2], *tokenizer, (*dictionaries)[0]));
            tokenizedFeatures->emplace_back(Preprocess((*features)[3], *tokenizer, (*dictionaries)[2]));
            tokenizedFeatures->emplace_back(Preprocess((*features)[3], *tokenizer, (*dictionaries)[3]));
        }

        auto naiveBayes0 = CreateBayes((*tokenizedFeatures)[0], target, 2);
        auto naiveBayes1 = CreateBayes((*tokenizedFeatures)[1], target, 2);
        auto naiveBayes2 = CreateBayes((*tokenizedFeatures)[2], target, 2);
        auto naiveBayes3 = CreateBayes((*tokenizedFeatures)[3], target, 2);
        auto naiveBayes4 = CreateBayes((*tokenizedFeatures)[4], target, 2);

        auto bow0 = CreateBoW((*dictionaries)[0]);
        auto bow3 = CreateBoW((*dictionaries)[2]);
        auto bow4 = CreateBoW((*dictionaries)[3]);

        *calcers = {
            naiveBayes0, bow0, naiveBayes1, naiveBayes2, naiveBayes3, naiveBayes4, bow3, bow4
        };

        *perFeatureDictionaries = {
            {0}, {1}, {0}, {2, 3}
        };

        *perTokenizedFeatureCalcers = {
            {0, 1}, {2}, {3}, {4, 5}, {6, 7}
        };
    }

    Y_UNIT_TEST(TestApply) {
        TVector<TTextFeature> features;
        TVector<TTokenizedTextFeature> tokenizedFeatures;
        TVector<TTextFeatureCalcerPtr> calcers;
        TVector<TDictionaryPtr> dictionaries;
        TTokenizerPtr tokenizer;
        TVector<TVector<ui32>> perFeatureDictionaries;
        TVector<TVector<ui32>> perTokenizedFeatureCalcers;

        CreateDataForTest(
            &features,
            &tokenizedFeatures,
            &calcers,
            &dictionaries,
            &tokenizer,
            &perFeatureDictionaries,
            &perTokenizedFeatureCalcers
        );

        TTextProcessingCollection textProcessingCollection = TTextProcessingCollection(
            calcers,
            dictionaries,
            perFeatureDictionaries,
            perTokenizedFeatureCalcers,
            tokenizer
        );

        AssertAllApplyEqual(
            features,
            tokenizedFeatures,
            perFeatureDictionaries,
            perTokenizedFeatureCalcers,
            calcers,
            textProcessingCollection
        );
    }

    Y_UNIT_TEST(TestSerialization) {
        TVector<TTextFeature> features;
        TVector<TTokenizedTextFeature> tokenizedFeatures;
        TVector<TTextFeatureCalcerPtr> calcers;
        TVector<TDictionaryPtr> dictionaries;
        TTokenizerPtr tokenizer;
        TVector<TVector<ui32>> perFeatureDictionaries;
        TVector<TVector<ui32>> perTokenizedFeatureCalcers;

        CreateDataForTest(
            &features,
            &tokenizedFeatures,
            &calcers,
            &dictionaries,
            &tokenizer,
            &perFeatureDictionaries,
            &perTokenizedFeatureCalcers
        );

        TTextProcessingCollection textProcessingCollection = TTextProcessingCollection(
            calcers,
            dictionaries,
            perFeatureDictionaries,
            perTokenizedFeatureCalcers,
            tokenizer
        );

        TStringStream stream;
        textProcessingCollection.Save(&stream);

        TTextProcessingCollection deserializedTextProcessingCollection;
        deserializedTextProcessingCollection.Load(&stream);

        UNIT_ASSERT_EQUAL(textProcessingCollection, deserializedTextProcessingCollection);
        AssertApplyEqual(
            textProcessingCollection,
            deserializedTextProcessingCollection,
            features
        );
    }
}
