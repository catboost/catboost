#include "text_features_data.h"

#include <catboost/private/libs/options/runtime_text_options.h>
#include <catboost/private/libs/text_processing/embedding.h>
#include <catboost/private/libs/text_processing/text_column_builder.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

using namespace NCB;
using namespace NCBTest;
using namespace NCatboostOptions;

TIntrusivePtr<TMultinomialNaiveBayes> NCBTest::CreateBayes(
    const NCBTest::TTokenizedTextFeature& features,
    TConstArrayRef<ui32> target,
    ui32 numClasses
) {
    auto naiveBayes = MakeIntrusive<TMultinomialNaiveBayes>(numClasses);
    TNaiveBayesVisitor bayesVisitor;

    for (ui32 sampleId: xrange(features.size())) {
        bayesVisitor.Update(target[sampleId], features[sampleId], naiveBayes.Get());
    }

    return naiveBayes;
}

TIntrusivePtr<TBM25> NCBTest::CreateBm25(
    const NCBTest::TTokenizedTextFeature& features,
    TConstArrayRef<ui32> target,
    ui32 numClasses
) {
    auto bm25 = MakeIntrusive<TBM25>(TBM25(numClasses));
    TBM25Visitor bm25Visitor;

    for (ui32 sampleId: xrange(features.size())) {
        bm25Visitor.Update(target[sampleId], features[sampleId], bm25.Get());
    }

    return bm25;
}

TIntrusivePtr<TBagOfWordsCalcer> NCBTest::CreateBoW(const TDictionaryPtr& dictionaryPtr) {
    return MakeIntrusive<TBagOfWordsCalcer>(dictionaryPtr->Size());
}

static void CreateCalcersAndDependencies(
    TConstArrayRef<TTokenizedTextFeature> tokenizedFeatures,
    TConstArrayRef<TDictionaryPtr> dictionaries,
    TConstArrayRef<ui32> target,
    TVector<TTextFeatureCalcerPtr>* calcers,
    TVector<TVector<ui32>>* perFeatureDictionaries,
    TVector<TVector<ui32>>* perTokenizedFeatureCalcers
) {
    auto naiveBayes0 = CreateBayes(tokenizedFeatures[0], target, 2);
    auto naiveBayes1 = CreateBayes(tokenizedFeatures[1], target, 2);
    auto naiveBayes2 = CreateBayes(tokenizedFeatures[2], target, 2);
    auto naiveBayes3 = CreateBayes(tokenizedFeatures[3], target, 2);
    auto naiveBayes4 = CreateBayes(tokenizedFeatures[4], target, 2);

    auto bow0 = CreateBoW(dictionaries[0]);
    auto bow3 = CreateBoW(dictionaries[2]);
    auto bow4 = CreateBoW(dictionaries[3]);

    auto bm0 = CreateBm25(tokenizedFeatures[0], target, 2);
    auto bm2 = CreateBm25(tokenizedFeatures[2], target, 2);
    auto bm4 = CreateBm25(tokenizedFeatures[4], target, 2);

    *calcers = {
        /*  0 */ bow0,
        /*  1 */ bm0,
        /*  2 */ naiveBayes0,
        /*  3 */ naiveBayes1,
        /*  4 */ bm2,
        /*  5 */ naiveBayes2,
        /*  6 */ bow3,
        /*  7 */ naiveBayes3,
        /*  8 */ bow4,
        /*  9 */ bm4,
        /* 10 */ naiveBayes4
    };

    *perFeatureDictionaries = {
        {0}, {1}, {2}, {3, 4}
    };

    *perTokenizedFeatureCalcers = {
        {0, 1, 2}, {3}, {4, 5}, {6, 7}, {8, 9, 10}
    };
}

void NCBTest::CreateTextDataForTest(
    TVector<TTextFeature>* features,
    TVector<TTokenizedTextFeature>* tokenizedFeatures,
    TVector<TTextFeatureCalcerPtr>* calcers,
    TVector<TDictionaryPtr>* dictionaries,
    TTokenizerPtr* tokenizer,
    TVector<TVector<ui32>>* perFeatureDictionaries,
    TVector<TVector<ui32>>* perTokenizedFeatureCalcers
) {
    TVector<ui32> target;
    TRuntimeTextOptions options;
    CreateTextDataForTest(features, tokenizedFeatures, dictionaries, tokenizer, &target, &options);
    CreateCalcersAndDependencies(
        MakeConstArrayRef(*tokenizedFeatures),
        MakeConstArrayRef(*dictionaries),
        MakeConstArrayRef(target),
        calcers,
        perFeatureDictionaries,
        perTokenizedFeatureCalcers
    );
}

TIntrusivePtr<NCB::TTextProcessingCollection> NCBTest::CreateTextProcessingCollectionForTest() {
    TVector<TTextFeature> features;
    TVector<TTokenizedTextFeature> tokenizedFeatures;
    TVector<TTextFeatureCalcerPtr> calcers;
    TVector<TDictionaryPtr> dictionaries;
    TTokenizerPtr tokenizer;
    TVector<TVector<ui32>> perFeatureDictionaries;
    TVector<TVector<ui32>> perTokenizedFeatureCalcers;
    CreateTextDataForTest(
        &features,
        &tokenizedFeatures,
        &calcers,
        &dictionaries,
        &tokenizer,
        &perFeatureDictionaries,
        &perTokenizedFeatureCalcers
    );
    return MakeIntrusive<TTextProcessingCollection>(
        calcers,
        dictionaries,
        perFeatureDictionaries,
        perTokenizedFeatureCalcers,
        tokenizer
    );
}
