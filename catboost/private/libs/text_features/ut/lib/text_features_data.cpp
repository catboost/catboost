#include "text_features_data.h"

#include <catboost/private/libs/options/runtime_text_options.h>
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
    auto naiveBayes = MakeIntrusive<TMultinomialNaiveBayes>(CreateGuid(), numClasses);
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
    auto bm25 = MakeIntrusive<TBM25>(CreateGuid(), numClasses);
    TBM25Visitor bm25Visitor;

    for (ui32 sampleId: xrange(features.size())) {
        bm25Visitor.Update(target[sampleId], features[sampleId], bm25.Get());
    }

    return bm25;
}

TIntrusivePtr<TBagOfWordsCalcer> NCBTest::CreateBoW(const TDictionaryPtr& dictionaryPtr) {
    return MakeIntrusive<TBagOfWordsCalcer>(CreateGuid(), dictionaryPtr->Size());
}

static void CreateCalcersAndDependencies(
    TMap<ui32, TTokenizedTextFeature>* tokenizedFeatures,
    TConstArrayRef<TDigitizer> digitizers,
    TConstArrayRef<ui32> target,
    ui32 textFeatureCount,
    const TRuntimeTextOptions& runtimeTextOptions,
    TVector<TTextFeatureCalcerPtr>* calcers,
    TVector<TVector<ui32>>* perFeatureDigitizers,
    TVector<TVector<ui32>>* perTokenizedFeatureCalcers
) {
    const ui32 classesCount = 2;

    perFeatureDigitizers->resize(textFeatureCount);
    perTokenizedFeatureCalcers->resize(tokenizedFeatures->size());

    for (const auto& [tokenizedFeatureIdx, tokenizedFeature] : *tokenizedFeatures) {
        const ui32 shiftedTokenizedIdx = tokenizedFeatureIdx - textFeatureCount;
        const auto& featureDescription = runtimeTextOptions.GetTokenizedFeatureDescription(shiftedTokenizedIdx);

        const auto& dictionary = digitizers[shiftedTokenizedIdx].Dictionary;
        const ui32 textFeatureIdx = featureDescription.TextFeatureId;
        perFeatureDigitizers->at(textFeatureIdx).push_back(shiftedTokenizedIdx);

        auto& featureCalcers = (*perTokenizedFeatureCalcers)[shiftedTokenizedIdx];

        for (const auto& featureCalcer: featureDescription.FeatureEstimators.Get()) {
            const EFeatureCalcerType calcerType = featureCalcer.CalcerType;
            featureCalcers.push_back(calcers->size());

            switch (calcerType) {
                case EFeatureCalcerType::BoW:
                    calcers->push_back(CreateBoW(dictionary));
                    break;
                case EFeatureCalcerType::NaiveBayes:
                    calcers->push_back(CreateBayes(tokenizedFeature, target, classesCount));
                    break;
                case EFeatureCalcerType::BM25:
                    calcers->push_back(CreateBm25(tokenizedFeature, target, classesCount));
                    break;
                default:
                    CB_ENSURE(false, "feature calcer " << calcerType << " is not present in tests");
            }
        }
    }
}

void NCBTest::CreateTextDataForTest(
    TVector<TTextFeature>* features,
    TMap<ui32, TTokenizedTextFeature>* tokenizedFeatures,
    TVector<TDigitizer>* digitizers,
    TVector<TTextFeatureCalcerPtr>* calcers,
    TVector<TVector<ui32>>* perFeatureDigitizers,
    TVector<TVector<ui32>>* perTokenizedFeatureCalcers
) {
    TVector<ui32> target;
    TTextProcessingOptions options;
    TTextDigitizers textDigitizers;
    CreateTextDataForTest(features, tokenizedFeatures, &target, &textDigitizers, &options);

    *digitizers = textDigitizers.GetDigitizers();

    TRuntimeTextOptions runtimeTextOptions(xrange(features->size()), options);
    CreateCalcersAndDependencies(
        tokenizedFeatures,
        MakeConstArrayRef(*digitizers),
        MakeConstArrayRef(target),
        features->size(),
        runtimeTextOptions,
        calcers,
        perFeatureDigitizers,
        perTokenizedFeatureCalcers
    );
}

TIntrusivePtr<NCB::TTextProcessingCollection> NCBTest::CreateTextProcessingCollectionForTest() {
    TVector<TTextFeature> features;
    TMap<ui32, TTokenizedTextFeature> tokenizedFeatures;
    TVector<TDigitizer> digitizers;
    TVector<TTextFeatureCalcerPtr> calcers;
    TVector<TVector<ui32>> perFeatureDigitizers;
    TVector<TVector<ui32>> perTokenizedFeatureCalcers;

    CreateTextDataForTest(
        &features,
        &tokenizedFeatures,
        &digitizers,
        &calcers,
        &perFeatureDigitizers,
        &perTokenizedFeatureCalcers
    );

    return MakeIntrusive<TTextProcessingCollection>(
        digitizers,
        calcers,
        perFeatureDigitizers,
        perTokenizedFeatureCalcers
    );
}
