#include "lib/text_features_data.h"
#include <catboost/private/libs/text_features/text_processing_collection.h>

#include <catboost/private/libs/text_features/bow.h>
#include <catboost/private/libs/text_processing/text_column_builder.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NCB;
using namespace NCBTest;
using namespace NCatboostOptions;

namespace {
    struct TTextProcessingIdx {
    public:
        ui32 featureIdx;
        ui32 digitizerId;
        ui32 calcerIdx;
    };
}

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
    const TTokenizedTextFeature& tokenizedTextFeature,
    const TTextProcessingIdx index,
    const TTextFeatureCalcerPtr& calcer,
    const TTextProcessingCollection& collection) {

    const auto& collectionCalcer = collection.GetCalcer(index.calcerIdx);
    UNIT_ASSERT(calcer);
    UNIT_ASSERT(collectionCalcer);
    UNIT_ASSERT(calcer->Id() == collectionCalcer->Id());

    const ui32 calcerFeatureCount = calcer->FeatureCount();
    const ui64 docCount = feature.size();

    TVector<float> result;

    result.yresize(collection.NumberOfOutputFeatures(index.featureIdx) * docCount);
    TVector<TStringBuf> buffer;

    collection.CalcFeatures(
        ToStringBufArray(feature, &buffer),
        index.featureIdx,
        MakeArrayRef(result));

    for (ui32 docId : xrange(docCount)) {
        TVector<float> calcerResult = calcer->Compute(tokenizedTextFeature[docId]);
        const ui32 calcerOffset = collection.GetRelativeCalcerOffset(index.featureIdx, calcer->Id());

        for (ui32 localIndex : xrange(calcerFeatureCount)) {
            UNIT_ASSERT_EQUAL(
                calcerResult[localIndex],
                result[(calcerOffset + localIndex) * docCount + docId]
            );
        }
    }
}

static void AssertAllApplyEqual(
    const TVector<TTextFeature>& features,
    const TMap<ui32, TTokenizedTextFeature>& tokenizedTextFeatures,
    const TVector<TVector<ui32>>& perFeatureDigitizers,
    const TVector<TVector<ui32>>& perTokenizedFeatureCalcers,
    const TVector<TTextFeatureCalcerPtr>& calcers,
    const TTextProcessingCollection& collection
) {
    ui32 tokenizedFeatureId = 0;
    for (ui32 featureId : xrange(features.size())) {
        for (ui32 digitizerId : perFeatureDigitizers[featureId]) {

            for (ui32 calcerId : perTokenizedFeatureCalcers[tokenizedFeatureId]) {
                AssertApplyEqual(
                    features[featureId],
                    tokenizedTextFeatures.at(tokenizedFeatureId + features.size()),
                    TTextProcessingIdx{ featureId, digitizerId, calcerId },
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

        UNIT_ASSERT_EQUAL(result1.ysize(), result2.ysize());
        for (ui32 idx : xrange(result1.size())) {
            UNIT_ASSERT_DOUBLES_EQUAL(result1[idx], result2[idx], 1e-8);
        }
    }
}

Y_UNIT_TEST_SUITE(TestTextProcessingCollection) {

    Y_UNIT_TEST(TestApply) {
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

        TTextProcessingCollection textProcessingCollection = TTextProcessingCollection(
            digitizers,
            calcers,
            perFeatureDigitizers,
            perTokenizedFeatureCalcers
        );

        AssertAllApplyEqual(
            features,
            tokenizedFeatures,
            perFeatureDigitizers,
            perTokenizedFeatureCalcers,
            calcers,
            textProcessingCollection
        );
    }

    Y_UNIT_TEST(TestSerialization) {
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

        TTextProcessingCollection textProcessingCollection = TTextProcessingCollection(
            digitizers,
            calcers,
            perFeatureDigitizers,
            perTokenizedFeatureCalcers
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
