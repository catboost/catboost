#include <catboost/private/libs/algo/full_model_saver.h>

#include <catboost/private/libs/feature_estimator/text_feature_estimators.h>
#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/text_features/ut/lib/text_features_data.h>

#include <library/cpp/unittest/registar.h>
#include <util/generic/xrange.h>

using namespace NCB;
using namespace NCatboostOptions;


static void CreateEstimators(
    const TTextDigitizers& textDigitizers,
    TVector<NCBTest::TTokenizedTextFeature>&& tokenizedFeatures,
    TVector<ui32>&& target,
    TConstArrayRef<TTokenizedFeatureDescription> tokenizedFeatureDescriptions,
    TFeatureEstimatorsPtr* featureEstimators
) {
    const ui32 numClasses = 2;
    auto textTarget = MakeIntrusive<TTextClassificationTarget>(std::move(target), numClasses);
    const ui32 numTokenizedFeatures = static_cast<ui32>(tokenizedFeatures.size());

    TFeatureEstimatorsBuilder estimatorsBuilder;

    for (ui32 tokenizedFeatureId = 0; tokenizedFeatureId < numTokenizedFeatures; tokenizedFeatureId++) {
        const auto& tokenizedFeatureDescription = tokenizedFeatureDescriptions[tokenizedFeatureId];

        TTextDataSetPtr learnTexts = MakeIntrusive<TTextDataSet>(
            NCB::TTextColumn::CreateOwning(
                std::move(tokenizedFeatures[tokenizedFeatureId])
            ),
            textDigitizers.GetDigitizer(tokenizedFeatureId).Dictionary
        );
        TVector<TTextDataSetPtr> testTexts{learnTexts};

        TEstimatorSourceId sourceFeatureIdx{
            tokenizedFeatureDescription.TextFeatureId,
            tokenizedFeatureId
        };

        TEmbeddingPtr embeddingPtr;
        {
            TVector<TFeatureEstimatorPtr> offlineEstimators = CreateEstimators(
                MakeConstArrayRef(tokenizedFeatureDescription.FeatureEstimators.Get()),
                embeddingPtr,
                learnTexts,
                MakeArrayRef(testTexts)
            );

            for (TFeatureEstimatorPtr estimator: offlineEstimators) {
                estimatorsBuilder.AddFeatureEstimator(std::move(estimator), sourceFeatureIdx);
            }
        }
        {
            TVector<TOnlineFeatureEstimatorPtr> onlineEstimators = CreateEstimators(
                MakeConstArrayRef(tokenizedFeatureDescription.FeatureEstimators.Get()),
                embeddingPtr,
                textTarget,
                learnTexts,
                MakeArrayRef(testTexts)
            );

            for (TOnlineFeatureEstimatorPtr estimator: onlineEstimators) {
                estimatorsBuilder.AddFeatureEstimator(std::move(estimator), sourceFeatureIdx);
            }
        }
    }

    *featureEstimators = estimatorsBuilder.Build();
}

static void CreateDataForTest(
    TTextDigitizers* textDigitizers,
    TVector<NCBTest::TTextFeature>* textFeatures,
    TVector<TEstimatedFeature>* estimatedFeatures,
    TFeatureEstimatorsPtr* estimators
) {
    TVector<NCBTest::TTokenizedTextFeature> tokenizedFeatures;
    TVector<TDictionaryPtr> dictionaries;
    TTokenizerPtr tokenizer;
    TVector<ui32> target;
    TTextProcessingOptions textProcessingOptions;

    NCBTest::CreateTextDataForTest(
        textFeatures,
        &tokenizedFeatures,
        &target,
        textDigitizers,
        &textProcessingOptions
    );

    TRuntimeTextOptions runtimeTextOptions{xrange(textFeatures->size()), textProcessingOptions};
    const auto& tokenizedFeatureDescriptions = runtimeTextOptions.GetTokenizedFeatureDescriptions();

    CreateEstimators(
        *textDigitizers,
        std::move(tokenizedFeatures),
        std::move(target),
        MakeConstArrayRef(tokenizedFeatureDescriptions),
        estimators
    );
    Y_ASSERT(estimators);

    (*estimators)->ForEach(
        [&] (TEstimatorId estimatorId, TFeatureEstimatorPtr featureEstimator) {
            Y_UNUSED(estimatorId);
            for (ui32 localId: xrange(featureEstimator->FeaturesMeta().FeaturesCount)) {
                const TGuid& guid = featureEstimator->Id();
                estimatedFeatures->push_back(
                    TEstimatedFeature{
                        SafeIntegerCast<int>((*estimators)->GetEstimatorSourceFeatureIdx(guid).TextFeatureId),
                        guid,
                        SafeIntegerCast<int>(localId)
                    }
                );
            }
        }
    );
}

static TVector<TEstimatedFeature> RandomSubsample(
    TConstArrayRef<TEstimatedFeature> estimatedFeatures,
    float threshold,
    TFastRng32* rng
) {
    TVector<TEstimatedFeature> sample;
    for (const auto& estimatedFeature : estimatedFeatures) {
        if (rng->GenRandReal1() > threshold) {
            sample.push_back(estimatedFeature);
        }
    }
    return sample;
}

static TVector<float> ApplyCollection(
    const TTextProcessingCollection& collection,
    const TVector<NCBTest::TTextFeature>& textFeatures
) {
    TVector<ui32> textFeatureIds = xrange(textFeatures.size());

    const ui32 docCount = textFeatures[0].size();
    const ui32 numFeatures = collection.TotalNumberOfOutputFeatures();
    TVector<float> result(numFeatures * docCount);

    collection.CalcFeatures(
        [&](ui32 textFeatureId, ui32 docId) -> TStringBuf { return textFeatures[textFeatureId][docId]; },
        MakeConstArrayRef(textFeatureIds),
        docCount,
        result
    );

    return result;
}

static void AssertEqualDoubleArrays(TArrayRef<float> expected, TArrayRef<float> actual, float epsilon = 1e-10) {
    UNIT_ASSERT_EQUAL(expected.size(), actual.size());
    Y_UNUSED(epsilon);
    for (ui32 i : xrange(expected.size())) {
        UNIT_ASSERT_DOUBLES_EQUAL(expected[i], actual[i], epsilon);
    }
}

static void AssertEqualCollections(
    const TTextProcessingCollection& originalCollection,
    const TTextProcessingCollection& actualCollection,
    const TVector<NCBTest::TTextFeature>& textFeatures,
    TMaybe<TVector<TEstimatedFeature>> partEstimatedFeatures = Nothing()
) {
    const ui32 docCount = textFeatures[0].size();

    UNIT_ASSERT_EQUAL(
        originalCollection.GetTextFeatureCount(),
        actualCollection.GetTextFeatureCount()
    );

    TVector<float> expectedResult = ApplyCollection(originalCollection, textFeatures);
    TVector<float> result = ApplyCollection(actualCollection, textFeatures);

    if (partEstimatedFeatures.Defined()) {
        TGuid lastGuid = partEstimatedFeatures->at(0).CalcerId;
        ui32 localId = 0;
        for (ui32 i: xrange(partEstimatedFeatures->size())) {
            TEstimatedFeature& feature = partEstimatedFeatures->at(i);
            const TGuid calcerId = feature.CalcerId;
            const ui32 originalLocalId = feature.LocalIndex;
            if (calcerId != lastGuid) {
                lastGuid = calcerId;
                localId = 0;
            }

            const ui32 originalOffset =
                originalCollection.GetAbsoluteCalcerOffset(calcerId) + originalLocalId;
            const ui32 partOffset =
                actualCollection.GetAbsoluteCalcerOffset(calcerId) + localId;

            AssertEqualDoubleArrays(
                TArrayRef<float>(expectedResult.data() + originalOffset * docCount, docCount),
                TArrayRef<float>(result.data() + partOffset * docCount, docCount)
            );
            localId++;
        }
    } else {
        UNIT_ASSERT_VALUES_EQUAL(expectedResult, result);
    }
}

Y_UNIT_TEST_SUITE(TextCollectionBuilderTest) {
    Y_UNIT_TEST(DifferentCreationTest) {
        TVector<NCBTest::TTextFeature> textFeatures;
        TTextProcessingCollection fromTrainTextProcessingCollection;

        {
            TFeatureEstimatorsPtr estimators;
            TTextDigitizers textDigitizers;
            TVector<TEstimatedFeature> estimatedFeatures;
            NPar::TLocalExecutor localExecutor;

            CreateDataForTest(
                &textDigitizers,
                &textFeatures,
                &estimatedFeatures,
                &estimators
            );

            TVector<TEstimatedFeature> reorderedEstimatedFeatures;

            CreateTextProcessingCollection(
                *estimators,
                textDigitizers,
                estimatedFeatures,
                &fromTrainTextProcessingCollection,
                &reorderedEstimatedFeatures,
                &localExecutor
            );
        }

        auto directTextProcessingCollection = NCBTest::CreateTextProcessingCollectionForTest();

        AssertEqualCollections(
            *directTextProcessingCollection,
            fromTrainTextProcessingCollection,
            textFeatures
        );
    }

    Y_UNIT_TEST(ReduceEstimatedFeaturesTest) {
        TFeatureEstimatorsPtr estimators;
        TTextDigitizers textDigitizers;
        TVector<TEstimatedFeature> fullEstimatedFeatures;
        TVector<NCBTest::TTextFeature> textFeatures;
        TMap<TEstimatorSourceId, TSet<TEstimatorId>> sourceIdToEstimators;
        NPar::TLocalExecutor localExecutor;

        CreateDataForTest(
            &textDigitizers,
            &textFeatures,
            &fullEstimatedFeatures,
            &estimators
        );

        TTextProcessingCollection textProcessingCollection;

        {
            TVector<TEstimatedFeature> textCollectionEstimatedFeatures;
            CreateTextProcessingCollection(
                *estimators,
                textDigitizers,
                fullEstimatedFeatures,
                &textProcessingCollection,
                &textCollectionEstimatedFeatures,
                &localExecutor
            );
            fullEstimatedFeatures = textCollectionEstimatedFeatures;
        }

        TFastRng32 rng(42, 0);

        auto assertPartEqualFull = [&](float threshold) {
            for (ui32 testRound: xrange(10u)) {
                Y_UNUSED(testRound);
                TVector<TEstimatedFeature> partEstimatedFeatures = RandomSubsample(
                    MakeConstArrayRef(fullEstimatedFeatures),
                    threshold,
                    &rng
                );
                if (partEstimatedFeatures.empty()) {
                    continue;
                }

                TTextProcessingCollection partialTextCollection;

                {
                    TVector<TEstimatedFeature> reorderedEstimatedFeatures;
                    CreateTextProcessingCollection(
                        *estimators,
                        textDigitizers,
                        partEstimatedFeatures,
                        &partialTextCollection,
                        &reorderedEstimatedFeatures,
                        &localExecutor
                    );
                }

                AssertEqualCollections(
                    textProcessingCollection,
                    partialTextCollection,
                    textFeatures,
                    partEstimatedFeatures
                );
            }
        };

        assertPartEqualFull(0.1);
        assertPartEqualFull(1.);
        assertPartEqualFull(-1.);
        assertPartEqualFull(0.9);
        assertPartEqualFull(0.5);
    }
}
