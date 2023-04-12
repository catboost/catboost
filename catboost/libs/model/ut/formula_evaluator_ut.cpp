#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/text_features/ut/lib/text_features_data.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/ymath.h>

using namespace NCB;
using namespace NCB::NModelEvaluation;

const TVector<TVector<float>> DATA = {
    {0.f, 0.f, 0.f},
    {3.f, 0.f, 0.f},
    {0.f, 1.f, 0.f},
    {3.f, 1.f, 0.f},
    {0.f, 0.f, 1.f},
    {3.f, 0.f, 1.f},
    {0.f, 1.f, 1.f},
    {3.f, 1.f, 1.f},
};

TVector<TConstArrayRef<float>> GetFeatureRef(const TVector<TVector<float>>& data) {
    TVector<TConstArrayRef<float>> features(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        features[i] = data[i];
    };
    return features;
}

const auto FLOAT_FEATURES = GetFeatureRef(DATA);

void CheckFlatCalcResult(
    const TFullModel& model,
    const TVector<double>& expectedPredicts,
    const TVector<ui32>& expectedLeafIndexes,
    const TVector<TConstArrayRef<float>>& features = FLOAT_FEATURES
) {
    const size_t approxDimension = model.GetDimensionsCount();
    const size_t treeCount = model.GetTreeCount();
    {
        TVector<double> predicts(features.size() * approxDimension);
        model.CalcFlat(features, predicts);
        UNIT_ASSERT_EQUAL(expectedPredicts, predicts);

        TVector<ui32> leafIndexes(features.size() * treeCount, 100);
        model.CalcLeafIndexes(features, {}, leafIndexes);
        UNIT_ASSERT_EQUAL(expectedLeafIndexes, leafIndexes);
    }

    for (size_t sampleIndex = 0; sampleIndex < features.size(); ++sampleIndex) {
        const auto sampleFeatures = features[sampleIndex];
        const TVector<double> expectedSamplePredict(
            expectedPredicts.begin() + sampleIndex * approxDimension,
            expectedPredicts.begin() + (sampleIndex + 1) * approxDimension
        );
        TVector<double> samplePredict(model.GetDimensionsCount());
        model.CalcFlatSingle(sampleFeatures, samplePredict);
        UNIT_ASSERT_EQUAL(expectedSamplePredict, samplePredict);

        const TVector<TCalcerIndexType> expectedSampleIndexes(
            expectedLeafIndexes.begin() + sampleIndex * treeCount,
            expectedLeafIndexes.begin() + (sampleIndex + 1) * treeCount
        );
        TVector<TCalcerIndexType> sampleLeafIndexes(treeCount);
        model.CalcLeafIndexesSingle(sampleFeatures, {}, sampleLeafIndexes);
        UNIT_ASSERT_EQUAL(expectedSampleIndexes, sampleLeafIndexes);
    }
}

Y_UNIT_TEST_SUITE(TObliviousTreeModel) {
    Y_UNIT_TEST(TestFlatCalcFloat) {
        auto model = SimpleFloatModel();
        CheckFlatCalcResult(model, xrange<double>(8), xrange<ui32>(8));
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, xrange<double>(8), xrange<ui32>(8));
    }

    Y_UNIT_TEST(TestFlatCalcFloatWithScaleAndBias) {
        auto model = SimpleFloatModel();
        model.SetScaleAndBias({0.5, {0.125}});
        auto norm = model.GetScaleAndBias();
        TVector<double> expectedPredicts;
        double bias = norm.GetOneDimensionalBias();
        for (int sampleId : xrange(8)) {
            expectedPredicts.push_back(sampleId * norm.Scale + bias);
        }
        CheckFlatCalcResult(model, expectedPredicts, xrange<ui32>(8));
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, xrange<ui32>(8));
    }

    Y_UNIT_TEST(TestTwoTrees) {
        auto model = SimpleFloatModel(2);
        TVector<ui32> expectedLeafIndexes;
        TVector<double> expectedPredicts;
        for (ui32 sampleId = 0; sampleId < 8; ++sampleId) {
            expectedLeafIndexes.push_back(sampleId);
            expectedLeafIndexes.push_back(sampleId);
            expectedPredicts.push_back(11.0 * sampleId);
        }
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes);
    }

    Y_UNIT_TEST(TestFlatCalcOnDeepTree) {
        const size_t treeDepth = 9;
        auto model = SimpleDeepTreeModel(treeDepth);

        TVector<TVector<float>> data;
        TVector<TCalcerIndexType> expectedLeafIndexes;
        TVector<double> expectedPredicts;
        for (size_t sampleId : xrange(1 << treeDepth)) {
            expectedLeafIndexes.push_back(sampleId);
            expectedPredicts.push_back(sampleId);
            TVector<float> sampleFeatures(treeDepth);
            for (auto featureId : xrange(treeDepth)) {
                sampleFeatures[featureId] = sampleId % 2;
                sampleId = sampleId >> 1;
            }
            data.push_back(std::move(sampleFeatures));
        }
        const auto features = GetFeatureRef(data);
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);
    }

    Y_UNIT_TEST(TestFlatCalcMultiVal) {
        auto model = MultiValueFloatModel();
        TVector<TConstArrayRef<float>> features(FLOAT_FEATURES.begin(), FLOAT_FEATURES.begin() + 4);
        TVector<double> expectedPredicts = {
            00., 10., 20.,
            01., 11., 21.,
            02., 12., 22.,
            03., 13., 23.,
        };
        CheckFlatCalcResult(model, expectedPredicts, xrange(4), features);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, xrange(4), features);
    }

    Y_UNIT_TEST(TestFlatCalcMultiValMultiProba) {
        auto model = MultiValueFloatModel();
        constexpr size_t DocCount = 4;
        TConstArrayRef<TConstArrayRef<float>> features(FLOAT_FEATURES.begin(), DocCount);
        TVector<double> expectedProbs = {
            Sigmoid(00.), Sigmoid(10.), Sigmoid(20.),
            Sigmoid(01.), Sigmoid(11.), Sigmoid(21.),
            Sigmoid(02.), Sigmoid(12.), Sigmoid(22.),
            Sigmoid(03.), Sigmoid(13.), Sigmoid(23.),
        };
        auto customEval = model.GetCurrentEvaluator()->Clone();
        customEval->SetPredictionType(NCB::NModelEvaluation::EPredictionType::MultiProbability);
        TVector<double> probs(model.GetDimensionsCount() * DocCount, 0);
        customEval->Calc<TStringBuf>(features, {}, probs);
        for (auto i : xrange(model.GetDimensionsCount() * DocCount)) {
            UNIT_ASSERT_DOUBLES_EQUAL(expectedProbs[i], probs[i], 1.0e-6);
        }
    }

    Y_UNIT_TEST(TestCatOnlyModel) {
        const auto model = TrainCatOnlyModel();

        const auto applySingle = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}};
            double result = 0.;
            model.Calc({}, f, MakeArrayRef(&result, 1));
        };
        UNIT_ASSERT_NO_EXCEPTION(applySingle());

        const auto applyBatch = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "k"}};
            double results[3];
            model.Calc({}, f, results);
        };
        UNIT_ASSERT_NO_EXCEPTION(applyBatch());
    }

    static void CheckCalcTextResult(
        const TFullModel& model,
        TConstArrayRef<TVector<TStringBuf>> transposedTextFeatures,
        TConstArrayRef<double> expectedResults,
        const ui32 docCount,
        const ui32 numEstimatedFeatures
    ) {
        TVector<double> results(docCount);
        const double epsilon = 1e-8;

        for (ui32 estimatedFeatureId = 0; estimatedFeatureId < numEstimatedFeatures; estimatedFeatureId++) {
            ui32 treeIndex = estimatedFeatureId;
            model.Calc(
                {},
                TConstArrayRef<TVector<TStringBuf>>{},
                transposedTextFeatures,
                treeIndex,
                treeIndex + 1,
                MakeArrayRef(results)
            );

            for (ui32 docId: xrange(docCount)) {
                UNIT_ASSERT_DOUBLES_EQUAL(
                    expectedResults[estimatedFeatureId * docCount + docId],
                    results[docId],
                    epsilon
                );
            }
        }
    }

    Y_UNIT_TEST(TestTextOnlyModel) {
        TVector<NCBTest::TTextFeature> features;
        TMap<ui32, NCBTest::TTokenizedTextFeature> tokenizedFeatures;
        TVector<TDigitizer> digitizers;
        TVector<TTextFeatureCalcerPtr> calcers;
        TVector<TVector<ui32>> perFeatureDigitizers;
        TVector<TVector<ui32>> perTokenizedFeatureCalcers;

        NCBTest::CreateTextDataForTest(
            &features,
            &tokenizedFeatures,
            &digitizers,
            &calcers,
            &perFeatureDigitizers,
            &perTokenizedFeatureCalcers
        );

        auto textProcessingCollection = MakeIntrusive<TTextProcessingCollection>(
            digitizers,
            calcers,
            perFeatureDigitizers,
            perTokenizedFeatureCalcers
        );

        const ui32 docCount = features[0].size();
        const ui32 numTextFeatures = features.size();
        const ui32 numEstimatedFeatures = textProcessingCollection->TotalNumberOfOutputFeatures();

        TVector<TVector<TStringBuf>> textFeatures;
        for (auto& feature: features) {
            textFeatures.emplace_back(feature.begin(), feature.end());
        }

        TVector<double> expectedResults(numEstimatedFeatures * docCount);
        auto model = SimpleTextModel(
            textProcessingCollection,
            MakeConstArrayRef(textFeatures),
            MakeArrayRef(expectedResults)
        );

        TVector<TVector<TStringBuf>> transposedTextFeatures;
        for (ui32 docId: xrange(docCount)) {
            auto& ref = transposedTextFeatures.emplace_back();
            for (ui32 featureId: xrange(numTextFeatures)) {
                ref.emplace_back(features[featureId][docId]);
            }
        }

        CheckCalcTextResult(
            model,
            MakeConstArrayRef(transposedTextFeatures),
            MakeConstArrayRef(expectedResults),
            docCount,
            numEstimatedFeatures
        );

        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();

        CheckCalcTextResult(
            model,
            MakeConstArrayRef(transposedTextFeatures),
            MakeConstArrayRef(expectedResults),
            docCount,
            numEstimatedFeatures
        );
    }
}

Y_UNIT_TEST_SUITE(TNonSymmetricTreeModel) {
    Y_UNIT_TEST(TestFlatCalcFloat) {
        auto modelCalcer = SimpleAsymmetricModel();
        TVector<double> canonVals = {
            101., 203., 102., 303.,
            111., 213., 112., 313.};
        TVector<ui32> expectedLeafIndexes = {
            1, 0, 0,
            0, 0, 1,
            2, 0, 0,
            0, 0, 2,
            1, 1, 0,
            0, 1, 1,
            2, 1, 0,
            0, 1, 2
        };
        CheckFlatCalcResult(modelCalcer, canonVals, expectedLeafIndexes);

        TStringStream strStream;
        modelCalcer.Save(&strStream);
        TFullModel deserializedModel;
        deserializedModel.Load(&strStream);
        CheckFlatCalcResult(deserializedModel, canonVals, expectedLeafIndexes);
    }
}
