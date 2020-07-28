
#include <catboost/private/libs/algo/apply.h>

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TLeafIndexCalcerOnPool) {
    const TVector<TVector<float>> DEFAULT_FEATURES = {
        {0.f, 0.f, 0.f},
        {3.f, 0.f, 0.f},
        {0.f, 1.f, 0.f},
        {3.f, 1.f, 0.f},
        {0.f, 0.f, 1.f},
        {3.f, 0.f, 1.f},
        {0.f, 1.f, 1.f},
        {3.f, 1.f, 1.f},
    };

    TObjectsDataProviderPtr CreateObjectsDataProviderWithFeatures(
        const TVector<TVector<float>>& featuresData) {

        auto dataProvider = CreateDataProvider<IRawObjectsOrderDataVisitor>(
            [&] (IRawObjectsOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                    (ui32)featuresData[0].size(),
                    TVector<ui32>{},
                    TVector<ui32>{},
                    TVector<ui32>{},
                    TVector<TString>{});

                visitor->Start(
                    /*inBlock*/false,
                    metaInfo,
                    /*haveUnknownNumberOfSparseFeatures*/ false,
                    (ui32)featuresData.size(),
                    EObjectsOrder::Undefined,
                    /*resourceHolders*/ {});
                visitor->StartNextBlock((ui32)featuresData.size());

                for (auto objectIdx : xrange(featuresData.size())) {
                    visitor->AddAllFloatFeatures(objectIdx, featuresData[objectIdx]);
                }

                visitor->Finish();
            });

        return dataProvider->ObjectsData;
    };

    void CheckLeafIndexCalcer(
        const TFullModel& model,
        const TVector<TVector<float>>& features,
        const TVector<ui32>& expectedLeafIndexes) {

        TObjectsDataProviderPtr objectsData = CreateObjectsDataProviderWithFeatures(features);

        const auto treeCount = model.GetTreeCount();

        TLeafIndexCalcerOnPool leafIndexCalcer(model, objectsData, 0, (int)treeCount);

        for (ui32 sampleIndex = 0; sampleIndex < objectsData->GetObjectCount(); ++sampleIndex) {
            const TVector<ui32> expectedSampleIndexes(
                expectedLeafIndexes.begin() + sampleIndex * treeCount,
                expectedLeafIndexes.begin() + (sampleIndex + 1) * treeCount
            );

            UNIT_ASSERT(leafIndexCalcer.CanGet());
            const TVector<ui32> sampleLeafIndexes = leafIndexCalcer.Get();
            UNIT_ASSERT_VALUES_EQUAL(expectedSampleIndexes, sampleLeafIndexes);
            const bool hasNextResult = leafIndexCalcer.Next();
            UNIT_ASSERT_EQUAL(hasNextResult, sampleIndex + 1 != objectsData->GetObjectCount());
            UNIT_ASSERT_EQUAL(hasNextResult, leafIndexCalcer.CanGet());
        }
        UNIT_ASSERT(!leafIndexCalcer.CanGet());
    }

    Y_UNIT_TEST(TestFloat) {
        auto model = SimpleFloatModel();
        CheckLeafIndexCalcer(model, DEFAULT_FEATURES, /*expectedLeafIndexes*/ xrange<ui32>(8));
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckLeafIndexCalcer(model, DEFAULT_FEATURES, /*expectedLeafIndexes*/ xrange<ui32>(8));
    }

    Y_UNIT_TEST(TestTwoTrees) {
        auto model = SimpleFloatModel(2);
        TVector<ui32> expectedLeafIndexes;
        for (ui32 sampleId = 0; sampleId < 8; ++sampleId) {
            expectedLeafIndexes.push_back(sampleId);
            expectedLeafIndexes.push_back(sampleId);
        }
        CheckLeafIndexCalcer(model, DEFAULT_FEATURES, expectedLeafIndexes);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckLeafIndexCalcer(model, DEFAULT_FEATURES, expectedLeafIndexes);
    }

    Y_UNIT_TEST(TestOnDeepTree) {
        const size_t treeDepth = 9;
        auto model = SimpleDeepTreeModel(treeDepth);

        TVector<TVector<float>> features;
        TVector<ui32> expectedLeafIndexes;
        for (size_t sampleId : xrange(1 << treeDepth)) {
            expectedLeafIndexes.push_back(sampleId);
            TVector<float> sampleFeatures(treeDepth);
            for (auto featureId : xrange(treeDepth)) {
                sampleFeatures[featureId] = sampleId % 2;
                sampleId = sampleId >> 1;
            }
            features.push_back(std::move(sampleFeatures));
        }
        CheckLeafIndexCalcer(model, features, expectedLeafIndexes);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckLeafIndexCalcer(model, features, expectedLeafIndexes);
    }

    Y_UNIT_TEST(TestMultiVal) {
        auto model = MultiValueFloatModel();
        TVector<TVector<float>> features(DEFAULT_FEATURES.begin(), DEFAULT_FEATURES.begin() + 4);
        CheckLeafIndexCalcer(model, features, /*expectedLeafIndexes*/ xrange(4));
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckLeafIndexCalcer(model, features, /*expectedLeafIndexes*/ xrange(4));
    }

    Y_UNIT_TEST(TNonSymmetricTreeModel) {
        auto model = SimpleAsymmetricModel();
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
        CheckLeafIndexCalcer(model, DEFAULT_FEATURES, expectedLeafIndexes);
    }
}
