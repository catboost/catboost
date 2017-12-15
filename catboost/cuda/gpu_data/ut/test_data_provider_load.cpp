#include <catboost/cuda/ut_helpers/test_utils.h>
#include <catboost/cuda/data/load_data.h>
#include <util/generic/set.h>
#include <library/unittest/registar.h>

using namespace std;
using namespace NCatboostCuda;

SIMPLE_UNIT_TEST_SUITE(TDataProviderTest) {
    //
    SIMPLE_UNIT_TEST(TestDataLoad) {
        TUnitTestPool pool;
        GenerateTestPool(pool);
        SavePoolToFile(pool, "test-pool.txt");
        SavePoolCDToFile("test-pool.txt.cd");
        const ui32 binarization = 32;

        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum, binarization);
        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.MaxTensorComplexity = 3;
        catFeatureParams.OneHotMaxSize = 6;
        TBinarizedFeaturesManager binarizedFeaturesManager(catFeatureParams, floatBinarization);
        TDataProvider dataProvider;
        TOnCpuGridBuilderFactory gridBuilderFactory;
        TDataProviderBuilder builder(binarizedFeaturesManager, dataProvider);

        ReadPool("test-pool.txt.cd",
                 "test-pool.txt",
                 "",
                 16,
                 true,
                 builder.SetShuffleFlag(false));

        UNIT_ASSERT_VALUES_EQUAL(pool.NumFeatures + 1, dataProvider.GetEffectiveFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.NumSamples, dataProvider.GetSampleCount());
        {
            auto binarizer = gridBuilderFactory.Create(floatBinarization.BorderSelectionType);

            //4th gid column set to cat-feature and takes id
            for (size_t f = 0; f < pool.NumFeatures; ++f) {
                TVector<float> feature = pool.GetFeature(f);
                Sort(feature.begin(), feature.end());
                auto borders = binarizer->BuildBorders(feature, floatBinarization.BorderCount);
                auto binarized = BinarizeLine<ui32>(~pool.Features + f * pool.NumSamples, pool.NumSamples, ENanMode::Forbidden, borders);

                auto& featureHolder = dataProvider.GetFeatureById(f + 1);
                UNIT_ASSERT_VALUES_EQUAL(featureHolder.GetType(), EFeatureValuesType::BinarizedFloat);
                const TBinarizedFloatValuesHolder& valuesHolder = dynamic_cast<const TBinarizedFloatValuesHolder&>(featureHolder);
                for (ui32 i = 0; i < borders.size(); ++i) {
                    UNIT_ASSERT_VALUES_EQUAL(borders[i], valuesHolder.GetBorders()[i]);
                }

                auto extracted = valuesHolder.ExtractValues();
                for (size_t doc = 0; doc < pool.NumSamples; ++doc) {
                    UNIT_ASSERT_VALUES_EQUAL(binarized[doc], extracted[doc]);
                }
            }
        }
        for (size_t i = 0; i < pool.Targets.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(pool.Targets[i], dataProvider.GetTargets()[i]);
            UNIT_ASSERT_VALUES_EQUAL(1.0, dataProvider.GetWeights()[i]);
        }

        for (size_t i = 0; i < pool.Qids.size(); ++i) {
            auto qid = dataProvider.GetQueryIds()[i];
            UNIT_ASSERT_VALUES_EQUAL(pool.Qids[i], qid);
        }

        {
            TMap<ui32, ui32> gidsBins;

            for (size_t i = 0; i < pool.Gids.size(); ++i) {
                if (gidsBins.count(pool.Gids[i]) == 0) {
                    gidsBins[pool.Gids[i]] = gidsBins.size();
                }

                auto& valuesHolder = dataProvider.GetFeatureById(0);
                auto gids = dynamic_cast<const TCatFeatureValuesHolder&>(valuesHolder).ExtractValues();
                if (gidsBins[pool.Gids[i]] != gids[i]) {
                    Cout << i << " " << gids[i] << Endl;
                }
                UNIT_ASSERT_VALUES_EQUAL(dynamic_cast<const TCatFeatureValuesHolder&>(valuesHolder).GetValue(i), gids[i]);
                UNIT_ASSERT_VALUES_EQUAL(gidsBins[pool.Gids[i]], gids[i]);
            }
        }
    }
}
