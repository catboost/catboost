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

        TBinarizationConfiguration binarizationConfiguration;
        TFeatureManagerOptions featureManagerOptions(binarizationConfiguration, 6);
        TBinarizedFeaturesManager binarizedFeaturesManager(featureManagerOptions);
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
        UNIT_ASSERT_VALUES_EQUAL(pool.Queries.size(), dataProvider.GetQueries().size());
        {
            auto binarizer = gridBuilderFactory.Create(binarizationConfiguration.DefaultFloatBinarization.BorderSelectionType);

            //4th gid column set to cat-feature and takes id
            for (size_t f = 0; f < pool.NumFeatures; ++f) {
                yvector<float> feature = pool.GetFeature(f);
                Sort(feature.begin(), feature.end());
                auto borders = binarizer->BuildBorders(feature, binarizationConfiguration.DefaultFloatBinarization.Discretization);
                auto binarized = BinarizeLine<ui32>(~pool.Features + f * pool.NumSamples, pool.NumSamples, borders);

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

        for (size_t i = 0; i < pool.Queries.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(pool.Queries[i].size(), dataProvider.GetQueries()[i].size());
            for (size_t j = 0; j < pool.Queries[i].size(); ++j) {
                UNIT_ASSERT_VALUES_EQUAL(pool.Queries[i][j], dataProvider.GetQueries()[i][j]);
            }
        }

        {
            ymap<ui32, ui32> gidsBins;

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
