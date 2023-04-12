#include <catboost/libs/dataset_statistics/statistics_data_structures.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace std;
using namespace NCB;


template <typename T>
void DoSerializeDeserialize(const T& item) {
    TStringStream strStream;
    item.Save(&strStream);
    T deserializedItem;
    deserializedItem.Load(&strStream);
    UNIT_ASSERT_EQUAL(item, deserializedItem);
}

Y_UNIT_TEST_SUITE(TStatisticsSerialization) {
    Y_UNIT_TEST(TestFloatFeatureStatisticsSerialization) {
        TFloatFeatureStatistics item;
        DoSerializeDeserialize(item);
        float i = 10.5;
        while(i > -1) {
            item.Update(i--);
        }
        DoSerializeDeserialize(item);
    }
    Y_UNIT_TEST(TestFeatureStatistics) {
        TFeatureStatistics item;
        item.FloatFeatureStatistics.resize(2);

        DoSerializeDeserialize(item);
        float i = 10.5;
        while(i > -1) {
            item.FloatFeatureStatistics[0].Update(i--);
            item.FloatFeatureStatistics[1].Update(i--);
        }
        DoSerializeDeserialize(item);
    }

    Y_UNIT_TEST(TestDatasetStatistics) {
        TDatasetStatistics item;
        TDataMetaInfo metaInfo;
        metaInfo.TargetType = ERawTargetType::Float;
        metaInfo.TargetCount = 1;
        metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
            (ui32)2,
            TVector<ui32>{},
            TVector<TString>{}
        );

        item.Init(metaInfo, TFeatureCustomBorders(), TFeatureCustomBorders());

        DoSerializeDeserialize(item);
        float i = 10.5;
        while(i > -1) {
            item.FeatureStatistics.FloatFeatureStatistics[0].Update(i--);
            item.FeatureStatistics.FloatFeatureStatistics[1].Update(i--);
            item.TargetsStatistics.Update(0, i);
        }
        DoSerializeDeserialize(item);
    }
}
