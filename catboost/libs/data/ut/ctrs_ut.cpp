#include <catboost/libs/data/ctrs.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TCtrs) {
    Y_UNIT_TEST(TPrecomputedOnlineCtrMetaData_Append) {
        TPrecomputedOnlineCtrMetaData data0;
        data0.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 0}] = 0;
        data0.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 1}] = 1;
        data0.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 2}] = 2;
        data0.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 1, 0, 0}] = 3;
        data0.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 1, 0, 1}] = 4;

        data0.ValuesCounts[0] = TOnlineCtrUniqValuesCounts{5, 6};

        TPrecomputedOnlineCtrMetaData data1;
        data1.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{1, 0, 0, 0}] = 0;
        data1.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{1, 0, 0, 1}] = 1;

        data1.ValuesCounts[1] = TOnlineCtrUniqValuesCounts{3, 8};

        data0.Append(data1);

        TPrecomputedOnlineCtrMetaData expectedData;
        expectedData.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 0}] = 0;
        expectedData.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 1}] = 1;
        expectedData.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 2}] = 2;
        expectedData.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 1, 0, 0}] = 3;
        expectedData.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 1, 0, 1}] = 4;
        expectedData.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{1, 0, 0, 0}] = 5;
        expectedData.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{1, 0, 0, 1}] = 6;

        expectedData.ValuesCounts[0] = TOnlineCtrUniqValuesCounts{5, 6};
        expectedData.ValuesCounts[1] = TOnlineCtrUniqValuesCounts{3, 8};

        UNIT_ASSERT_EQUAL(data0, expectedData);
    }

    Y_UNIT_TEST(TPrecomputedOnlineCtrMetaData_Serialize) {
        TPrecomputedOnlineCtrMetaData data;
        data.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 0}] = 0;
        data.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 1}] = 1;
        data.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 0, 0, 2}] = 2;
        data.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 1, 0, 0}] = 3;
        data.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{0, 1, 0, 1}] = 4;
        data.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{1, 0, 0, 0}] = 5;
        data.OnlineCtrIdxToFeatureIdx[TOnlineCtrIdx{1, 0, 0, 1}] = 6;

        data.ValuesCounts[0] = TOnlineCtrUniqValuesCounts{5, 6};
        data.ValuesCounts[1] = TOnlineCtrUniqValuesCounts{3, 8};

        TPrecomputedOnlineCtrMetaData deserializedData
            = TPrecomputedOnlineCtrMetaData::DeserializeFromJson(data.SerializeToJson());
        UNIT_ASSERT_EQUAL(data, deserializedData);
    }
}
