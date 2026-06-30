#include <catboost/libs/data/meta_info.h>


#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TFeatureMetaInfo) {
    Y_UNIT_TEST(Operator_Equal) {
        UNIT_ASSERT_EQUAL(
            TFeatureMetaInfo(EFeatureType::Float, "A", /*isSparse*/ false, true, false),
            TFeatureMetaInfo(EFeatureType::Float, "A", /*isSparse*/ false, true, false)
        );
        UNIT_ASSERT_EQUAL(
            TFeatureMetaInfo(EFeatureType::Float, "A", /*isSparse*/ false, true),
            TFeatureMetaInfo(EFeatureType::Float, "A", /*isSparse*/ false, true, true)
        );
        UNIT_ASSERT_UNEQUAL(
            TFeatureMetaInfo(EFeatureType::Categorical, "A",/*isSparse*/ false, true),
            TFeatureMetaInfo(EFeatureType::Float, "A", /*isSparse*/ false, true)
        );
        UNIT_ASSERT_UNEQUAL(
            TFeatureMetaInfo(EFeatureType::Categorical, "A", /*isSparse*/ false, true),
            TFeatureMetaInfo(EFeatureType::Categorical, "A1", /*isSparse*/ false, true)
        );
        UNIT_ASSERT_UNEQUAL(
            TFeatureMetaInfo(EFeatureType::Categorical, "A", /*isSparse*/ false, true),
            TFeatureMetaInfo(EFeatureType::Categorical, "A", /*isSparse*/ false, false)
        );
    }
}


Y_UNIT_TEST_SUITE(TDataMetaInfo) {
    Y_UNIT_TEST(Operator_Equal) {
        UNIT_ASSERT_EQUAL(NCB::TDataMetaInfo(), NCB::TDataMetaInfo());
        {
            NCB::TDataColumnsMetaInfo dataColumnsMetaInfo{
                {
                    {EColumn::Label, ""},
                    {EColumn::GroupId, ""},
                    {EColumn::Num, "feat1"},
                    {EColumn::Num, "feat2"}
                }
            };

            TVector<TString> featureNames{"feat1", "feat2"};

            NCB::TDataMetaInfo dataMetaInfo(
                NCB::TDataColumnsMetaInfo(dataColumnsMetaInfo),
                ERawTargetType::String,
                true,
                false,
                false,
                false,
                false,
                true,
                Nothing(),
                &featureNames
            );

            UNIT_ASSERT_EQUAL(dataMetaInfo, dataMetaInfo);

            {
                NCB::TDataMetaInfo dataMetaInfo2(
                    NCB::TDataColumnsMetaInfo(dataColumnsMetaInfo),
                    ERawTargetType::String,
                    false,
                    false,
                    false,
                    false,
                    false,
                    true,
                    Nothing(),
                    &featureNames
                );

                UNIT_ASSERT_UNEQUAL(dataMetaInfo, dataMetaInfo2);
            }

            {
                NCB::TDataColumnsMetaInfo dataColumnsMetaInfo3{
                    {
                        {EColumn::Label, ""},
                        {EColumn::GroupId, ""},
                        {EColumn::Num, "feat1"},
                        {EColumn::Num, "feat2"},
                        {EColumn::Categ, "cat_feat3"}
                    }
                };

                TVector<TString> featureNames3{"feat1", "feat2", "cat_feat3"};


                NCB::TDataMetaInfo dataMetaInfo3(
                    NCB::TDataColumnsMetaInfo(dataColumnsMetaInfo3),
                    ERawTargetType::String,
                    true,
                    false,
                    false,
                    false,
                    false,
                    true,
                    Nothing(),
                    &featureNames3
                );

                UNIT_ASSERT_UNEQUAL(dataMetaInfo, dataMetaInfo3);
            }
        }
    }
}
