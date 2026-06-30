
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/folder/tempdir.h>

#include <limits>

using namespace NCB;

Y_UNIT_TEST_SUITE(TrainModelTests) {
    Y_UNIT_TEST(TrainWithoutNansTestWithNans) {
        // Train doesn't have NaNs, so TrainModel implicitly forbids them (during quantization), and
        // test data have NaN feature, so the entire training process fails.
        //
        TTempDir trainDir;

        TDataProviders dataProviders;

        // GPU CatBoost requires at least 4*numberof_devices documents in dataset

        dataProviders.Learn = CreateDataProvider(
            [&](IRawFeaturesOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.TargetType = ERawTargetType::Float;
                metaInfo.TargetCount = 1;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                    (ui32)3,
                    TVector<ui32>{},
                    TVector<TString>{"aaa", "bbb", "ccc"});

                visitor->Start(metaInfo, 4, EObjectsOrder::Undefined, {});

                visitor->AddFloatFeature(
                    0,
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+0.5f, +1.5f, -2.5f, 0.3f}));
                visitor->AddFloatFeature(
                    1,
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+0.7f, +6.4f, +2.4f, 0.7f}));
                visitor->AddFloatFeature(
                    2,
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{-2.0f, -1.0f, +6.0f, -1.2f}));

                visitor->AddTarget(
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f, 0.0f, 0.2f, 0.0f})
                );

                visitor->Finish();
            });

        dataProviders.Test.emplace_back(
            CreateDataProvider(
                [&](IRawFeaturesOrderDataVisitor* visitor) {
                    visitor->Start(dataProviders.Learn->MetaInfo, 1, EObjectsOrder::Undefined, {});

                    visitor->AddFloatFeature(
                        0,
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(
                            TVector<float>{std::numeric_limits<float>::quiet_NaN()}));
                    visitor->AddFloatFeature(
                        1,
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+1.5f}));
                    visitor->AddFloatFeature(
                        2,
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{-2.5f}));
                    visitor->AddTarget(
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f})
                    );

                    visitor->Finish();
                }));

        TFullModel model;
        TEvalResult evalResult;
        NJson::TJsonValue params;
        params.InsertValue("iterations", 5);
        params.InsertValue("random_seed", 1);
        params.InsertValue("train_dir", trainDir.Name());
        params.InsertValue("task_type", "GPU");
        params.InsertValue("devices", "0");

        const auto f = [&] {
            TrainModel(
                params,
                nullptr,
                {},
                {},
                Nothing(),
                std::move(dataProviders),
                /*initModel*/ Nothing(),
                /*initLearnProgress*/ nullptr,
                "",
                &model,
                {&evalResult});
        };

        UNIT_ASSERT_NO_EXCEPTION(f());
    }
}
