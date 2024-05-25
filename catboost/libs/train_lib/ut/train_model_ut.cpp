
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/folder/tempdir.h>
#include <util/generic/array_ref.h>
#include <util/generic/xrange.h>
#include <util/random/fast.h>

#include <limits>


using namespace NCB;


template <typename Prng>
static void FillWithRandom(TArrayRef<TVector<float>> matrix, Prng& prng) {
    for (auto& row : matrix) {
        for (auto& cell : row) {
            cell = prng.GenRandReal1();
        }
    }
}

template <typename Prng>
static void FillWithRandom(TArrayRef<float> array, Prng& prng) {
    for (auto& v : array) {
        v = prng.GenRandReal1();
    }
}

static TDataProviderPtr SmallFloatPool() {
    return CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.TargetType = ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                (ui32)3,
                TVector<ui32>{},
                TVector<TString>{}
            );

            visitor->Start(metaInfo, 3, EObjectsOrder::Undefined, {});

            visitor->AddFloatFeature(
                0,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+0.5f, +1.5f, -2.5f})
            );
            visitor->AddFloatFeature(
                1,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+0.7f, +6.4f, +2.4f})
            );
            visitor->AddFloatFeature(
                2,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{-2.0f, -1.0f, +6.0f})
            );

            visitor->AddTarget(
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f, 0.0f, 0.2f})
            );

            visitor->Finish();
        }
    );
}

Y_UNIT_TEST_SUITE(TrainModelTests) {
    Y_UNIT_TEST(TrainWithoutNansTestWithNans) {
        // Train doesn't have NaNs, so TrainModel implicitly forbids them (during quantization), but
        // test data has NaNs and we just allow that
        //
        // See MLTOOLS-1602 and MLTOOLS-2235 for details (though there aren't much details).
        //
        TTempDir trainDir;

        TDataProviders dataProviders;


        dataProviders.Learn = SmallFloatPool();

        dataProviders.Test.emplace_back(
            CreateDataProvider(
                [&] (IRawFeaturesOrderDataVisitor* visitor) {
                    visitor->Start(dataProviders.Learn->MetaInfo, 1, EObjectsOrder::Undefined, {});

                    visitor->AddFloatFeature(
                        0,
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(
                            TVector<float>{std::numeric_limits<float>::quiet_NaN()}
                        )
                    );
                    visitor->AddFloatFeature(
                        1,
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{+1.5f})
                    );
                    visitor->AddFloatFeature(
                        2,
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{-2.5f})
                    );
                    visitor->AddTarget(
                        MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f})
                    );

                    visitor->Finish();
                }
            )
        );

        TFullModel model;
        TEvalResult evalResult;
        NJson::TJsonValue params;
        params.InsertValue("iterations", 5);
        params.InsertValue("random_seed", 1);
        params.InsertValue("train_dir", trainDir.Name());

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
                {&evalResult}
            );
        };

        UNIT_ASSERT_NO_EXCEPTION(f());
    }

    Y_UNIT_TEST(TrainWithoutNansApplyWithNans) {
        // Train doesn't have NaNs, so TrainModel implicitly forbids them (during quantization), but
        // during model application we allow NaNs (because it's too expensive to check for their
        // presence).
        //
        // See MLTOOLS-1602 and MLTOOLS-2235 for details (though there aren't much details).
        //
        TTempDir trainDir;

        TDataProviders dataProviders;
        dataProviders.Learn = SmallFloatPool();
        dataProviders.Test.push_back(dataProviders.Learn);

        TFullModel model;
        TEvalResult evalResult;
        NJson::TJsonValue params;
        params.InsertValue("iterations", 5);
        params.InsertValue("random_seed", 1);
        params.InsertValue("train_dir", trainDir.Name());
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
            {&evalResult}
        );

        const float numeric[] = {std::numeric_limits<float>::quiet_NaN(), +1.5f, -2.5f};
        double predictions[1];
        const auto f = [&] { model.Calc(numeric, {}, predictions); };
        UNIT_ASSERT_NO_EXCEPTION(f());
    }

    Y_UNIT_TEST(TrainWithDifferentRandomStrength) {
        // In general models trained with different random strength (--random-strength) should be
        // different.
        //
        // issue: MLTOOLS-2464

        const ui64 seed = 20181029;
        const ui32 objectCount = 100;
        const ui32 numericFeatureCount = 2;
        const double randomStrength[2] = {2., 5000.};

        TFullModel models[2];
        for (size_t i = 0; i < 2; ++i) {
            TTempDir trainDir;

            TVector<TVector<float>> factors(numericFeatureCount);
            ResizeRank2(numericFeatureCount, objectCount, factors);

            TVector<float> target(objectCount);

            TFastRng<ui64> prng(seed);
            FillWithRandom(factors, prng);
            FillWithRandom(target, prng);

            TDataProviders dataProviders;
            dataProviders.Learn = CreateDataProvider(
                [&] (IRawFeaturesOrderDataVisitor* visitor) {
                    TDataMetaInfo metaInfo;
                    metaInfo.TargetType = ERawTargetType::Float;
                    metaInfo.TargetCount = 1;
                    metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                        numericFeatureCount,
                        TVector<ui32>{},
                        TVector<ui32>{},
                        TVector<ui32>{},
                        TVector<TString>{});

                    visitor->Start(metaInfo, objectCount, EObjectsOrder::Undefined, {});

                    for (auto featureIdx : xrange(numericFeatureCount)) {
                        visitor->AddFloatFeature(
                            featureIdx,
                            MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(factors[featureIdx]))
                        );
                    }
                    visitor->AddTarget(MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(target)));

                    visitor->Finish();
                }
            );
            dataProviders.Test.push_back(dataProviders.Learn);

            TEvalResult evalResult;
            NJson::TJsonValue params;
            params.InsertValue("iterations", 20);
            params.InsertValue("random_seed", 1);
            params.InsertValue("train_dir", trainDir.Name());
            params.InsertValue("random_strength", randomStrength[i]);
            params.InsertValue("boosting_type", "Plain");
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
                &models[i],
                {&evalResult}
            );
        }

        TVector<float> object(numericFeatureCount);
        {
            TFastRng<ui64> prng(seed);
            prng.Advance(objectCount * numericFeatureCount);
            FillWithRandom(object, prng);
        }

        double predictions[2][1];
        models[0].Calc(object, {}, predictions[0]);
        models[1].Calc(object, {}, predictions[1]);

        UNIT_ASSERT_VALUES_UNEQUAL(predictions[0][0], predictions[1][0]);
    }
}
