
#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/unittest/registar.h>
#include <library/json/json_reader.h>

#include <util/random/fast.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TTrainTest) {
    Y_UNIT_TEST(TestRepeatableTrain) {
        const size_t TestDocCount = 1000;
        const ui32 FactorCount = 10;

        TReallyFastRng32 rng(123);

        TVector<float> target(TestDocCount);
        TVector<TVector<float>> features(FactorCount); // [featureIdx][objectIdx]

        for (size_t j = 0; j < FactorCount; ++j) {
            features[j].yresize(TestDocCount);
        }

        for (size_t i = 0; i < TestDocCount; ++i) {
            target[i] = rng.GenRandReal2();
            for (size_t j = 0; j < FactorCount; ++j) {
                features[j][i] = rng.GenRandReal2();
            }
        }

        TDataProviders dataProviders;
        dataProviders.Learn = CreateDataProvider(
            [&] (IRawFeaturesOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.HasTarget = true;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                    FactorCount,
                    TVector<ui32>{},
                    TVector<TString>{},
                    nullptr);

                visitor->Start(metaInfo, TestDocCount, EObjectsOrder::Undefined, {});

                for (auto factorId : xrange(FactorCount)) {
                    visitor->AddFloatFeature(
                        factorId,
                        TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>(features[factorId]))
                    );
                }
                visitor->AddTarget(target);

                visitor->Finish();
            }
        );
        dataProviders.Test.push_back(
            CreateDataProvider(
                [&] (IRawFeaturesOrderDataVisitor* visitor) {
                    TDataMetaInfo metaInfo = dataProviders.Learn->MetaInfo;
                    visitor->Start(metaInfo, 0, EObjectsOrder::Undefined, {});

                    for (auto factorId : xrange(FactorCount)) {
                        visitor->AddFloatFeature(
                            factorId,
                            TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>())
                        );
                    }

                    visitor->AddTarget(TConstArrayRef<TString>());

                    visitor->Finish();
                }
            )
        );

        NJson::TJsonValue plainFitParams;
        plainFitParams.InsertValue("random_seed", 5);
        plainFitParams.InsertValue("iterations", 1);
        plainFitParams.InsertValue("train_dir", ".");
        plainFitParams.InsertValue("thread_count", 1);
        NJson::TJsonValue metadata;
        metadata["a"] = "b";
        plainFitParams.InsertValue("metadata", metadata);
        TEvalResult testApprox;
        TFullModel model;
        TrainModel(
            plainFitParams,
            nullptr,
            Nothing(),
            Nothing(),
            dataProviders,
            "",
            &model,
            {&testApprox}
        );
        {
            TrainModel(
                plainFitParams,
                nullptr,
                Nothing(),
                Nothing(),
                dataProviders,
                "model_for_test.cbm",
                nullptr,
                {&testApprox}
            );
            TFullModel otherCallVariant = ReadModel("model_for_test.cbm");
            UNIT_ASSERT(model.ModelInfo.contains("a"));
            UNIT_ASSERT_VALUES_EQUAL(model.ModelInfo["a"], "b");
            UNIT_ASSERT_EQUAL(model, otherCallVariant);
        }

        UNIT_ASSERT_EQUAL((size_t)dataProviders.Learn->ObjectsData->GetObjectCount(), TestDocCount);
        UNIT_ASSERT_EQUAL((size_t)dataProviders.Learn->MetaInfo.GetFeatureCount(), FactorCount);

        const auto& rawObjectsData = dynamic_cast<TRawObjectsDataProvider&>(
            *(dataProviders.Learn->ObjectsData)
        );
        for (size_t j = 0; j < FactorCount; ++j) {
            UNIT_ASSERT( Equal<float>(features[j], (**rawObjectsData.GetFloatFeature(j)).GetArrayData()) );
        }
    }
}
