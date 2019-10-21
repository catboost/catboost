#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/unittest/registar.h>

using namespace std;
using namespace NCB;

static void AssertModelSumEqualSliced(TDataProviderPtr dataProvider) {
    NJson::TJsonValue params;
    params.InsertValue("learning_rate", 0.01);
    params.InsertValue("iterations", 100);
    params.InsertValue("random_seed", 1);
    TFullModel bigModel;
    TEvalResult evalResult;

    TDataProviders dataProviders;
    dataProviders.Learn = dataProvider;
    dataProviders.Test.push_back(dataProvider);

    THolder<TLearnProgress> learnProgress;

    TrainModel(
        params,
        nullptr,
        Nothing(),
        Nothing(),
        dataProviders,
        Nothing(),
        &learnProgress,
        "",
        &bigModel,
        {&evalResult});

    auto bigResult = ApplyModelMulti(bigModel, *(dataProvider->ObjectsData));
    bigModel.ObliviousTrees.GetMutable()->DropUnusedFeatures();
    TVector<TFullModel> partModels;
    TVector<const TFullModel*> modelPtrs;
    TVector<double> modelWeights(5, 1.0);
    for (size_t i = 0; i < 5; ++i) {
        auto partModel = bigModel.CopyTreeRange(i * 20, (i + 1) * 20);
        partModel.ObliviousTrees.GetMutable()->DropUnusedFeatures();
        partModels.emplace_back(partModel);
    }
    for (auto& model : partModels) {
        modelPtrs.push_back(&model);
    }
    auto mergedModel = SumModels(modelPtrs, modelWeights);
    UNIT_ASSERT_EQUAL(*mergedModel.ObliviousTrees, *bigModel.ObliviousTrees);

    auto mergedResult = ApplyModelMulti(mergedModel, *dataProvider->ObjectsData);
    UNIT_ASSERT_VALUES_EQUAL(bigResult.ysize(), mergedResult.ysize());
    for (int idx = 0; idx < bigResult.ysize(); ++idx) {
        UNIT_ASSERT_VALUES_EQUAL(bigResult[idx], mergedResult[idx]);
    }
}

Y_UNIT_TEST_SUITE(TModelSummTests) {
    Y_UNIT_TEST(SimpleModelMerge) {
        const auto model1 = SimpleFloatModel();
        const auto model2 = SimpleFloatModel();
        const auto largeModel = SumModels({&model1, &model2}, {2.0, 1.0});
    }

    Y_UNIT_TEST(FloatModelMergeTest) {
        auto bigModel = TrainFloatCatboostModel(40);
        bigModel.ObliviousTrees.GetMutable()->DropUnusedFeatures();
        TVector<TFullModel> partModels;
        TVector<const TFullModel*> modelPtrs;
        TVector<double> modelWeights(5, 1.0);
        for (size_t i = 0; i < 5; ++i) {
            auto partModel = bigModel.CopyTreeRange(i * 8, (i + 1) * 8);
            partModel.ObliviousTrees.GetMutable()->DropUnusedFeatures();
            partModels.emplace_back(partModel);
        }
        for (auto& model : partModels) {
            modelPtrs.push_back(&model);
        }
        auto mergedModel = SumModels(modelPtrs, modelWeights);
        UNIT_ASSERT_EQUAL(*mergedModel.ObliviousTrees, *bigModel.ObliviousTrees);
    }

    Y_UNIT_TEST(SumEqualSliced) {
        AssertModelSumEqualSliced(GetAdultPool());
        AssertModelSumEqualSliced(GetMultiClassPool());
    }
}
