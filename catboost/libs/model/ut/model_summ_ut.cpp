#include "model_test_helpers.h"

#include <catboost/libs/algo/apply.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/unittest/registar.h>

using namespace std;

Y_UNIT_TEST_SUITE(TModelSummTests) {

    Y_UNIT_TEST(FloatModelMergeTest) {
        auto bigModel = TrainFloatCatboostModel(40);
        bigModel.ObliviousTrees.DropUnusedFeatures();
        TVector<TFullModel> partModels;
        TVector<const TFullModel*> modelPtrs;
        TVector<double> modelWeights(5, 1.0);
        for (size_t i = 0; i < 5; ++i) {
            auto partModel = bigModel.CopyTreeRange(i * 8, (i + 1) * 8);
            partModel.ObliviousTrees.DropUnusedFeatures();
            partModels.emplace_back(partModel);
        }
        for (auto& model : partModels) {
            modelPtrs.push_back(&model);
        }
        auto mergedModel = SumModels(modelPtrs, modelWeights);
        UNIT_ASSERT_EQUAL(mergedModel.ObliviousTrees, bigModel.ObliviousTrees);
    }

    Y_UNIT_TEST(AdultModelMergeTest) {
        NJson::TJsonValue params;
        params.InsertValue("learning_rate", 0.01);
        params.InsertValue("iterations", 100);
        params.InsertValue("random_seed", 1);
        TFullModel bigModel;
        TEvalResult evalResult;

        TPool pool = GetAdultPool();

        TrainModel(params, Nothing(), Nothing(), TClearablePoolPtrs(pool, {&pool}), "", &bigModel, {&evalResult});
        auto bigResult = ApplyModel(bigModel, pool);
        bigModel.ObliviousTrees.DropUnusedFeatures();
        TVector<TFullModel> partModels;
        TVector<const TFullModel*> modelPtrs;
        TVector<double> modelWeights(5, 1.0);
        for (size_t i = 0; i < 5; ++i) {
            auto partModel = bigModel.CopyTreeRange(i * 20, (i + 1) * 20);
            partModel.ObliviousTrees.DropUnusedFeatures();
            partModels.emplace_back(partModel);
        }
        for (auto& model : partModels) {
            modelPtrs.push_back(&model);
        }
        auto mergedModel = SumModels(modelPtrs, modelWeights);
        UNIT_ASSERT_EQUAL(mergedModel.ObliviousTrees, bigModel.ObliviousTrees);

        auto mergedResult = ApplyModel(mergedModel, pool);
        UNIT_ASSERT_VALUES_EQUAL(bigResult.ysize(), mergedResult.ysize());
        for (int idx = 0; idx < bigResult.ysize(); ++idx) {
            UNIT_ASSERT_VALUES_EQUAL(bigResult[idx], mergedResult[idx]);
        }
    }
}
