#pragma once

#include <catboost/libs/algo/train_model.h>

TFullModel TrainFloatCatboostModel() {
    TPool pool;
    pool.Docs.resize(3);
    pool.Docs[0].Target = 1;
    pool.Docs[0].Factors.assign({0.5f, 0.7f, -2.0f});
    pool.Docs[1].Target = 0;
    pool.Docs[1].Factors.assign({1.5f, 6.4f, -1.0f});
    pool.Docs[2].Target = 0.2;
    pool.Docs[2].Factors.assign({-2.5f, 2.4f, 6.0f});

    TFullModel model;
    yvector<yvector<double>> testApprox;
    NJson::TJsonValue params;
    params.InsertValue("iterations", 5);
    TrainModel(params, Nothing(), Nothing(), pool, pool, "", &model, &testApprox);

    return model;
}
