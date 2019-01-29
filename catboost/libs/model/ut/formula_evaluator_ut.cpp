#include <library/unittest/registar.h>

#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>

#include <util/folder/tempdir.h>


using namespace NCB;


static TFullModel SimpleFloatModel() {
    TFullModel model;
    model.ObliviousTrees.FloatFeatures = {
        TFloatFeature{
            false, 0, 0,
            {}, // bin splits 0, 1
            ""
        },
        TFloatFeature{
            false, 1, 1,
            {0.5f}, // bin split 2
            ""
        },
        TFloatFeature{
            false, 2, 2,
            {0.5f}, // bin split 3
            ""
        }
    };
    for (auto i : xrange(301)) {
        model.ObliviousTrees.FloatFeatures[0].Borders.push_back(-298.0f + i);
    }
    {
        TVector<int> tree = {300, 301, 302};
        model.ObliviousTrees.AddBinTree(tree);
        model.ObliviousTrees.LeafValues = {
            {0., 1., 2., 3., 4., 5., 6., 7.}
        };
    }
    model.UpdateDynamicData();
    return model;
}

static TFullModel MultiValueFloatModel() {
    TFullModel model;
    model.ObliviousTrees.FloatFeatures = {
        TFloatFeature{
            false, 0, 0,
            {0.5f}, // bin split 0
            ""
        },
        TFloatFeature{
            false, 1, 1,
            {0.5f}, // bin split 1
            ""
        }
    };
    {
        TVector<int> tree = {0, 1};
        model.ObliviousTrees.AddBinTree(tree);
        model.ObliviousTrees.LeafValues = {
            {00., 10., 20.,
             01., 11., 21.,
             02., 12., 22.,
             03., 13., 23.}
        };
        model.ObliviousTrees.ApproxDimension = 3;
    }
    model.UpdateDynamicData();
    return model;
}

// Deterministically train model that has only 3 categoric features.
static TFullModel TrainCatOnlyModel() {
    TTempDir trainDir;

    TDataProviders dataProviders;
    dataProviders.Learn = CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.HasTarget = true;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                (ui32)3,
                TVector<ui32>{0, 1, 2},
                TVector<TString>{},
                nullptr);

            visitor->Start(metaInfo, 3, EObjectsOrder::Undefined, {});

            visitor->AddCatFeature(0, TConstArrayRef<TStringBuf>{"a", "b", "c"});
            visitor->AddCatFeature(1, TConstArrayRef<TStringBuf>{"d", "e", "f"});
            visitor->AddCatFeature(2, TConstArrayRef<TStringBuf>{"g", "h", "k"});

            visitor->AddTarget({1.0f, 0.0f, 0.2f});

            visitor->Finish();
        }
    );
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
        std::move(dataProviders),
        "",
        &model,
        {&evalResult}
    );

    return model;
}

Y_UNIT_TEST_SUITE(TObliviousTreeModel) {
    Y_UNIT_TEST(TestFlatCalcFloat) {
        auto modelCalcer = SimpleFloatModel();
        TVector<double> result(8);
        TVector<TVector<float>> data = {
            {0.f, 0.f, 0.f},
            {3.f, 0.f, 0.f},
            {0.f, 1.f, 0.f},
            {3.f, 1.f, 0.f},
            {0.f, 0.f, 1.f},
            {3.f, 0.f, 1.f},
            {0.f, 1.f, 1.f},
            {3.f, 1.f, 1.f},
        };
        TVector<TConstArrayRef<float>> features(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            features[i] = data[i];
        };
        modelCalcer.CalcFlat(
            features,
            result);
        TVector<double> canonVals = {
            0., 1., 2., 3.,
            4., 5., 6., 7.};
        UNIT_ASSERT_EQUAL(canonVals, result);
    }

    Y_UNIT_TEST(TestFlatCalcMultiVal) {
        auto modelCalcer = MultiValueFloatModel();
        TVector<TVector<float>> data = {
            {0.f, 0.f},
            {1.f, 0.f},
            {0.f, 1.f},
            {1.f, 1.f}};
        TVector<TConstArrayRef<float>> features(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            features[i] = data[i];
        };
        TVector<double> result(features.size() * 3);
        modelCalcer.CalcFlat(
            features,
            result);
        TVector<double> canonVals = {
            00., 10., 20.,
            01., 11., 21.,
            02., 12., 22.,
            03., 13., 23.,
        };
        UNIT_ASSERT_EQUAL(canonVals, result);
    }

    Y_UNIT_TEST(TestCatOnlyModel) {
        const auto model = TrainCatOnlyModel();

        const auto applySingle = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}};
            double result = 0.;
            model.Calc({}, f, MakeArrayRef(&result, 1));
        };
        UNIT_ASSERT_NO_EXCEPTION(applySingle());

        const auto applyBatch = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "k"}};
            double results[3];
            model.Calc({}, f, results);
        };
        UNIT_ASSERT_NO_EXCEPTION(applyBatch());
    }
}
