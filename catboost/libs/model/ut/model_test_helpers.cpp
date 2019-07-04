#include "model_test_helpers.h"

#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/data_new/load_data.h>
#include <catboost/libs/data_new/ut/lib/for_loader.h>
#include <catboost/libs/train_lib/train_model.h>

#include <util/string/builder.h>
#include <util/folder/tempdir.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


TFullModel TrainFloatCatboostModel(int iterations, int seed) {
    TFastRng64 rng(seed);
    const ui32 docCount = 50000;
    const ui32 factorCount = 3;

    TDataProviders dataProviders;
    dataProviders.Learn = CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.HasTarget = true;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                factorCount,
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{});

            visitor->Start(metaInfo, docCount, EObjectsOrder::Undefined, {});

            for (auto factorId : xrange(factorCount)) {
                TVector<float> vec(docCount);
                for (auto& val : vec) {
                    val = rng.GenRandReal1();
                }
                visitor->AddFloatFeature(
                    factorId,
                    TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(vec))
                );
            }

            TVector<float> vec(docCount);
            for (auto& val : vec) {
                val = rng.GenRandReal1();
            }
            visitor->AddTarget(vec);

            visitor->Finish();
        }
    );
    dataProviders.Test.push_back(dataProviders.Learn);


    TFullModel model;
    TEvalResult evalResult;
    NJson::TJsonValue params;
    params.InsertValue("iterations", iterations);
    params.InsertValue("random_seed", seed);
    TrainModel(
            params,
            nullptr,
            Nothing(),
            Nothing(),
            std::move(dataProviders),
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            "",
            &model,
            {&evalResult}
    );

    return model;
}

TDataProviderPtr GetAdultPool() {
    TSrcData srcData;
    srcData.DsvFileData =
        "0\t1\tn\t1\t57.0\tSelf-emp-not-inc\t174760.0\tAssoc-acdm\t12.0\tMarried-spouse-absent\tFarming-fishing\tUnmarried\tAmer-Indian-Eskimo\tMale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t1\tn\t1\t26.0\tPrivate\t94392.0\t11th\t7.0\tSeparated\tOther-service\tUnmarried\tWhite\tFemale\t0.0\t0.0\t20.0\tUnited-States\n"
        "0\t1\tn\t1\t55.0\tPrivate\t451603.0\tHS-grad\t9.0\tDivorced\tCraft-repair\tUnmarried\tWhite\tMale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t1\tn\t1\t52.0\tPrivate\t84788.0\t10th\t6.0\tNever-married\tMachine-op-inspct\tNot-in-family\tWhite\tMale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t1\tn\t1\t23.0\tPrivate\t204641.0\t10th\t6.0\tNever-married\tHandlers-cleaners\tUnmarried\tWhite\tMale\t0.0\t0.0\t50.0\tUnited-States\n"
        "0\t1\tn\t1\t44.0\t?\t210875.0\t11th\t7.0\tDivorced\t?\tNot-in-family\tBlack\tFemale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t51.0\tLocal-gov\t117496.0\tMasters\t14.0\tMarried-civ-spouse\tProf-specialty\tWife\tWhite\tFemale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t44.0\tPrivate\t238574.0\tProf-school\t15.0\tMarried-civ-spouse\tProf-specialty\tHusband\tWhite\tMale\t0.0\t0.0\t50.0\tUnited-States\n"
        "0\t0\tn\t1\t62.0\t?\t191118.0\tSome-college\t10.0\tMarried-civ-spouse\t?\tHusband\tWhite\tMale\t7298.0\t0.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t46.0\tSelf-emp-not-inc\t198759.0\tProf-school\t15.0\tMarried-civ-spouse\tProf-specialty\tHusband\tWhite\tMale\t0.0\t2415.0\t80.0\tUnited-States\n"
        "0\t0\tn\t1\t37.0\tPrivate\t215503.0\tHS-grad\t9.0\tMarried-civ-spouse\tExec-managerial\tHusband\tWhite\tMale\t4386.0\t0.0\t45.0\tUnited-States\n"
        "0\t0\tn\t1\t49.0\tSelf-emp-inc\t158685.0\tHS-grad\t9.0\tMarried-civ-spouse\tAdm-clerical\tWife\tWhite\tFemale\t0.0\t2377.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t52.0\tLocal-gov\t311569.0\tMasters\t14.0\tMarried-civ-spouse\tExec-managerial\tHusband\tWhite\tMale\t0.0\t0.0\t50.0\tUnited-States";

    TStringBuilder cdOutput;
    cdOutput << "1\tTarget" << Endl;
    TVector<int> categIdx = {0, 2, 3, 5, 7, 9, 10, 11, 12, 13, 17};
    for (int i : categIdx) {
        cdOutput << i << "\tCateg" << Endl;
    }
    srcData.CdFileData = cdOutput;

    TReadDatasetMainParams readDatasetMainParams;

    TVector<THolder<TTempFile>> srcDataFiles;
    SaveSrcData(srcData, &readDatasetMainParams, &srcDataFiles);
    TVector<TString> classNames;

    return ReadDataset(
        readDatasetMainParams.PoolPath,
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        readDatasetMainParams.DsvPoolFormatParams,
        /*ignoredFeatures*/ {},
        EObjectsOrder::Undefined,
        /*threadCount*/ 16,
        /*verbose*/true,
        &classNames
    );
}

TFullModel SimpleFloatModel(size_t treeCount) {
    TFullModel model;
    TObliviousTrees* trees = model.ObliviousTrees.GetMutable();
    trees->FloatFeatures = {
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
        trees->FloatFeatures[0].Borders.push_back(-298.0f + i);
    }
    {
        double tenPower = 1.0;
        for (size_t treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
            TVector<int> tree = {300, 301, 302};
            trees->AddBinTree(tree);
            for (int leafIndex = 0; leafIndex < 8; ++leafIndex) {
                trees->LeafValues.push_back(leafIndex * tenPower);
            }
            tenPower *= 10.0;
        }
    }
    model.UpdateDynamicData();
    return model;
}

TFullModel SimpleDeepTreeModel(size_t treeDepth) {
    TFullModel model;
    TObliviousTrees* trees = model.ObliviousTrees.GetMutable();
    for (size_t featureIndex : xrange(treeDepth)) {
        const auto feature = TFloatFeature(false, featureIndex, featureIndex, {0.5f}, "");
        trees->FloatFeatures.push_back(feature);
    }
    for (size_t val : xrange(1 << treeDepth)) {
        trees->LeafValues.push_back(val);
    }
    TVector<int> tree = xrange(treeDepth);
    trees->AddBinTree(tree);
    model.UpdateDynamicData();
    return model;
}

TFullModel SimpleAsymmetricModel() {
    TVector<TFloatFeature> floatFeatures {
        TFloatFeature{
            false, 0, 0,
            {},
            ""
        },
        TFloatFeature{
            false, 1, 1,
            {},
            ""
        },
        TFloatFeature{
            false, 2, 2,
            {},
            ""
        }
    };

    TNonSymmetricTreeModelBuilder builder(floatFeatures, TVector<TCatFeature>{}, 1);

    THolder<TNonSymmetricTreeNode> treeHead = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->SplitCondition = TModelSplit(TFloatSplit(0, 0.5));

    treeHead->Left = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Left->SplitCondition = TModelSplit(TFloatSplit(1, 0.5));

    treeHead->Left->Left = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Left->Left->Value = 1.0;


    treeHead->Left->Right = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Left->Right->Value = 2.0;

    treeHead->Right = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Right->Value = 3.0;

    builder.AddTree(std::move(treeHead));
    treeHead = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->SplitCondition = TModelSplit(TFloatSplit(2, 0.5));
    treeHead->Left = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Left->Value = 0.0;
    treeHead->Right = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Right->Value = 10.0;
    builder.AddTree(std::move(treeHead));

    treeHead = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->SplitCondition = TModelSplit(TFloatSplit(0, 0.5));
    treeHead->Left = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Left->Value = 100.0;

    treeHead->Right = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Right->SplitCondition = TModelSplit(TFloatSplit(1, 0.5));

    treeHead->Right->Left = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Right->Left->Value = 200.0;


    treeHead->Right->Right = MakeHolder<TNonSymmetricTreeNode>();
    treeHead->Right->Right->Value = 300.0;

    builder.AddTree(std::move(treeHead));

    TFullModel model;
    builder.Build(model.ObliviousTrees.GetMutable());
    model.UpdateDynamicData();
    return model;
}

TFullModel TrainCatOnlyModel() {
    TTempDir trainDir;

    TDataProviders dataProviders;
    dataProviders.Learn = CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.HasTarget = true;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                (ui32)3,
                TVector<ui32>{0, 1, 2},
                TVector<ui32>{},
                TVector<TString>{});

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
        /*initModel*/ Nothing(),
        /*initLearnProgress*/ nullptr,
        "",
        &model,
        {&evalResult}
    );

    return model;
}

TFullModel MultiValueFloatModel() {
    TFullModel model;
    TObliviousTrees* trees = model.ObliviousTrees.GetMutable();
    trees->FloatFeatures = {
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
        trees->AddBinTree(tree);
        trees->LeafValues = {
            00., 10., 20.,
            01., 11., 21.,
            02., 12., 22.,
            03., 13., 23.
        };
        trees->ApproxDimension = 3;
    }
    model.UpdateDynamicData();
    return model;
}
