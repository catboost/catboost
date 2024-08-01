#include "model_test_helpers.h"

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/ut/lib/for_loader.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/json/json_value.h>

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
            metaInfo.TargetType = ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                factorCount,
                TVector<ui32>{},
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
                    MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(vec))
                );
            }

            TVector<float> vec(docCount);
            for (auto& val : vec) {
                val = rng.GenRandReal1();
            }
            visitor->AddTarget(MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(vec)));

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
    srcData.DatasetFileData =
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
    TVector<NJson::TJsonValue> classLabels;

    return ReadDataset(
        /*taskType*/Nothing(),
        readDatasetMainParams.PoolPath,
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        readDatasetMainParams.ColumnarPoolFormatParams,
        /*ignoredFeatures*/ {},
        EObjectsOrder::Undefined,
        /*threadCount*/ 16,
        /*verbose*/true,
        /*loadSampleIds*/ false,
        /*forceUnitAutoPairWeights*/ false,
        &classLabels
    );
}

TDataProviderPtr GetMultiClassPool() {
    TSrcData srcData;
    srcData.DatasetFileData = // Cloudness pool first 9 features
        "0\t0\t1481112000\t1\t52.3699989319\t143.179992676\t1481112000\t41924.0\t0.545332845052\t1\n"
        "1\t0\t1484784000\t1\t52.0999984741\t102.699996948\t1484784000\t41777.0\t9.84666646322\t1\n"
        "2\t0\t1479567600\t1\t59.0166702271\t54.6500015259\t1479567600\t41493.0\t21.6433334351\t1\n"
        "2\t0\t1486112400\t1\t53.3499984741\t75.4499969482\t1486112400\t40746.0\t17.0299997965\t1\n"
        "2\t0\t1479772800\t1\t53.8824996948\t28.0307006836\t1479772800\t33919.0\t4.86871337891\t0\n"
        "0\t0\t1481760000\t1\t48.2200012207\t46.7299995422\t1481760000\t49599.0\t6.11533330282\t0\n"
        "2\t0\t1481878800\t1\t59.6500015259\t154.270004272\t1481878800\t41483.0\t22.2846669515\t0\n"
        "1\t0\t1482742800\t1\t54.2999992371\t155.916671753\t1482742800\t41947.0\t22.3944447835\t0\n"
        "2\t0\t1480431600\t1\t48.0666694641\t46.1166687012\t1480431600\t49598.0\t21.0744445801\t0\n"
        "1\t0\t1484872200\t1\t56.3802986145\t85.2082977295\t1484872200\t33961.0\t8.68055318197\t0\n"
        "2\t0\t1486090800\t1\t51.8699989319\t58.1800003052\t1486090800\t41961.0\t9.87866668701\t0\n"
        "2\t0\t1485896400\t1\t58.5200004578\t58.8499984741\t1485896400\t41495.0\t3.92333323161\t0\n"
        "0\t0\t1485653400\t1\t46.8886985779\t142.718002319\t1485653400\t33779.0\t13.514533488\t1\n"
        "2\t0\t1482323400\t1\t55.9725990295\t37.4146003723\t1482323400\t34193.0\t17.4943066915\t0\n"
        "1\t0\t1485066600\t1\t51.0222015381\t71.4669036865\t1485066600\t33632.0\t13.7644602458\t1\n"
        "0\t0\t1480644000\t1\t55.3058013916\t61.5032997131\t1480644000\t34003.0\t9.10021998088\t0";

    TStringBuilder cdOutput;
    cdOutput << "0\tTarget\n" << "9\tCateg" << Endl;

    TReadDatasetMainParams readDatasetMainParams;

    TVector<THolder<TTempFile>> srcDataFiles;
    SaveSrcData(srcData, &readDatasetMainParams, &srcDataFiles);
    TVector<NJson::TJsonValue> classLabels;

    return ReadDataset(
        /*taskType*/Nothing(),
        readDatasetMainParams.PoolPath,
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        TPathWithScheme(),
        readDatasetMainParams.ColumnarPoolFormatParams,
        /*ignoredFeatures*/ {},
        EObjectsOrder::Undefined,
        /*threadCount*/ 16,
        /*verbose*/true,
        /*loadSampleIds*/ false,
        /*forceUnitAutoPairWeights*/ false,
        &classLabels
    );
}

TFullModel SimpleFloatModel(size_t treeCount) {
    TFullModel model;
    TModelTrees* trees = model.ModelTrees.GetMutable();
    trees->SetFloatFeatures(
        {
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
        }
    );
    for (auto i : xrange(301)) {
        trees->AddFloatFeatureBorder(0, -298.0f + i);
    }
    {
        double tenPower = 1.0;
        for (size_t treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
            TVector<int> tree = {300, 301, 302};
            trees->AddBinTree(tree);
            for (int leafIndex = 0; leafIndex < 8; ++leafIndex) {
                trees->AddLeafValue(leafIndex * tenPower);
            }
            tenPower *= 10.0;
        }
    }
    model.UpdateDynamicData();
    return model;
}

TFullModel SimpleTextModel(
    TIntrusivePtr<TTextProcessingCollection> textCollection,
    TConstArrayRef<TVector<TStringBuf>> textFeaturesValues,
    TArrayRef<double> expectedResults
) {
    TFullModel model;

    model.TextProcessingCollection = textCollection;
    TVector<TTextFeature> textFeatures;
    const int textFeatureCount = static_cast<int>(textCollection->GetTextFeatureCount());

    for (int featureId = 0; featureId < textFeatureCount; featureId++) {
        TTextFeature textFeature(true, featureId, featureId, ToString(featureId));
        textFeatures.emplace_back(textFeature);
    }

    TObliviousTreeBuilder treeBuilder(TVector<TFloatFeature>{}, TVector<TCatFeature>{}, textFeatures, TVector<TEmbeddingFeature>{}, 1);

    {
        const int docCount = textFeaturesValues[0].size();
        TVector<float> estimatedFeatures;
        estimatedFeatures.yresize(docCount * textCollection->TotalNumberOfOutputFeatures());
        {
            TVector<ui32> textFeatureIds = xrange(textFeatureCount);
            textCollection->CalcFeatures(
                [&](ui32 textFeatureIdx, ui32 docId) { return textFeaturesValues[textFeatureIdx][docId]; },
                MakeConstArrayRef(textFeatureIds),
                docCount,
                MakeArrayRef(estimatedFeatures)
            );
        }

        ui32 estimatedFeatureIdx = 0;
        for (const auto& producedFeature: textCollection->GetProducedFeatures()) {
            TEstimatedFeatureSplit estimatedFeatureSplit(TModelEstimatedFeature{
                SafeIntegerCast<int>(producedFeature.FeatureId),
                producedFeature.CalcerId,
                SafeIntegerCast<int>(producedFeature.LocalId),
                EEstimatedSourceFeatureType::Text},
                /* split */ 0.f
            );

            const ui32 calcerOffset = textCollection->GetAbsoluteCalcerOffset(producedFeature.CalcerId);
            const ui32 estimatedFeatureOffset = calcerOffset + producedFeature.LocalId;
            TArrayRef<float> estimatedFeatureValues(
                estimatedFeatures.data() + estimatedFeatureOffset * docCount,
                estimatedFeatures.data() + (estimatedFeatureOffset + 1) * docCount
            );

            {
                float mean = 0.;
                for (auto featureValue : estimatedFeatureValues) {
                    mean += featureValue;
                }
                mean /= (float) estimatedFeatureValues.size();

                estimatedFeatureSplit.Split = mean;
            }

            TVector<double> treeLeafValues = {estimatedFeatureIdx * 10., estimatedFeatureIdx * 10. + 1};
            TVector<double> treeLeafWeights = {1., 1.};
            treeBuilder.AddTree({TModelSplit(estimatedFeatureSplit)}, treeLeafValues, treeLeafWeights);

            for (ui32 docId : xrange(docCount)) {
                double estimatedFeatureValue = estimatedFeatures[estimatedFeatureOffset * docCount + docId];
                expectedResults[estimatedFeatureOffset * docCount + docId] =
                    (estimatedFeatureValue <= estimatedFeatureSplit.Split) ?
                    treeLeafValues[0] : treeLeafValues[1];
            }
            estimatedFeatureIdx++;
        }
    }

    treeBuilder.Build(model.ModelTrees.GetMutable());
    model.UpdateDynamicData();

    return model;
}

TFullModel SimpleDeepTreeModel(size_t treeDepth) {
    TFullModel model;
    TModelTrees* trees = model.ModelTrees.GetMutable();
    for (size_t featureIndex : xrange(treeDepth)) {
        const auto feature = TFloatFeature(false, featureIndex, featureIndex, {0.5f}, "");
        trees->AddFloatFeature(feature);
    }
    for (size_t val : xrange(1 << treeDepth)) {
        trees->AddLeafValue(val);
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

    TNonSymmetricTreeModelBuilder builder(floatFeatures, TVector<TCatFeature>{}, TVector<TTextFeature>{}, TVector<TEmbeddingFeature>{}, 1);

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
    builder.Build(model.ModelTrees.GetMutable());
    model.UpdateDynamicData();
    return model;
}

TFullModel DefaultTrainCatOnlyModel(const NJson::TJsonValue& params) {
    TDataProviders dataProviders;
    dataProviders.Learn = CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.TargetType = ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                (ui32)3,
                TVector<ui32>{0, 1, 2},
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{});

            visitor->Start(metaInfo, 3, EObjectsOrder::Undefined, {});

            visitor->AddCatFeature(0, TConstArrayRef<TStringBuf>{"a", "a", "b"});
            visitor->AddCatFeature(1, TConstArrayRef<TStringBuf>{"d", "c", "d"});
            visitor->AddCatFeature(2, TConstArrayRef<TStringBuf>{"e", "f", "f"});

            visitor->AddTarget(
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>{1.0f, 0.0f, 0.2f})
            );

            visitor->Finish();
        }
    );
    dataProviders.Test.push_back(dataProviders.Learn);

    TFullModel model;
    TEvalResult evalResult;

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

    return model;
}

TFullModel TrainCatOnlyModel() {
    TTempDir trainDir;
    NJson::TJsonValue params;
    params.InsertValue("iterations", 5);
    params.InsertValue("random_seed", 1);
    params.InsertValue("train_dir", trainDir.Name());
    return DefaultTrainCatOnlyModel(params);
}

TFullModel TrainCatOnlyNoOneHotModel() {
    TTempDir trainDir;
    NJson::TJsonValue params;
    params.InsertValue("iterations", 5);
    params.InsertValue("random_seed", 1);
    params.InsertValue("train_dir", trainDir.Name());
    params.InsertValue("one_hot_max_size", 0);
    return DefaultTrainCatOnlyModel(params);
}

TFullModel MultiValueFloatModel() {
    TFullModel model;
    TModelTrees* trees = model.ModelTrees.GetMutable();
    trees->SetFloatFeatures(
        {
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
        }
    );
    {
        TVector<int> tree = {0, 1};
        trees->AddBinTree(tree);
        trees->SetLeafValues(
            {
                00., 10., 20.,
                01., 11., 21.,
                02., 12., 22.,
                03., 13., 23.
            }
        );
        trees->SetApproxDimension(3);
    }
    model.UpdateDynamicData();
    return model;
}
