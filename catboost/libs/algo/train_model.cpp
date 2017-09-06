#include "learn_context.h"
#include "train_model.h"
#include "apply.h"
#include "eval_helpers.h"
#include "train.h"
#include "helpers.h"

#include <catboost/libs/helpers/mem_usage.h>

#include <library/grid_creator/binarization.h>

#include <util/random/shuffle.h>
#include <util/generic/vector.h>

static void PrepareTargetBinary(float border, yvector<float>* target) {
    for (int i = 0; i < (*target).ysize(); ++i) {
        (*target)[i] = ((*target)[i] > border);
    }
}

void TrainModel(const NJson::TJsonValue& jsonParams,
                const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                TPool& learnPool,
                const TPool& testPool,
                const TString& outputModelPath,
                TFullModel* modelPtr,
                yvector<yvector<double>>* testApprox) {
    TrainModelBody(jsonParams, objectiveDescriptor, evalMetricDescriptor, learnPool, testPool, outputModelPath, /*clearLearnPool*/ false, modelPtr,testApprox);
}

void TrainModelBody(const NJson::TJsonValue& jsonParams,
                const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                TPool& learnPool,
                const TPool& testPool,
                const TString& outputModelPath,
                bool clearLearnPool,
                TFullModel* modelPtr,
                yvector<yvector<double>>* testApprox) {
    CB_ENSURE(!learnPool.Docs.empty(), "Train dataset is empty");

    auto sortedCatFeatures = learnPool.CatFeatures;
    Sort(sortedCatFeatures.begin(), sortedCatFeatures.end());

    if (!testPool.Docs.empty()) {
        auto catFeaturesTest = testPool.CatFeatures;
        Sort(catFeaturesTest.begin(), catFeaturesTest.end());

        CB_ENSURE(sortedCatFeatures == catFeaturesTest, "Cat features in train and test should be the same.");
    }
    if ((modelPtr == nullptr) == outputModelPath.empty()) {
        if (modelPtr == nullptr) {
            ythrow TCatboostException() << "Both modelPtr == nullptr and outputModelPath empty";
        } else {
            ythrow TCatboostException() << "Both modelPtr != nullptr and outputModelPath non empty";
        }
    }
    const int featureCount = learnPool.Docs[0].Factors.ysize();

    TLearnContext ctx(
        jsonParams,
        objectiveDescriptor,
        evalMetricDescriptor,
        featureCount,
        sortedCatFeatures,
        learnPool.FeatureId);

    if (ctx.Params.Verbose) {
        SetVerboseLogingMode();
    } else {
        SetSilentLogingMode();
    }

    auto loggingGuard = Finally([&] { SetSilentLogingMode(); });

    yvector<size_t> indices(learnPool.Docs.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (!ctx.Params.HasTime) {
        Shuffle(indices.begin(), indices.end(), ctx.Rand);
    }

    ApplyPermutation(InvertPermutation(indices), &learnPool);

    auto permutationGuard = Finally([&] { ApplyPermutation(indices, &learnPool); });

    TTrainData trainData;
    trainData.LearnSampleCount = learnPool.Docs.ysize();

    trainData.Target.reserve(learnPool.Docs.size() + testPool.Docs.size());

    bool trainHasBaseline = !learnPool.Docs[0].Baseline.empty();
    bool testHasBaseline = trainHasBaseline;
    if (!testPool.Docs.empty()) {
        testHasBaseline = !testPool.Docs[0].Baseline.empty();
    }
    if (trainHasBaseline && !testHasBaseline) {
        CB_ENSURE(false, "Baseline for test is not provided");
    }
    if (testHasBaseline && !trainHasBaseline) {
        CB_ENSURE(false, "Baseline for train is not provided");
    }
    if (trainHasBaseline && testHasBaseline && !testPool.Docs.empty()) {
        CB_ENSURE(learnPool.Docs[0].Baseline.ysize() == testPool.Docs[0].Baseline.ysize(), "Baseline dimensions differ.");
    }

    for (const auto& doc : learnPool.Docs) {
        trainData.Target.push_back(doc.Target);
        trainData.Baseline.push_back(doc.Baseline);
        trainData.Weights.push_back(doc.Weight);
    }

    float minTarget = *MinElement(trainData.Target.begin(), trainData.Target.end());
    float maxTarget = *MaxElement(trainData.Target.begin(), trainData.Target.end());
    CB_ENSURE(minTarget != maxTarget, "All targets are equal");

    for (const auto& doc : testPool.Docs) {
        trainData.Target.push_back(doc.Target);
        trainData.Baseline.push_back(doc.Baseline);
        trainData.Weights.push_back(doc.Weight);
    }

    if (ctx.Params.LossFunction == ELossFunction::Logloss) {
        PrepareTargetBinary(ctx.Params.Border, &trainData.Target);
        float minTarget = *MinElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
        float maxTarget = *MaxElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
        CB_ENSURE(minTarget == 0, "All targets are greater than border");
        CB_ENSURE(maxTarget == 1, "All targets are smaller than border");
    }

    if (IsMultiClassError(ctx.Params.LossFunction)) {
        CB_ENSURE(AllOf(trainData.Target, [](float x) { return floor(x) == x && x >= 0; }), "if loss-function is MultiClass then each target label should be nonnegative integer");
        ctx.LearnProgress.Model.ApproxDimension = GetClassesCount(trainData.Target, ctx.Params.ClassesCount);
        CB_ENSURE(ctx.LearnProgress.Model.ApproxDimension > 1, "All targets are equal");
    }

    ctx.OutputMeta();

    if (!ctx.Params.ClassWeights.empty()) {
        int dataSize = trainData.Target.ysize();
        for (int i = 0; i < dataSize; ++i) {
            CB_ENSURE(trainData.Target[i] < ctx.Params.ClassWeights.ysize(), "class " + ToString(trainData.Target[i]) + " is missing in class weights");
            trainData.Weights[i] *= ctx.Params.ClassWeights[trainData.Target[i]];
        }
    }

    ctx.InitData(trainData);

    ctx.LearnProgress.Model.Borders = GenerateBorders(learnPool.Docs, &ctx);

    learnPool.Docs.insert(learnPool.Docs.end(), testPool.Docs.begin(), testPool.Docs.end());

    auto learnPoolGuard = Finally([&] {
        learnPool.Docs.resize(trainData.LearnSampleCount);
        learnPool.Docs.shrink_to_fit();
    });

    PrepareAllFeatures(
        learnPool.Docs,
        ctx.CatFeatures,
        ctx.LearnProgress.Model.Borders,
        ctx.Params.IgnoredFeatures,
        trainData.LearnSampleCount,
        ctx.Params.OneHotMaxSize,
        ctx.Params.NanMode,
        ctx.LocalExecutor,
        &trainData.AllFeatures);

    if (clearLearnPool) {
        learnPool.Docs.clear();
        learnPool.Docs.shrink_to_fit();
    }

    float minWeight = *MinElement(trainData.Weights.begin(), trainData.Weights.begin() + trainData.LearnSampleCount);
    float maxWeight = *MaxElement(trainData.Weights.begin(), trainData.Weights.begin() + trainData.LearnSampleCount);

    if (minWeight == maxWeight) {
        trainData.Weights.clear();
        trainData.Weights.shrink_to_fit();
    } else { // Learn weight sum should be equal to learn sample count
        CB_ENSURE(minWeight > 0, "weights should be positive");
    }
    ctx.LearnProgress.Model.CatFeatures = sortedCatFeatures;
    ctx.LearnProgress.Model.FeatureIds = learnPool.FeatureId;

    DumpMemUsage("Before start train");

    testApprox->resize(ctx.LearnProgress.Model.ApproxDimension);

    TTrainFunc trainFunc;
    switch (ctx.Params.LossFunction) {
        case ELossFunction::RMSE:
            trainFunc = Train<TQuadError>;
            break;
        case ELossFunction::Logloss:
            trainFunc = Train<TBinclassError>;
            break;
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            trainFunc = Train<TQuantileError>;
            break;
        case ELossFunction::LogLinQuantile:
            trainFunc = Train<TLogLinearQuantileError>;
            break;
        case ELossFunction::Poisson:
            trainFunc = Train<TPoissonError>;
            break;
        case ELossFunction::CrossEntropy:
            trainFunc = Train<TCrossEntropyError>;
            break;
        case ELossFunction::MAPE:
            trainFunc = Train<TMAPError>;
            break;
        case ELossFunction::MultiClass:
            trainFunc = Train<TMultiClassError>;
            break;
        case ELossFunction::MultiClassOneVsAll:
            trainFunc = Train<TMultiClassOneVsAllError>;
            break;
        case ELossFunction::Custom:
            trainFunc = Train<TCustomError>;
            break;
        default:
            CB_ENSURE(false, "provided error function is not supported");
    }

    trainFunc(trainData, &ctx, testApprox);

    *testApprox = PrepareEval(ctx.Params.PredictionType, *testApprox, &ctx.LocalExecutor);

    TOneHotFeaturesInfo oneHotFeaturesInfo;
    for (const auto& bestTree : ctx.LearnProgress.Model.TreeStruct) {
        for (const auto& split : bestTree.SelectedSplits) {
            if (split.Type != ESplitType::OneHotFeature) {
                continue;
            }
            oneHotFeaturesInfo.FeatureHashToOrigString[split.OneHotFeature.Value] = learnPool.CatFeaturesHashToString.at(split.OneHotFeature.Value);
        }
    }
    if (modelPtr) {
        *modelPtr = std::move(ctx.LearnProgress.Model);
        modelPtr->OneHotFeaturesInfo = std::move(oneHotFeaturesInfo);
        yvector<TModelCtr> ctrsForModelCalcTasks;
        for (const auto& bestTree : modelPtr->TreeStruct) {
            for (const auto& split : bestTree.SelectedSplits) {
                if (split.Type != ESplitType::OnlineCtr) {
                    continue;
                }
                const auto& ctr = split.OnlineCtr.Ctr;
                if (modelPtr->CtrCalcerData.LearnCtrs.has(ctr)) {
                    continue;
                }
                modelPtr->CtrCalcerData.LearnCtrs[ctr];
                ctrsForModelCalcTasks.emplace_back(ctr);
            }
        }

        ctx.LocalExecutor.ExecRange([&](int i) {
            auto& ctr = ctrsForModelCalcTasks[i];
            TCtrValueTable* resTable = &modelPtr->CtrCalcerData.LearnCtrs.at(ctr);
            CalcFinalCtrs(ctr,
                trainData,
                ctx.LearnProgress.AveragingFold.LearnPermutation,
                ctx.LearnProgress.AveragingFold.LearnTargetClass[ctr.TargetBorderClassifierIdx],
                ctx.LearnProgress.AveragingFold.TargetClassesCount[ctr.TargetBorderClassifierIdx],
                ctx.Params.CtrLeafCountLimit,
                ctx.Params.StoreAllSimpleCtr,
                resTable);
        }, 0, ctrsForModelCalcTasks.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        yvector<TModelCtr> usedCtrs;
        {
            yhash_set<TModelCtr> ctrsSet;
            for (const auto& bestTree : ctx.LearnProgress.Model.TreeStruct) {
                for (const auto& split : bestTree.SelectedSplits) {
                    if (split.Type != ESplitType::OnlineCtr) {
                        continue;
                    }
                    ctrsSet.insert(split.OnlineCtr.Ctr);
                }
            }
            usedCtrs.assign(ctrsSet.begin(), ctrsSet.end());
        }
        TStreamedFullModelSaver saver(outputModelPath, usedCtrs.size(), ctx.LearnProgress.Model, oneHotFeaturesInfo);
        ctx.LocalExecutor.ExecRange([&](int i) {
            auto& ctr = usedCtrs[i];
            TCtrValueTable resTable;
            CalcFinalCtrs(ctr,
                trainData,
                ctx.LearnProgress.AveragingFold.LearnPermutation,
                ctx.LearnProgress.AveragingFold.LearnTargetClass[ctr.TargetBorderClassifierIdx],
                ctx.LearnProgress.AveragingFold.TargetClassesCount[ctr.TargetBorderClassifierIdx],
                ctx.Params.CtrLeafCountLimit,
                ctx.Params.StoreAllSimpleCtr,
                &resTable);
            saver.SaveOneCtr(ctr, resTable);
        }, 0, usedCtrs.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

void TrainOneIteration(const TTrainData& trainData, TLearnContext* ctx)
{
    if (ctx->Params.Verbose) {
        SetVerboseLogingMode();
    } else {
        SetSilentLogingMode();
    }

    auto loggingGuard = Finally([&] { SetSilentLogingMode(); });

    TTrainOneIterationFunc trainFunc;
    switch (ctx->Params.LossFunction) {
        case ELossFunction::RMSE:
            trainFunc = TrainOneIter<TQuadError>;
            break;
        case ELossFunction::Logloss:
            trainFunc = TrainOneIter<TBinclassError>;
            break;
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            trainFunc = TrainOneIter<TQuantileError>;
            break;
        case ELossFunction::LogLinQuantile:
            trainFunc = TrainOneIter<TLogLinearQuantileError>;
            break;
        case ELossFunction::Poisson:
            trainFunc = TrainOneIter<TPoissonError>;
            break;
        case ELossFunction::CrossEntropy:
            trainFunc = TrainOneIter<TCrossEntropyError>;
            break;
        case ELossFunction::MAPE:
            trainFunc = TrainOneIter<TMAPError>;
            break;
        case ELossFunction::MultiClass:
            trainFunc = TrainOneIter<TMultiClassError>;
            break;
        case ELossFunction::MultiClassOneVsAll:
            trainFunc = TrainOneIter<TMultiClassOneVsAllError>;
            break;
        case ELossFunction::Custom:
            trainFunc = TrainOneIter<TCustomError>;
            break;
        default:
            CB_ENSURE(false, "provided error function is not supported");
    }

    trainFunc(trainData, ctx);
}
