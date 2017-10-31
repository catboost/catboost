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

static bool IsCorrectQueryIdsFormat(const yvector<ui32>& queryIds) {
    yhash_set<ui32> queryGroupIds;
    ui32 lastId = queryIds.empty() ? 0 : queryIds[0];
    for (ui32 id : queryIds) {
        if (id != lastId) {
            if (queryGroupIds.has(id)) {
                return false;
            }
            queryGroupIds.insert(lastId);
            lastId = id;
        }
    }
    return true;
}

static yhash<ui32, ui32> CalcQueriesSize(const yvector<ui32>& queriesId) {
    yhash<ui32, ui32> queriesSize;
    for (int docId = 0; docId < queriesId.ysize(); ++docId) {
        ++queriesSize[queriesId[docId]];
    }
    return queriesSize;
}

class TCPUModelTrainer : public IModelTrainer {
    void TrainModel(const NJson::TJsonValue& jsonParams,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TPool& learnPool,
        bool allowClearPool,
        const TPool& testPool,
        const TString& outputModelPath,
        TFullModel* modelPtr,
        yvector<yvector<double>>* testApprox) const override
    {
        CB_ENSURE(learnPool.Docs.GetDocCount() != 0, "Train dataset is empty");

        auto sortedCatFeatures = learnPool.CatFeatures;
        Sort(sortedCatFeatures.begin(), sortedCatFeatures.end());
        if (testPool.Docs.GetDocCount() != 0) {
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
        const int featureCount = learnPool.Docs.GetFactorsCount();

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

        yvector<size_t> indices(learnPool.Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);
        if (!(ctx.Params.HasTime || IsQuerywiseError(ctx.Params.LossFunction))) {
            Shuffle(indices.begin(), indices.end(), ctx.Rand);
        }

        ApplyPermutation(InvertPermutation(indices), &learnPool);

        auto permutationGuard = Finally([&] { ApplyPermutation(indices, &learnPool); });

        TTrainData trainData;
        trainData.LearnSampleCount = learnPool.Docs.GetDocCount();

        trainData.Target.reserve(learnPool.Docs.GetDocCount() + testPool.Docs.GetDocCount());

        trainData.Pairs.reserve(learnPool.Pairs.size() + testPool.Pairs.size());
        trainData.Pairs.insert(trainData.Pairs.end(), learnPool.Pairs.begin(), learnPool.Pairs.end());
        trainData.Pairs.insert(trainData.Pairs.end(), testPool.Pairs.begin(), testPool.Pairs.end());
        for (int pairInd = learnPool.Pairs.ysize(); pairInd < trainData.Pairs.ysize(); ++pairInd) {
            trainData.Pairs[pairInd].WinnerId += trainData.LearnSampleCount;
            trainData.Pairs[pairInd].LoserId += trainData.LearnSampleCount;
        }

        bool trainHasBaseline = learnPool.Docs.GetBaselineDimension() != 0;
        bool testHasBaseline = trainHasBaseline;
        if (testPool.Docs.GetDocCount() != 0) {
            testHasBaseline = testPool.Docs.GetBaselineDimension() != 0;
        }
        if (trainHasBaseline && !testHasBaseline) {
            CB_ENSURE(false, "Baseline for test is not provided");
        }
        if (testHasBaseline && !trainHasBaseline) {
            CB_ENSURE(false, "Baseline for train is not provided");
        }
        if (trainHasBaseline && testHasBaseline && testPool.Docs.GetDocCount() != 0) {
            CB_ENSURE(learnPool.Docs.GetBaselineDimension() == testPool.Docs.GetBaselineDimension(), "Baseline dimensions differ.");
        }

        bool nonZero = !AllOf(learnPool.Docs.Weight, [] (float weight) { return weight == 0; });
        CB_ENSURE(nonZero, "All documents have zero weights");

        trainData.Target = learnPool.Docs.Target;
        trainData.Weights = learnPool.Docs.Weight;
        trainData.QueryId = learnPool.Docs.QueryId;
        trainData.Baseline = learnPool.Docs.Baseline;

        float minTarget = *MinElement(trainData.Target.begin(), trainData.Target.end());
        float maxTarget = *MaxElement(trainData.Target.begin(), trainData.Target.end());
        CB_ENSURE(minTarget != maxTarget || IsPairwiseError(ctx.Params.LossFunction), "All targets are equal");

        trainData.Target.insert(trainData.Target.end(), testPool.Docs.Target.begin(), testPool.Docs.Target.end());
        trainData.Weights.insert(trainData.Weights.end(), testPool.Docs.Weight.begin(), testPool.Docs.Weight.end());
        trainData.QueryId.insert(trainData.QueryId.end(), testPool.Docs.QueryId.begin(), testPool.Docs.QueryId.end());
        for (int dim = 0; dim < testPool.Docs.GetBaselineDimension(); ++dim) {
            trainData.Baseline[dim].insert(trainData.Baseline[dim].end(), testPool.Docs.Baseline[dim].begin(), testPool.Docs.Baseline[dim].end());
        }

        if (IsQuerywiseError(ctx.Params.LossFunction)) {
            bool isDataCorrect = IsCorrectQueryIdsFormat(trainData.QueryId);
            if (testPool.Docs.GetDocCount() != 0) {
                isDataCorrect &= learnPool.Docs.QueryId.back() != testPool.Docs.QueryId.front();
            }
            CB_ENSURE(isDataCorrect , "Train Pool & Test Pool should be grouped by QueryId and should have different QueryId");
            trainData.QuerySize = CalcQueriesSize(trainData.QueryId);
        }

        if (ctx.Params.LossFunction == ELossFunction::Logloss) {
            PrepareTargetBinary(ctx.Params.Border, &trainData.Target);
            float minTarget = *MinElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
            float maxTarget = *MaxElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
            CB_ENSURE(minTarget == 0, "All targets are greater than border");
            CB_ENSURE(maxTarget == 1, "All targets are smaller than border");
        }

        if (trainHasBaseline) {
            CB_ENSURE((trainData.Baseline.ysize() > 1) == IsMultiClassError(ctx.Params.LossFunction), "Loss-function is MultiClass iff baseline dimension > 1");
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

        GenerateBorders(learnPool.Docs, &ctx, &ctx.LearnProgress.Model.Borders, &ctx.LearnProgress.Model.HasNans);

        if (testPool.Docs.GetDocCount() != 0) {
            CB_ENSURE(testPool.Docs.GetFactorsCount() == learnPool.Docs.GetFactorsCount(), "train pool factors count == " << learnPool.Docs.GetFactorsCount() << " and test pool factors count == " << testPool.Docs.GetFactorsCount());
            CB_ENSURE(testPool.Docs.GetBaselineDimension() == learnPool.Docs.GetBaselineDimension(), "train pool baseline dimension == " << learnPool.Docs.GetBaselineDimension() << " and test pool baseline dimension == " << testPool.Docs.GetBaselineDimension());

            if (!IsPairwiseError(ctx.Params.LossFunction)) {
                float minTestTarget = *MinElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
                float maxTestTarget = *MaxElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
                CB_ENSURE(minTestTarget != maxTestTarget, "All targets in test are equal.");
            }
        }
        learnPool.Docs.Append(testPool.Docs);
        const int factorsCount = learnPool.Docs.GetFactorsCount();
        const int approxDim = learnPool.Docs.GetBaselineDimension();

        auto learnPoolGuard = Finally([&] {
            if (!allowClearPool) {
                learnPool.Docs.Resize(trainData.LearnSampleCount, factorsCount, approxDim);
            }
        });

        PrepareAllFeatures(
            ctx.CatFeatures,
            ctx.LearnProgress.Model.Borders,
            ctx.LearnProgress.Model.HasNans,
            ctx.Params.IgnoredFeatures,
            trainData.LearnSampleCount,
            ctx.Params.OneHotMaxSize,
            ctx.Params.NanMode,
            allowClearPool,
            ctx.LocalExecutor,
            &learnPool.Docs,
            &trainData.AllFeatures);

        if (allowClearPool) {
            learnPool.Docs.Clear();
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
            case ELossFunction::Logloss:
                trainFunc = Train<TLoglossError>;
                break;
            case ELossFunction::CrossEntropy:
                trainFunc = Train<TCrossEntropyError>;
                break;
            case ELossFunction::RMSE:
                trainFunc = Train<TRMSEError>;
                break;
            case ELossFunction::MAE:
            case ELossFunction::Quantile:
                trainFunc = Train<TQuantileError>;
                break;
            case ELossFunction::LogLinQuantile:
                trainFunc = Train<TLogLinQuantileError>;
                break;
            case ELossFunction::MAPE:
                trainFunc = Train<TMAPError>;
                break;
            case ELossFunction::Poisson:
                trainFunc = Train<TPoissonError>;
                break;
            case ELossFunction::MultiClass:
                trainFunc = Train<TMultiClassError>;
                break;
            case ELossFunction::MultiClassOneVsAll:
                trainFunc = Train<TMultiClassOneVsAllError>;
                break;
            case ELossFunction::PairLogit:
                trainFunc = Train<TPairLogitError>;
                break;
            case ELossFunction::QueryRMSE:
                trainFunc = Train<TQueryRmseError>;
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
            yvector<TModelCtrBase> ctrsForModelCalcTasks;
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
            MATRIXNET_DEBUG_LOG << "Started parallel calculation of " << ctrsForModelCalcTasks.size() << " unique ctrs" << Endl;
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
                    ctx.Params.CounterCalcMethod,
                    resTable);
            }, 0, ctrsForModelCalcTasks.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
            MATRIXNET_DEBUG_LOG << "CTR calculation finished" << Endl;
        } else {
            yvector<TModelCtrBase> usedCtrs;
            {
                yhash_set<TModelCtrBase> ctrsSet;
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
            MATRIXNET_DEBUG_LOG << "Async calculation and writing of " << usedCtrs.size() << " unique ctrs started" << Endl;
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
                    ctx.Params.CounterCalcMethod,
                    &resTable);
                saver.SaveOneCtr(ctr, resTable);
                MATRIXNET_DEBUG_LOG << "Finished CTR number: " << i << Endl;
            }, 0, usedCtrs.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }
};
TTrainerFactory::TRegistrator<TCPUModelTrainer> CPURegistrator(ECalcerType::CPU);

void TrainModel(const NJson::TJsonValue& jsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& learnPool,
    bool allowClearPool,
    const TPool& testPool,
    const TString& outputModelPath,
    TFullModel* modelPtr,
    yvector<yvector<double>>* testApprox)
{
    THolder<IModelTrainer> modelTrainerHolder;
    bool isGpuCalcerType = jsonParams["calcer_type"].GetStringSafe("CPU") == "GPU";
    if (isGpuCalcerType && TTrainerFactory::Has(ECalcerType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ECalcerType::GPU);
    } else {
        CB_ENSURE(!isGpuCalcerType, "GPU Calcer not found.");
        modelTrainerHolder = TTrainerFactory::Construct(ECalcerType::CPU);
    }
    modelTrainerHolder->TrainModel(jsonParams, objectiveDescriptor, evalMetricDescriptor, learnPool, allowClearPool, testPool, outputModelPath, modelPtr, testApprox);
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
        case ELossFunction::Logloss:
            trainFunc = TrainOneIter<TLoglossError>;
            break;
        case ELossFunction::CrossEntropy:
            trainFunc = TrainOneIter<TCrossEntropyError>;
            break;
        case ELossFunction::RMSE:
            trainFunc = TrainOneIter<TRMSEError>;
            break;
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            trainFunc = TrainOneIter<TQuantileError>;
            break;
        case ELossFunction::LogLinQuantile:
            trainFunc = TrainOneIter<TLogLinQuantileError>;
            break;
        case ELossFunction::MAPE:
            trainFunc = TrainOneIter<TMAPError>;
            break;
        case ELossFunction::Poisson:
            trainFunc = TrainOneIter<TPoissonError>;
            break;
        case ELossFunction::MultiClass:
            trainFunc = TrainOneIter<TMultiClassError>;
            break;
        case ELossFunction::MultiClassOneVsAll:
            trainFunc = TrainOneIter<TMultiClassOneVsAllError>;
            break;
        case ELossFunction::PairLogit:
            trainFunc = TrainOneIter<TPairLogitError>;
            break;
        case ELossFunction::QueryRMSE:
            trainFunc = TrainOneIter<TQueryRmseError>;
            break;
        case ELossFunction::Custom:
            trainFunc = TrainOneIter<TCustomError>;
            break;
        default:
            CB_ENSURE(false, "provided error function is not supported");
    }

    trainFunc(trainData, ctx);
}
