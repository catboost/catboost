#include "learn_context.h"
#include "train_model.h"
#include "apply.h"
#include "train.h"
#include "helpers.h"
#include "model_build_helper.h"
#include "tree_print.h"

#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/libs/helpers/mem_usage.h>

#include <library/grid_creator/binarization.h>

#include <util/random/shuffle.h>
#include <util/generic/vector.h>

static void PrepareTargetBinary(float border, TVector<float>* target) {
    for (int i = 0; i < (*target).ysize(); ++i) {
        (*target)[i] = ((*target)[i] > border);
    }
}

static bool IsCorrectQueryIdsFormat(const TVector<ui32>& queryIds) {
    THashSet<ui32> queryGroupIds;
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

static yhash<ui32, ui32> CalcQueriesSize(const TVector<ui32>& queriesId) {
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
        TEvalResult* evalResult) const override
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

        SetLogingLevel(ctx.Params.LoggingLevel);

        auto loggingGuard = Finally([&] { SetSilentLogingMode(); });

        TVector<size_t> indices(learnPool.Docs.GetDocCount());
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
            ctx.LearnProgress.ApproxDimension = GetClassesCount(trainData.Target, ctx.Params.ClassesCount);
            CB_ENSURE(ctx.LearnProgress.ApproxDimension > 1, "All targets are equal");
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

        GenerateBorders(learnPool, &ctx, &ctx.LearnProgress.FloatFeatures);

        if (testPool.Docs.GetDocCount() != 0) {
            CB_ENSURE(testPool.Docs.GetFactorsCount() == learnPool.Docs.GetFactorsCount(), "train pool factors count == " << learnPool.Docs.GetFactorsCount() << " and test pool factors count == " << testPool.Docs.GetFactorsCount());
            CB_ENSURE(testPool.Docs.GetBaselineDimension() == learnPool.Docs.GetBaselineDimension(), "train pool baseline dimension == " << learnPool.Docs.GetBaselineDimension() << " and test pool baseline dimension == " << testPool.Docs.GetBaselineDimension());

            if (!IsPairwiseError(ctx.Params.LossFunction)) {
                float minTestTarget = *MinElement(trainData.Target.begin() + trainData.LearnSampleCount, trainData.Target.begin() + trainData.GetSampleCount());
                float maxTestTarget = *MaxElement(trainData.Target.begin() + trainData.LearnSampleCount, trainData.Target.begin() + trainData.GetSampleCount());
                if (minTestTarget == maxTestTarget) {
                    MATRIXNET_WARNING_LOG << "Test target is constant or not set." << Endl;
                }
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
            ctx.LearnProgress.FloatFeatures,
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
        ctx.LearnProgress.CatFeatures.resize(sortedCatFeatures.size());
        for (size_t i = 0; i < sortedCatFeatures.size(); ++i) {
            auto& catFeature = ctx.LearnProgress.CatFeatures[i];
            catFeature.FeatureIndex = i;
            catFeature.FlatFeatureIndex = sortedCatFeatures[i];
            if (catFeature.FlatFeatureIndex < learnPool.FeatureId.ysize()) {
                catFeature.FeatureId = learnPool.FeatureId[catFeature.FlatFeatureIndex];
            }
        }

        evalResult->GetRawValuesRef().resize(ctx.LearnProgress.ApproxDimension);
        DumpMemUsage("Before start train");

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
        trainFunc(trainData, &ctx, &(evalResult->GetRawValuesRef()));
        evalResult->SetPredictionTypes(ctx.Params.PredictionTypes);

        TObliviousTrees obliviousTrees;
        yhash<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
        {
            TObliviousTreeBuilder builder(ctx.LearnProgress.FloatFeatures, ctx.LearnProgress.CatFeatures);
            for (size_t treeId = 0; treeId < ctx.LearnProgress.TreeStruct.size(); ++treeId) {
                TVector<TModelSplit> modelSplits;
                for (const auto& split : ctx.LearnProgress.TreeStruct[treeId].Splits) {
                    auto modelSplit = split.GetModelSplit(ctx);
                    modelSplits.push_back(modelSplit);
                    if (modelSplit.Type == ESplitType::OnlineCtr) {
                        featureCombinationToProjectionMap[modelSplit.OnlineCtr.Ctr.Base.Projection] = split.Ctr.Projection;
                    }
                }
                builder.AddTree(modelSplits, ctx.LearnProgress.LeafValues[treeId]);
            }
            obliviousTrees = builder.Build();
        }

        for (auto& oheFeature : obliviousTrees.OneHotFeatures) {
            for (const auto& value : oheFeature.Values) {
                oheFeature.StringValues.push_back(learnPool.CatFeaturesHashToString.at(value));
            }
        }
        auto ctrTableGenerator = [&] (const TModelCtrBase& ctr) -> TCtrValueTable {
            TCtrValueTable resTable;
            CalcFinalCtrs(
                ctr.CtrType,
                featureCombinationToProjectionMap.at(ctr.Projection),
                trainData,
                ctx.LearnProgress.AveragingFold.LearnPermutation,
                ctx.LearnProgress.AveragingFold.LearnTargetClass[ctr.TargetBorderClassifierIdx],
                ctx.LearnProgress.AveragingFold.TargetClassesCount[ctr.TargetBorderClassifierIdx],
                ctx.Params.CtrLeafCountLimit,
                ctx.Params.StoreAllSimpleCtr,
                ctx.Params.CounterCalcMethod,
                &resTable);
            resTable.ModelCtrBase = ctr;
            MATRIXNET_DEBUG_LOG << "Finished CTR: " << ctr.CtrType << " " << BuildDescription(ctx.Layout, ctr.Projection) << Endl;
            return resTable;
        };
        if (modelPtr) {
            modelPtr->ObliviousTrees = std::move(obliviousTrees);
            modelPtr->ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
            TVector<TModelCtrBase> usedCtrBases = modelPtr->ObliviousTrees.GetUsedModelCtrBases();
            modelPtr->CtrProvider = new TStaticCtrProvider;
            TMutex lock;
            MATRIXNET_DEBUG_LOG << "Started parallel calculation of " << usedCtrBases.size() << " unique ctrs" << Endl;
            ctx.LocalExecutor.ExecRange([&](int i) {
                auto& ctr = usedCtrBases[i];
                auto table = ctrTableGenerator(ctr);
                with_lock(lock) {
                    modelPtr->CtrProvider->AddCtrCalcerData(std::move(table));
                }
            }, 0, usedCtrBases.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
            MATRIXNET_DEBUG_LOG << "CTR calculation finished" << Endl;
            modelPtr->UpdateDynamicData();
        } else {
            TFullModel Model;
            Model.ObliviousTrees = std::move(obliviousTrees);
            Model.ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
            TVector<TModelCtrBase> usedCtrBases = Model.ObliviousTrees.GetUsedModelCtrBases();

            Model.CtrProvider = new TStaticCtrOnFlightSerializationProvider(usedCtrBases, ctrTableGenerator, ctx.LocalExecutor);
            MATRIXNET_DEBUG_LOG << "Async calculation and writing of " << usedCtrBases.size() << " unique ctrs started" << Endl;
            OutputModel(Model, outputModelPath);
        }
    }
};
TTrainerFactory::TRegistrator<TCPUModelTrainer> CPURegistrator(ETaskType::CPU);

void TrainModel(const NJson::TJsonValue& jsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& learnPool,
    bool allowClearPool,
    const TPool& testPool,
    const TString& outputModelPath,
    TFullModel* modelPtr,
    TEvalResult* evalResult)
{
    THolder<IModelTrainer> modelTrainerHolder;
    bool isGpuTaskType = jsonParams["task_type"].GetStringSafe("CPU") == "GPU";
    if (isGpuTaskType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(!isGpuTaskType, "GPU Device not found.");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }
    modelTrainerHolder->TrainModel(jsonParams, objectiveDescriptor, evalMetricDescriptor, learnPool, allowClearPool, testPool, outputModelPath, modelPtr, evalResult);
}

void TrainOneIteration(const TTrainData& trainData, TLearnContext* ctx)
{
    SetLogingLevel(ctx->Params.LoggingLevel);

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
