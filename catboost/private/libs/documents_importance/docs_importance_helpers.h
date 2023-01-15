#pragma once

#include "enums.h"
#include "tree_statistics.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/json_helper.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/system/types.h>
#include <util/system/yassert.h>


/*
 * This is the implementation of the LeafInfluence algorithm from the following paper:
 * https://arxiv.org/pdf/1802.06640.pdf
 */

// This class describes the update set method (section 3.1.3 from the paper).
struct TUpdateMethod {
    TUpdateMethod() = default;
    explicit TUpdateMethod(EUpdateType updateType, int topSize = -1)
        : UpdateType(updateType)
        , TopSize(topSize)
    {
        CB_ENSURE(UpdateType != EUpdateType::TopKLeaves || TopSize >= 0,
            "You should provide top size for TopKLeaves method. It should be nonnegative integer.");
    }

    EUpdateType UpdateType;
    int TopSize;
};

// The class for document importances evaluation.
class TDocumentImportancesEvaluator {
public:
    TDocumentImportancesEvaluator(
        const TFullModel& model,
        const NCB::TProcessedDataProvider& processedData,
        const TUpdateMethod& updateMethod,
        TAtomicSharedPtr<NPar::TLocalExecutor> localExecutor,
        int logPeriod
    )
        : Model(model)
        , UpdateMethod(updateMethod)
        , TreeCount(model.GetTreeCount())
        , DocCount(processedData.GetObjectCount())
        , LocalExecutor(std::move(localExecutor))
    {
        NJson::TJsonValue paramsJson = ReadTJsonValue(model.ModelInfo.at("params"));
        LossFunction = FromString<ELossFunction>(paramsJson["loss_function"]["type"].GetString());
        LeafEstimationMethod = FromString<ELeavesEstimation>(paramsJson["tree_learner_options"]["leaf_estimation_method"].GetString());
        LeavesEstimationIterations = paramsJson["tree_learner_options"]["leaf_estimation_iterations"].GetUInteger();
        LearningRate = paramsJson["boosting_options"]["learning_rate"].GetDouble();
        TMaybe<double> startingApprox = Nothing();
        if (paramsJson["boost_from_average"].GetBoolean()) {
             startingApprox = NCB::CalcOneDimensionalOptimumConstApprox(
                NCatboostOptions::ParseLossDescription(ToString(LossFunction)),
                processedData.TargetData->GetOneDimensionalTarget().GetOrElse(TConstArrayRef<float>()),
                GetWeights(*processedData.TargetData)
            );
        }

        THolder<ITreeStatisticsEvaluator> treeStatisticsEvaluator;
        const ELeavesEstimation leavesEstimationMethod = FromString<ELeavesEstimation>(paramsJson["tree_learner_options"]["leaf_estimation_method"].GetString());
        if (leavesEstimationMethod == ELeavesEstimation::Gradient) {
            treeStatisticsEvaluator = MakeHolder<TGradientTreeStatisticsEvaluator>(DocCount);
        } else {
        Y_ASSERT(leavesEstimationMethod == ELeavesEstimation::Newton);
            treeStatisticsEvaluator = MakeHolder<TNewtonTreeStatisticsEvaluator>(DocCount);
        }
        TreesStatistics = treeStatisticsEvaluator->EvaluateTreeStatistics(model, processedData, startingApprox, logPeriod);
    }

    // Getting the importance of all train objects for all objects from pool.
    TVector<TVector<double>> GetDocumentImportances(const NCB::TProcessedDataProvider& processedData, int logPeriod = 0);

private:
    // Evaluate first derivatives at the final approxes
    void UpdateFinalFirstDerivatives(const TVector<TVector<ui32>>& leafIndices, TConstArrayRef<float> target);
    // Leaves derivatives will be updated based on objects from these leaves.
    TVector<ui32> GetLeafIdToUpdate(ui32 treeId, const TVector<double>& jacobian);
    // Algorithm 4 from paper.
    void UpdateLeavesDerivatives(ui32 removedDocId, TVector<TVector<TVector<double>>>* leafDerivatives);
    // Getting the importance of one train object for all objects from pool.
    void GetDocumentImportancesForOneTrainDoc(
        const TVector<TVector<TVector<double>>>& leafDerivatives,
        const TVector<TVector<ui32>>& leafIndices,
        TVector<double>* documentImportance
    );
    // Evaluate leaf derivatives at a given removedDocId weight (Equation (6) from paper).
    void UpdateLeavesDerivativesForTree(
        const TVector<ui32>& leafIdToUpdate,
        ui32 removedDocId,
        const TVector<double>& jacobian,
        ui32 treeId,
        ui32 leavesEstimationIteration,
        TVector<double>* leafDerivatives
    );

private:
    TFullModel Model;
    TVector<TTreeStatistics> TreesStatistics; // [treeCount]
    TVector<double> FinalFirstDerivatives; // [docCount]
    TUpdateMethod UpdateMethod;
    ELossFunction LossFunction;
    ELeavesEstimation LeafEstimationMethod;
    ui32 LeavesEstimationIterations;
    float LearningRate;
    ui32 TreeCount;
    ui32 DocCount;
    TAtomicSharedPtr<NPar::TLocalExecutor> LocalExecutor;
};
