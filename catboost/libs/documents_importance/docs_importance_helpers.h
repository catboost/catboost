#pragma once

#include "enums.h"
#include "tree_statistics.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/options/catboost_options.h>

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
        const TPool& pool,
        const TUpdateMethod& updateMethod,
        int threadCount
    )
        : Model(model)
        , UpdateMethod(updateMethod)
        , TreeCount(model.ObliviousTrees.GetTreeCount())
        , DocCount(pool.Docs.GetDocCount())
        , ThreadCount(threadCount)
    {
        NJson::TJsonValue paramsJson = ReadTJsonValue(model.ModelInfo.at("params"));
        TCatboostOptions params = NCatboostOptions::LoadOptions(paramsJson);
        LossFunction = params.LossFunctionDescription->GetLossFunction();
        LeavesEstimationIterations = params.ObliviousTreeOptions->LeavesEstimationIterations.Get();
        LearningRate = params.BoostingOptions->LearningRate;

        THolder<ITreeStatisticsEvaluator> treeStatisticsEvaluator;
        const ELeavesEstimation leavesEstimationMethod = params.ObliviousTreeOptions->LeavesEstimationMethod.Get();
        if (leavesEstimationMethod == ELeavesEstimation::Gradient) {
            treeStatisticsEvaluator = MakeHolder<TGradientTreeStatisticsEvaluator>(DocCount);
        } else {
        Y_ASSERT(leavesEstimationMethod == ELeavesEstimation::Newton);
            treeStatisticsEvaluator = MakeHolder<TNewtonTreeStatisticsEvaluator>(DocCount);
        }
        TreesStatistics = treeStatisticsEvaluator->EvaluateTreeStatistics(model, pool);
    }

    // Getting the importance of all train objects for all objects from pool.
    TVector<TVector<double>> GetDocumentImportances(const TPool& pool);

private:
    // Evaluate first derivatives at the final approxes
    void UpdateFinalFirstDerivatives(const TVector<TVector<ui32>>& leafIndices, const TPool& pool);
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
    ui32 LeavesEstimationIterations;
    float LearningRate;
    ui32 TreeCount;
    ui32 DocCount;
    int ThreadCount;
};
