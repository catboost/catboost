#include "docs_importance_helpers.h"
#include "ders_helpers.h"

#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/algo/index_calcer.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>

#include <numeric>


using namespace NCB;


TVector<TVector<double>> TDocumentImportancesEvaluator::GetDocumentImportances(
    const TProcessedDataProvider& processedData, int logPeriod
) {
    TVector<TVector<ui32>> leafIndices(TreeCount);
    auto binarizedFeatures = MakeQuantizedFeaturesForEvaluator(Model, *processedData.ObjectsData.Get());
    LocalExecutor->ExecRange([&] (int treeId) {
        leafIndices[treeId] = BuildIndicesForBinTree(Model, binarizedFeatures.Get(), treeId);
    }, NPar::ILocalExecutor::TExecRangeParams(0, TreeCount), NPar::TLocalExecutor::WAIT_COMPLETE);

    UpdateFinalFirstDerivatives(leafIndices, *processedData.TargetData->GetOneDimensionalTarget());
    TVector<TVector<double>> documentImportances(DocCount, TVector<double>(processedData.GetObjectCount()));
    const size_t docBlockSize = 1000;
    TImportanceLogger documentsLogger(DocCount, "documents processed", "Processing documents...", logPeriod);
    TProfileInfo processDocumentsProfile(DocCount);

    for (size_t start = 0; start < DocCount; start += docBlockSize) {
        const size_t end = Min<size_t>(start + docBlockSize, DocCount);
        processDocumentsProfile.StartIterationBlock();

        LocalExecutor->ExecRange([&] (int docId) {
            // The derivative of leaf values with respect to train doc weight.
            TVector<TVector<TVector<double>>> leafDerivatives(TreeCount, TVector<TVector<double>>(LeavesEstimationIterations)); // [treeCount][LeavesEstimationIterationsCount][leafCount]
            UpdateLeavesDerivatives(docId, &leafDerivatives);
            GetDocumentImportancesForOneTrainDoc(leafDerivatives, leafIndices, &documentImportances[docId]);
        }, NPar::ILocalExecutor::TExecRangeParams(start, end), NPar::TLocalExecutor::WAIT_COMPLETE);

        processDocumentsProfile.FinishIterationBlock(end - start);
        auto profileResults = processDocumentsProfile.GetProfileResults();
        documentsLogger.Log(profileResults);
        }
    return documentImportances;
}

void TDocumentImportancesEvaluator::UpdateFinalFirstDerivatives(const TVector<TVector<ui32>>& leafIndices, TConstArrayRef<float> target) {
    const ui32 docCount = SafeIntegerCast<ui32>(target.size());
    TVector<double> finalApproxes(docCount);

    for (ui32 treeId = 0; treeId < TreeCount; ++treeId) {
        const TVector<ui32>& leafIndicesRef = leafIndices[treeId];
        for (ui32 it = 0; it < LeavesEstimationIterations; ++it) {
            const TVector<double>& leafValues = TreesStatistics[treeId].LeafValues[it];
            for (ui32 docId = 0; docId < docCount; ++docId) {
                finalApproxes[docId] += leafValues[leafIndicesRef[docId]];
            }
        }
    }

    FinalFirstDerivatives.resize(docCount);
    EvaluateDerivatives(LossFunction, LeafEstimationMethod, finalApproxes, target, &FinalFirstDerivatives, nullptr, nullptr);
}

TVector<ui32> TDocumentImportancesEvaluator::GetLeafIdToUpdate(ui32 treeId, const TVector<double>& jacobian) {
    TVector<ui32> leafIdToUpdate;
    const ui32 leafCount = 1 << Model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeId];

    if (UpdateMethod.UpdateType == EUpdateType::AllPoints) {
        leafIdToUpdate.resize(leafCount);
        std::iota(leafIdToUpdate.begin(), leafIdToUpdate.end(), 0);
    } else if (UpdateMethod.UpdateType == EUpdateType::TopKLeaves) {
        const TVector<ui32>& leafIndices = TreesStatistics[treeId].LeafIndices;
        TVector<double> leafJacobians(leafCount);
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafJacobians[leafIndices[docId]] += Abs(jacobian[docId]);
        }

        TVector<ui32> orderedLeafIndices(leafCount);
        std::iota(orderedLeafIndices.begin(), orderedLeafIndices.end(), 0);
        StableSort(orderedLeafIndices.begin(), orderedLeafIndices.end(), [&](ui32 firstDocId, ui32 secondDocId) {
            return leafJacobians[firstDocId] > leafJacobians[secondDocId];
        });

        leafIdToUpdate = TVector<ui32>(
            orderedLeafIndices.begin(),
            orderedLeafIndices.begin() + Min<ui32>(UpdateMethod.TopSize, leafCount)
        );
    }

    return leafIdToUpdate;
}

void TDocumentImportancesEvaluator::UpdateLeavesDerivatives(ui32 removedDocId, TVector<TVector<TVector<double>>>* leafDerivatives) {
    TVector<double> jacobian(DocCount);
    for (ui32 treeId = 0; treeId < TreeCount; ++treeId) {
        auto& treeStatistics = TreesStatistics[treeId];
        for (ui32 it = 0; it < LeavesEstimationIterations; ++it) {
            const TVector<ui32> leafIdToUpdate = GetLeafIdToUpdate(treeId, jacobian);
            TVector<double>& leafDerivativesRef = (*leafDerivatives)[treeId][it];

            // Updating Leaves Derivatives
            UpdateLeavesDerivativesForTree(
                leafIdToUpdate,
                removedDocId,
                jacobian,
                treeId,
                it,
                &leafDerivativesRef
            );

            // Updating Jacobian
            bool isRemovedDocUpdated = false;
            for (ui32 leafId : leafIdToUpdate) {
                for (ui32 docId : treeStatistics.LeavesDocId[leafId]) {
                    jacobian[docId] += leafDerivativesRef[leafId];
                }
                isRemovedDocUpdated |= (treeStatistics.LeafIndices[removedDocId] == leafId);
            }
            if (!isRemovedDocUpdated) {
                ui32 removedDocLeafId = treeStatistics.LeafIndices[removedDocId];
                jacobian[removedDocId] += leafDerivativesRef[removedDocLeafId];
            }
        }
    }
}

void TDocumentImportancesEvaluator::GetDocumentImportancesForOneTrainDoc(
    const TVector<TVector<TVector<double>>>& leafDerivatives,
    const TVector<TVector<ui32>>& leafIndices,
    TVector<double>* documentImportance
) {
    const ui32 docCount = documentImportance->size();
    TVector<double> predictedDerivatives(docCount);

    for (ui32 treeId = 0; treeId < TreeCount; ++treeId) {
        const TVector<ui32>& leafIndicesRef = leafIndices[treeId];
        for (ui32 it = 0; it < LeavesEstimationIterations; ++it) {
            const TVector<double>& leafDerivativesRef = leafDerivatives[treeId][it];
            for (ui32 docId = 0; docId < docCount; ++docId) {
                predictedDerivatives[docId] += leafDerivativesRef[leafIndicesRef[docId]];
            }
        }
    }

    for (ui32 docId = 0; docId < docCount; ++docId) {
        (*documentImportance)[docId] = FinalFirstDerivatives[docId] * predictedDerivatives[docId];
    }
}

void TDocumentImportancesEvaluator::UpdateLeavesDerivativesForTree(
    const TVector<ui32>& leafIdToUpdate,
    ui32 removedDocId,
    const TVector<double>& jacobian,
    ui32 treeId,
    ui32 leavesEstimationIteration,
    TVector<double>* leafDerivatives
) {
    auto& leafDerivativesRef = *leafDerivatives;
    const auto& treeStatistics = TreesStatistics[treeId];
    const TVector<double>& formulaNumeratorMultiplier = treeStatistics.FormulaNumeratorMultiplier[leavesEstimationIteration];
    const TVector<double>& formulaNumeratorAdding = treeStatistics.FormulaNumeratorAdding[leavesEstimationIteration];
    const TVector<double>& formulaDenominators = treeStatistics.FormulaDenominators[leavesEstimationIteration];
    const ui32 removedDocLeafId = treeStatistics.LeafIndices[removedDocId];

    leafDerivativesRef.resize(treeStatistics.LeafCount);
    Fill(leafDerivativesRef.begin(), leafDerivativesRef.end(), 0);
    bool isRemovedDocUpdated = false;
    for (ui32 leafId : leafIdToUpdate) {
        for (ui32 docId : treeStatistics.LeavesDocId[leafId]) {
            leafDerivativesRef[leafId] += formulaNumeratorMultiplier[docId] * jacobian[docId];
        }
        if (leafId == removedDocLeafId) {
            leafDerivativesRef[leafId] += formulaNumeratorAdding[removedDocId];
        }
        leafDerivativesRef[leafId] *= -LearningRate / formulaDenominators[leafId];
        isRemovedDocUpdated |= (leafId == removedDocLeafId);
    }
    if (!isRemovedDocUpdated) {
        leafDerivativesRef[removedDocLeafId] += jacobian[removedDocId] * formulaNumeratorMultiplier[removedDocId];
        leafDerivativesRef[removedDocLeafId] += formulaNumeratorAdding[removedDocId];
        leafDerivativesRef[removedDocLeafId] *= -LearningRate / formulaDenominators[removedDocLeafId];
    }
}
