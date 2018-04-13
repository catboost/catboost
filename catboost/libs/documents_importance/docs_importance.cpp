#include "docs_importance.h"
#include "enums.h"

#include <util/generic/ymath.h>

static TUpdateMethod ParseUpdateMethod(const TString& updateMethod) {
    TString errorMessage = "Incorrect update-method param value. Should be one of: SinglePoint, \
        TopKLeaves, AllPoints or TopKLeaves:top=2 to set the top size in TopKLeaves method.";
    TVector<TString> tokens = StringSplitter(updateMethod).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(tokens.size() <= 2, errorMessage);
    EUpdateType updateType;
    CB_ENSURE(TryFromString<EUpdateType>(tokens[0], updateType), tokens[0] + " update method is not supported");
    CB_ENSURE(tokens.size() == 1 || (tokens.size() == 2 && updateType == EUpdateType::TopKLeaves), errorMessage);
    int topSize = 0;
    if (tokens.size() == 2) {
        TVector<TString> keyValue = StringSplitter(tokens[1]).SplitLimited('=', 2).ToList<TString>();
        CB_ENSURE(keyValue[0] == "top", errorMessage);
        CB_ENSURE(TryFromString<int>(keyValue[1], topSize), "Top size should be nonnegative integer, got: " + keyValue[1]);
    }
    return TUpdateMethod(updateType, topSize);
}

static TDStrResult GetFinalDocumentImportances(const TVector<TVector<double>>& importances, EDocumentStrengthType docImpMethod, int topSize) {
    const ui32 trainDocCount = importances.size();
    Y_ASSERT(importances.size() != 0);
    const ui32 testDocCount = importances[0].size();
    TVector<TVector<double>> preprocessedImportances;
    if (docImpMethod == EDocumentStrengthType::PerPool) {
        preprocessedImportances = TVector<TVector<double>>(1, TVector<double>(trainDocCount));
        for (ui32 trainDocId = 0; trainDocId < trainDocCount; ++trainDocId) {
            for (ui32 testDocId = 0; testDocId < testDocCount; ++testDocId) {
                preprocessedImportances[0][trainDocId] += importances[trainDocId][testDocId];
            }
        }
        for (ui32 trainDocId = 0; trainDocId < trainDocCount; ++trainDocId) {
            preprocessedImportances[0][trainDocId] /= testDocCount;
        }

    } else {
        Y_ASSERT(docImpMethod == EDocumentStrengthType::PerObject || docImpMethod == EDocumentStrengthType::Raw);
        preprocessedImportances = TVector<TVector<double>>(testDocCount, TVector<double>(trainDocCount));
        for (ui32 trainDocId = 0; trainDocId < trainDocCount; ++trainDocId) {
            for (ui32 testDocId = 0; testDocId < testDocCount; ++testDocId) {
                preprocessedImportances[testDocId][trainDocId] = importances[trainDocId][testDocId];
            }
        }
    }

    const ui32 docCount = preprocessedImportances.size();
    TDStrResult result(preprocessedImportances.size(), topSize);
    for (ui32 testDocId = 0; testDocId < docCount; ++testDocId) {
        TVector<ui32> indices(trainDocCount);
        std::iota(indices.begin(), indices.end(), 0);
        if (docImpMethod != EDocumentStrengthType::Raw) {
            Sort(indices.begin(), indices.end(), [&](ui32 first, ui32 second) {
                return Abs(preprocessedImportances[testDocId][first]) > Abs(preprocessedImportances[testDocId][second]);
            });
        }
        for (int i = 0; i < topSize; ++i) {
            result.Scores[testDocId][i] = preprocessedImportances[testDocId][indices[i]];
        }
        result.Indices[testDocId].swap(indices);
        result.Indices[testDocId].resize(topSize);
    }
    return result;
}

TDStrResult GetDocumentImportances(
    const TFullModel& model,
    const TPool& trainPool,
    const TPool& testPool,
    const TString& dstrTypeStr,
    int topSize,
    const TString& updateMethodStr,
    int threadCount
) {
    if (topSize == -1) {
        topSize = trainPool.Docs.GetDocCount();
    } else {
        CB_ENSURE(topSize >= 0, "Top size should be nonnegative integer or -1 (for unlimited top size).");
        topSize = Min<int>(topSize, trainPool.Docs.GetDocCount());
    }

    TUpdateMethod updateMethod = ParseUpdateMethod(updateMethodStr);
    EDocumentStrengthType dstrType = FromString<EDocumentStrengthType>(dstrTypeStr);
    TDocumentImportancesEvaluator leafInfluenceEvaluator(model, trainPool, updateMethod, threadCount);
    const TVector<TVector<double>> documentImportances = leafInfluenceEvaluator.GetDocumentImportances(testPool);
    return GetFinalDocumentImportances(documentImportances, dstrType, topSize);
}

