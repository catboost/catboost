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

static TDStrResult GetFinalDocumentImportances(
    const TVector<TVector<double>>& rawImportances,
    EDocumentStrengthType docImpMethod,
    int topSize,
    EImportanceValuesSign importanceValuesSign
) {
    const ui32 trainDocCount = rawImportances.size();
    Y_ASSERT(rawImportances.size() != 0);
    const ui32 testDocCount = rawImportances[0].size();
    TVector<TVector<double>> preprocessedImportances;
    if (docImpMethod == EDocumentStrengthType::PerPool) {
        preprocessedImportances = TVector<TVector<double>>(1, TVector<double>(trainDocCount));
        for (ui32 trainDocId = 0; trainDocId < trainDocCount; ++trainDocId) {
            for (ui32 testDocId = 0; testDocId < testDocCount; ++testDocId) {
                preprocessedImportances[0][trainDocId] += rawImportances[trainDocId][testDocId];
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
                preprocessedImportances[testDocId][trainDocId] = rawImportances[trainDocId][testDocId];
            }
        }
    }

    TDStrResult result(preprocessedImportances.size());
    for (ui32 testDocId = 0; testDocId < preprocessedImportances.size(); ++testDocId) {
        TVector<double>& preprocessedImportancesRef = preprocessedImportances[testDocId];

        const ui32 docCount = preprocessedImportancesRef.size();
        TVector<ui32> indices(docCount);
        std::iota(indices.begin(), indices.end(), 0);
        if (docImpMethod != EDocumentStrengthType::Raw) {
            Sort(indices.begin(), indices.end(), [&](ui32 first, ui32 second) {
                return Abs(preprocessedImportancesRef[first]) > Abs(preprocessedImportancesRef[second]);
            });
        }

        std::function<bool(double)> predicate;
        if (importanceValuesSign == EImportanceValuesSign::Positive) {
            predicate = [](double v){return v > 0;};
        } else if (importanceValuesSign == EImportanceValuesSign::Negative) {
            predicate = [](double v){return v < 0;};
        } else {
            Y_ASSERT(importanceValuesSign == EImportanceValuesSign::All);
            predicate = [](double){return true;};
        }

        int currentSize = 0;
        for (ui32 i = 0; i < docCount; ++i) {
            if (currentSize == topSize) {
                break;
            }
            if (predicate(preprocessedImportancesRef[indices[i]])) {
                result.Scores[testDocId].push_back(preprocessedImportancesRef[indices[i]]);
                result.Indices[testDocId].push_back(indices[i]);
            }
            ++currentSize;
        }
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
    const TString& importanceValuesSignStr,
    int threadCount
) {
    if (topSize == -1) {
        topSize = trainPool.Docs.GetDocCount();
    } else {
        CB_ENSURE(topSize >= 0, "Top size should be nonnegative integer or -1 (for unlimited top size).");
    }

    TUpdateMethod updateMethod = ParseUpdateMethod(updateMethodStr);
    EDocumentStrengthType dstrType = FromString<EDocumentStrengthType>(dstrTypeStr);
    EImportanceValuesSign importanceValuesSign = FromString<EImportanceValuesSign>(importanceValuesSignStr);
    TDocumentImportancesEvaluator leafInfluenceEvaluator(model, trainPool, updateMethod, threadCount);
    const TVector<TVector<double>> documentImportances = leafInfluenceEvaluator.GetDocumentImportances(testPool);
    return GetFinalDocumentImportances(documentImportances, dstrType, topSize, importanceValuesSign);
}

