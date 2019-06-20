#include "data_split.h"
#include "query_info_helper.h"

#include <catboost/libs/logging/logging.h>

#include <util/generic/array_ref.h>

TVector<std::pair<ui32, ui32>> Split(ui32 docCount, ui32 partCount) {
    TVector<std::pair<ui32, ui32>> result(partCount);
    for (ui32 part = 0; part < partCount; ++part) {
        ui32 partStartIndex, partEndIndex;
        InitElementRange(part, partCount, docCount, &partStartIndex, &partEndIndex);
        CB_ENSURE(
            partEndIndex - partStartIndex > 0,
            "Not enough documents for splitting into requested amount of parts"
        );
        result[part] = {partStartIndex, partEndIndex};
    }
    return result;
}

TVector<std::pair<ui32, ui32>> Split(ui32 docCount, const TVector<TGroupId>& queryId, ui32 partCount) {
    TVector<TQueryInfo> queryInfo;
    UpdateQueriesInfo(queryId, /*groupWeight=*/{}, /*subgroupId=*/{}, /*begin=*/0, docCount, &queryInfo);
    TVector<ui32> queryIndices = GetQueryIndicesForDocs(queryInfo, docCount);

    TVector<std::pair<ui32, ui32>> result(partCount);
    const ui32 partSize = docCount / partCount;
    ui32 currentPartEnd = 0;
    for (ui32 part = 0; part < partCount; ++part) {
        ui32 partStartIndex = currentPartEnd;
        ui32 partEndIndex = Min(partStartIndex + partSize, docCount);
        partEndIndex = queryInfo[queryIndices[partEndIndex - 1]].End;

        currentPartEnd = partEndIndex;
        if (part + 1 == partCount) {
            partEndIndex = docCount;
        }
        CB_ENSURE(
            partEndIndex - partStartIndex > 0,
            "Not enough documents for splitting into requested amount of parts"
        );
        result[part] = {partStartIndex, partEndIndex};
    }
    return result;
}

TVector<TVector<ui32>> StratifiedSplit(const TVector<float>& target, ui32 partCount) {
    TVector<std::pair<float, ui32>> targetWithDoc(target.size());
    for (ui32 i = 0; i < targetWithDoc.size(); ++i) {
        targetWithDoc[i].first = target[i];
        targetWithDoc[i].second = i;
    }
    Sort(targetWithDoc.begin(), targetWithDoc.end());

    TVector<TVector<ui32>> splittedByTarget;
    for (ui32 i = 0; i < targetWithDoc.size(); ++i) {
        if (i == 0 || targetWithDoc[i].first != targetWithDoc[i - 1].first) {
            splittedByTarget.emplace_back();
        }
        splittedByTarget.back().push_back(targetWithDoc[i].second);
    }

    ui32 minLen = target.size();
    for (const auto& part : splittedByTarget) {
        if (part.size() < minLen) {
            minLen = part.size();
        }
    }
    if (minLen < partCount) {
        CATBOOST_WARNING_LOG << " Warning: The least populated class in y has only " << minLen << " members,"
            " which is too few. The minimum number of members in any class cannot be less than parts count="
            << partCount << Endl;
    }
    TVector<TVector<ui32>> result(partCount);
    for (const auto& part : splittedByTarget) {
        for (ui32 fold = 0; fold < partCount; ++fold) {
            ui32 foldStartIndex, foldEndIndex;
            InitElementRange(fold, partCount, part.size(), &foldStartIndex, &foldEndIndex);
            for (ui32 idx = foldStartIndex; idx < foldEndIndex; ++idx) {
                result[fold].push_back(part[idx]);
            }
        }
    }

    for (auto& part : result) {
        CB_ENSURE(!part.empty(), "Not enough documents for splitting into " << partCount << " parts");
        Sort(part.begin(), part.end());
    }
    return result;
}

void SplitPairs(
    const TVector<TPair>& pairs,
    ui32 testDocsBegin,
    ui32 testDocsEnd,
    TVector<TPair>* learnPairs,
    TVector<TPair>* testPairs
) {
    for (const auto& pair : pairs) {
        bool isWinnerInTest = testDocsBegin <= pair.WinnerId && pair.WinnerId < testDocsEnd;
        bool isLoserInTest = testDocsBegin <= pair.LoserId && pair.LoserId < testDocsEnd;
        Y_VERIFY(isWinnerInTest == isLoserInTest);
        if (isWinnerInTest) {
            testPairs->emplace_back(pair.WinnerId, pair.LoserId, pair.Weight);
        } else {
            learnPairs->emplace_back(pair.WinnerId, pair.LoserId, pair.Weight);
        }
    }
}

void SplitPairsAndReindex(
    const TVector<TPair>& pairs,
    ui32 testDocsBegin,
    ui32 testDocsEnd,
    TVector<TPair>* learnPairs,
    TVector<TPair>* testPairs
) {
    ui32 testDocs = testDocsEnd - testDocsBegin;
    for (const auto& pair : pairs) {
        auto winnerId = pair.WinnerId;
        auto loserId = pair.LoserId;
        bool isWinnerInTest = testDocsBegin <= winnerId && winnerId < testDocsEnd;
        bool isLoserInTest = testDocsBegin <= loserId && loserId < testDocsEnd;
        Y_VERIFY(isWinnerInTest == isLoserInTest);
        if (isWinnerInTest) {
            winnerId -= testDocsBegin;
            loserId -= testDocsBegin;
            testPairs->emplace_back(winnerId, loserId, pair.Weight);
        } else {
            if (winnerId > testDocsBegin)
                winnerId -= testDocs;
            if (loserId > testDocsBegin)
                loserId -= testDocs;
            learnPairs->emplace_back(winnerId, loserId, pair.Weight);
        }
    }
}
