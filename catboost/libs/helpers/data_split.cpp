#include "data_split.h"
#include "query_info_helper.h"

#include <catboost/libs/logging/logging.h>

TVector<std::pair<size_t, size_t>> Split(size_t docCount, int partCount) {
    TVector<std::pair<size_t, size_t>> result(partCount);
    for (int part = 0; part < partCount; ++part) {
        int partStartIndex, partEndIndex;
        InitElementRange(part, partCount, docCount, &partStartIndex, &partEndIndex);
        CB_ENSURE(partEndIndex - partStartIndex > 0, "Not enough documents for splitting into requested amount of parts");
        result[part] = {partStartIndex, partEndIndex};
    }
    return result;
}

TVector<std::pair<size_t, size_t>> Split(size_t docCount, const TVector<TGroupId>& queryId, int partCount) {
    TVector<TQueryInfo> queryInfo;
    UpdateQueriesInfo(queryId, /*subgroupId=*/{}, /*begin=*/0, docCount, &queryInfo);
    TVector<int> queryIndices = GetQueryIndicesForDocs(queryInfo, docCount);

    TVector<std::pair<size_t, size_t>> result(partCount);
    const size_t partSize = docCount / partCount;
    size_t currentPartEnd = 0;
    for (int part = 0; part < partCount; ++part) {
        size_t partStartIndex = currentPartEnd;
        size_t partEndIndex = Min(partStartIndex + partSize, docCount);
        partEndIndex = queryInfo[queryIndices[partEndIndex - 1]].End;

        currentPartEnd = partEndIndex;
        if (part + 1 == partCount) {
            partEndIndex = docCount;
        }
        CB_ENSURE(partEndIndex - partStartIndex > 0, "Not enough documents for splitting into requested amount of parts");
        result[part] = {partStartIndex, partEndIndex};
    }
    return result;
}

TVector<TVector<size_t>> StratifiedSplit(const TVector<float>& target, int partCount) {
    TVector<std::pair<float, int>> targetWithDoc(target.ysize());
    for (int i = 0; i < targetWithDoc.ysize(); ++i) {
        targetWithDoc[i].first = target[i];
        targetWithDoc[i].second = i;
    }
    Sort(targetWithDoc.begin(), targetWithDoc.end());

    TVector<TVector<int>> splittedByTarget;
    for (int i = 0; i < targetWithDoc.ysize(); ++i) {
        if (i == 0 || targetWithDoc[i].first != targetWithDoc[i - 1].first) {
            splittedByTarget.emplace_back();
        }
        splittedByTarget.back().push_back(targetWithDoc[i].second);
    }

    int minLen = target.ysize();
    for (const auto& part : splittedByTarget) {
        if (part.ysize() < minLen) {
            minLen = part.ysize();
        }
    }
    if (minLen < partCount) {
        MATRIXNET_WARNING_LOG << " Warning: The least populated class in y has only " << minLen << " members, which is too few. The minimum number of members in any class cannot be less than parts count=" << partCount << Endl;
    }
    TVector<TVector<size_t>> result(partCount);
    for (const auto& part : splittedByTarget) {
        for (int fold = 0; fold < partCount; ++fold) {
            int foldStartIndex, foldEndIndex;
            InitElementRange(fold, partCount, part.ysize(), &foldStartIndex, &foldEndIndex);
            for (int idx = foldStartIndex; idx < foldEndIndex; ++idx) {
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
    int testDocsBegin,
    int testDocsEnd,
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
    int testDocsBegin,
    int testDocsEnd,
    TVector<TPair>* learnPairs,
    TVector<TPair>* testPairs
) {
    int testDocs = testDocsEnd - testDocsBegin;
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
