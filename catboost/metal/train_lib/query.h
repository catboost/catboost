#pragma once

#include <catboost/libs/data/objects_grouping.h>
#include <catboost/libs/data/target.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/metal/native/metal_trainer.h>
#include <catboost/private/libs/options/loss_description.h>

#include <util/generic/vector.h>

namespace NCB {
    struct TMetalQueryData {
        CBMQueryOptions Options = {};
        TVector<ui32> Offsets;
    };

    // Shared GPU target preparation has already multiplied object and group
    // weights. These boundaries describe the same prepared object order;
    // query training must not multiply TQueryInfo::Weight a second time.
    inline TMetalQueryData PrepareMetalQueryData(
        const TObjectsGrouping& grouping,
        const NCatboostOptions::TLossDescription& loss)
    {
        const auto objective = loss.GetLossFunction();
        CB_ENSURE(objective == ELossFunction::QueryRMSE || objective == ELossFunction::QuerySoftMax ||
                  objective == ELossFunction::YetiRank || objective == ELossFunction::YetiRankPairwise || objective == ELossFunction::QueryCrossEntropy,
                  "Metal query preparation requires QueryRMSE, QuerySoftMax, YetiRank, or QueryCrossEntropy");
        TMetalQueryData result;
        result.Options.group_count = grouping.GetGroupCount();
        result.Options.beta = objective == ELossFunction::QuerySoftMax ?
            NCatboostOptions::GetQuerySoftMaxBeta(loss) : 1.0f;
        result.Options.lambda = objective == ELossFunction::QuerySoftMax ?
            NCatboostOptions::GetQuerySoftMaxLambdaReg(loss) : 0.0f;
        result.Offsets.reserve(ui64(result.Options.group_count) + 1);
        result.Offsets.push_back(0);
        if (grouping.IsTrivial()) {
            for (ui32 row = 0; row < grouping.GetObjectCount(); ++row) {
                result.Offsets.push_back(row + 1);
            }
        } else {
            for (const auto& group : grouping.GetNonTrivialGroups()) {
                CB_ENSURE(group.Begin == result.Offsets.back() && group.End > group.Begin,
                          "Metal query groups must form contiguous nonempty object ranges");
                result.Offsets.push_back(group.End);
            }
        }
        CB_ENSURE(result.Options.group_count &&
                  result.Offsets.size() == ui64(result.Options.group_count) + 1 &&
                  result.Offsets.back() == grouping.GetObjectCount(),
                  "Metal query boundaries must cover every training object");
        return result;
    }

    struct TMetalPairData {
        CBMPairOptions Options = {};
        TVector<ui32> Winners;
        TVector<ui32> Losers;
        TVector<float> Weights;
        TVector<ui32> GroupOffsets;
    };

    inline TMetalPairData PrepareMetalPairData(const TTargetDataProvider& target, ui32 rows) {
        const auto groups = target.GetGroupInfo();
        CB_ENSURE(groups && !groups->empty(), "Metal PairLogit requires prepared pairs and group information");
        TMetalPairData result;
        result.GroupOffsets.push_back(0);
        for (const auto& group : *groups) {
            CB_ENSURE(group.Begin == result.GroupOffsets.back() && group.End > group.Begin && group.End <= rows,
                      "Metal PairLogit groups must cover contiguous nonempty object ranges");
            CB_ENSURE(group.Competitors.empty() || group.Competitors.size() == group.GetSize(),
                      "Metal PairLogit competitor lists must match each query size");
            result.GroupOffsets.push_back(group.End);
            for (ui32 winner = 0; winner < group.Competitors.size(); ++winner) {
                for (const auto& competitor : group.Competitors[winner]) {
                    CB_ENSURE(competitor.Id < group.GetSize() && competitor.Id != winner,
                              "Metal PairLogit requires distinct winner/loser objects within a query");
                    CB_ENSURE(result.Winners.size() < Max<ui32>(), "Metal PairLogit pair count exceeds uint32 capacity");
                    result.Winners.push_back(group.Begin + winner);
                    result.Losers.push_back(group.Begin + competitor.Id);
                    // Prepared supplied/generated edges already carry their
                    // correct weight. Object and group weights are not added.
                    result.Weights.push_back(competitor.Weight);
                }
            }
        }
        CB_ENSURE(!result.Winners.empty() && result.GroupOffsets.back() == rows,
                  "Metal PairLogit requires at least one pair and complete query coverage");
        result.Options.pair_count = result.Winners.size();
        result.Options.group_count = result.GroupOffsets.size() - 1;
        return result;
    }
}
