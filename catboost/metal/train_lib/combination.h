#pragma once

#include "query.h"
#include <catboost/metal/native/metal_combination.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <util/string/cast.h>
#include <cmath>

namespace NCB {
    struct TMetalCombinationData {
        CBMCombinationOptions Options = {};
        TVector<CBMCombinationComponent> Components;
        TVector<ui32> GroupOffsets, Winners, Losers;
        TVector<float> PairWeights;
        ui32 YetiCount = 0;
    };

    inline TVector<CBMCombinationComponent> ParseMetalCombinationComponents(
        const NCatboostOptions::TLossDescription& description)
    {
        CB_ENSURE(description.GetLossFunction() == ELossFunction::Combination,
                  "Metal Combination preparation requires Combination loss");
        CheckCombinationParameters(description.GetLossParamsMap());
        TVector<CBMCombinationComponent> query, point;
        IterateOverCombination(description.GetLossParamsMap(), [&](const auto& loss, float weight) {
            CB_ENSURE(std::isfinite(weight) && weight > 0,
                      "Metal Combination weights must be finite and positive");
            CBMCombinationComponent component = {};
            component.weight = weight;
            component.beta = 1;
            const auto function = loss.GetLossFunction();
            switch (function) {
                case ELossFunction::RMSE: component.objective = 0; break;
                case ELossFunction::Logloss:
                    component.objective = 1;
                    component.border = NCatboostOptions::GetLogLossBorder(loss); break;
                case ELossFunction::CrossEntropy: component.objective = 2; break;
                case ELossFunction::Poisson: component.objective = 3; break;
                case ELossFunction::Huber:
                    component.objective = 4; component.param = NCatboostOptions::GetHuberParam(loss); break;
                case ELossFunction::Expectile:
                    component.objective = 5; component.param = NCatboostOptions::GetAlpha(loss); break;
                case ELossFunction::Lq:
                    component.objective = 6; component.param = FromString<float>(loss.GetLossParamsMap().at("q")); break;
                case ELossFunction::Tweedie:
                    component.objective = 7; component.param = NCatboostOptions::GetTweedieParam(loss); break;
                case ELossFunction::LogLinQuantile:
                    component.objective = 8; component.param = NCatboostOptions::GetAlpha(loss); break;
                case ELossFunction::Quantile:
                    component.objective = 9; component.param = NCatboostOptions::GetAlpha(loss); break;
                case ELossFunction::MAE: component.objective = 10; component.param = .5f; break;
                case ELossFunction::MAPE: component.objective = 11; break;
                case ELossFunction::QueryRMSE: component.objective = 12; break;
                case ELossFunction::QuerySoftMax:
                    component.objective = 13;
                    component.beta = NCatboostOptions::GetQuerySoftMaxBeta(loss);
                    component.lambda = NCatboostOptions::GetQuerySoftMaxLambdaReg(loss); break;
                case ELossFunction::PairLogit: component.objective = 14; break;
                case ELossFunction::YetiRank:
                    CB_ENSURE(!loss.GetLossParamsMap().contains("mode") || loss.GetLossParamsMap().at("mode") == "Classic",
                              "Metal Combination supports classic CUDA YetiRank components");
                    component.objective = 17;
                    component.permutations = NCatboostOptions::GetYetiRankPermutations(loss);
                    component.decay = NCatboostOptions::GetYetiRankDecay(loss); break;
                default:
                    ythrow TCatBoostException() << "Unsupported CUDA/Metal Combination component: " << function;
            }
            CB_ENSURE(std::isfinite(component.param) && std::isfinite(component.border) &&
                      std::isfinite(component.beta) && std::isfinite(component.lambda) && std::isfinite(component.decay),
                      "Metal Combination parameters must be finite");
            (component.objective >= 12 ? query : point).push_back(component);
        });
        // CUDA creates separate query and pointwise target vectors, preserving
        // declaration order within each vector, then accumulates queries first.
        query.insert(query.end(), point.begin(), point.end());
        CB_ENSURE(!query.empty() && query.size() <= 128,
                  "Metal Combination requires 1 to 128 active components");
        return query;
    }

    inline TMetalCombinationData PrepareMetalCombinationData(
        const NCatboostOptions::TLossDescription& loss,
        const TTargetDataProvider& target, const TObjectsGrouping& grouping)
    {
        TMetalCombinationData result;
        result.Components = ParseMetalCombinationComponents(loss);
        bool needGroups = false, needPairs = false;
        for (const auto& component : result.Components) {
            needGroups |= component.objective >= 12;
            needPairs |= component.objective == 14;
            result.YetiCount += component.objective == 17;
        }
        if (needGroups) {
            result.GroupOffsets.push_back(0);
            if (grouping.IsTrivial()) {
                for (ui32 row = 0; row < grouping.GetObjectCount(); ++row) result.GroupOffsets.push_back(row + 1);
            } else {
                for (const auto& group : grouping.GetNonTrivialGroups()) {
                    CB_ENSURE(group.Begin == result.GroupOffsets.back() && group.End > group.Begin,
                              "Metal Combination groups must be contiguous and nonempty");
                    result.GroupOffsets.push_back(group.End);
                }
            }
            result.Options.group_count = result.GroupOffsets.size() - 1;
        }
        if (needPairs) {
            auto pairs = PrepareMetalPairData(target, grouping.GetObjectCount());
            result.Winners = std::move(pairs.Winners);
            result.Losers = std::move(pairs.Losers);
            result.PairWeights = std::move(pairs.Weights);
            result.Options.pair_count = pairs.Options.pair_count;
        }
        result.Options.component_count = result.Components.size();
        return result;
    }
}
