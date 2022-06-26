#include "helpers.h"

const TMirrorBuffer<ui32>& NCatboostCuda::GetBinsForModel(TScopedCacheHolder& cacheHolder,
                                                          const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
                                                          const NCatboostCuda::TFeatureParallelDataSet& dataSet,
                                                          const NCatboostCuda::TObliviousTreeStructure& structure) {
    bool hasPermutationCtrs = HasPermutationDependentSplit(structure, featuresManager);
    const auto& scope = hasPermutationCtrs ? dataSet.GetPermutationDependentScope() : dataSet.GetPermutationIndependentScope();
    return cacheHolder.Cache(scope, structure, [&]() -> TMirrorBuffer<ui32> {
        const bool hasHistory = dataSet.HasCtrHistoryDataSet();
        TMirrorBuffer<ui32> learnBins;
        TMirrorBuffer<ui32> testBins;

        if (hasHistory) {
            learnBins = TMirrorBuffer<ui32>::Create(dataSet.LinkedHistoryForCtr().GetSamplesMapping());
            testBins = TMirrorBuffer<ui32>::Create(dataSet.GetSamplesMapping());
        } else {
            learnBins = TMirrorBuffer<ui32>::Create(dataSet.GetSamplesMapping());
        }

        {
            TTreeUpdater builder(cacheHolder,
                                 featuresManager,
                                 dataSet.GetCtrTargets(),
                                 hasHistory ? dataSet.LinkedHistoryForCtr() : dataSet,
                                 learnBins,
                                 hasHistory ? &dataSet : nullptr,
                                 hasHistory ? &testBins : nullptr);

            for (auto& split : structure.Splits) {
                builder.AddSplit(split);
            }
        }

        if (hasHistory) {
            cacheHolder.CacheOnly(dataSet.LinkedHistoryForCtr(), structure, [&]() -> TMirrorBuffer<ui32> {
                return std::move(learnBins);
            });
        }
        return hasHistory ? std::move(testBins) : std::move(learnBins);
    });
}

void NCatboostCuda::CacheBinsForModel(TScopedCacheHolder& cacheHolder,
                                      const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
                                      const NCatboostCuda::TFeatureParallelDataSet& dataSet,
                                      const NCatboostCuda::TObliviousTreeStructure& structure,
                                      TMirrorBuffer<ui32>&& bins) {
    bool hasPermutationCtrs = HasPermutationDependentSplit(structure, featuresManager);
    const auto& scope = hasPermutationCtrs ? dataSet.GetPermutationDependentScope() : dataSet.GetPermutationIndependentScope();
    cacheHolder.CacheOnly(scope, structure, [&]() -> TMirrorBuffer<ui32> {
        TMirrorBuffer<ui32> cachedBins = std::move(bins);
        return cachedBins;
    });
}

bool NCatboostCuda::HasPermutationDependentSplit(const NCatboostCuda::TObliviousTreeStructure& structure,
                                                 const NCatboostCuda::TBinarizedFeaturesManager& featuresManager) {
    for (const auto& split : structure.Splits) {
        if (featuresManager.IsCtr(split.FeatureId)) {
            auto ctr = featuresManager.GetCtr(split.FeatureId);
            if (featuresManager.IsPermutationDependent(ctr)) {
                return true;
            }
        }
    }
    return false;
}

TString NCatboostCuda::SplitConditionToString(
    const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
    const NCatboostCuda::TBinarySplit& split) {
    TString splitTypeMessage;

    if (split.SplitType == EBinSplitType::TakeBin) {
        splitTypeMessage = "TakeBin";
    } else {
        const auto& borders = featuresManager.GetBorders(split.FeatureId);
        auto nanMode = featuresManager.GetNanMode(split.FeatureId);
        TStringBuilder messageBuilder;
        if (nanMode == ENanMode::Forbidden) {
            messageBuilder << ">" << featuresManager.GetBorders(split.FeatureId)[split.BinIdx];
        } else if (nanMode == ENanMode::Min) {
            if (split.BinIdx > 0) {
                messageBuilder << ">" << featuresManager.GetBorders(split.FeatureId)[split.BinIdx - 1];
            } else {
                messageBuilder << "== -inf (nan)";
            }
        } else {
            CB_ENSURE(nanMode == ENanMode::Max, "Unexpected nan mode");
            if (split.BinIdx < borders.size()) {
                messageBuilder << ">" << featuresManager.GetBorders(split.FeatureId)[split.BinIdx];
            } else {
                CB_ENSURE(split.BinIdx == borders.size(), "Bin index is too large");
                messageBuilder << "== +inf (nan)";
            }
        }

        splitTypeMessage = messageBuilder;
    }
    return splitTypeMessage;
}

TString NCatboostCuda::SplitConditionToString(
    const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
    const NCatboostCuda::TBinarySplit& split,
    ESplitValue value) {
    TString splitTypeMessage;
    const bool inverse = value == ESplitValue::Zero;

    if (split.SplitType == EBinSplitType::TakeBin) {
        splitTypeMessage = inverse ? "SkipBin" : "TakeBin";
    } else {
        const auto& borders = featuresManager.GetBorders(split.FeatureId);
        auto nanMode = featuresManager.GetNanMode(split.FeatureId);
        TStringBuilder messageBuilder;
        if (nanMode == ENanMode::Forbidden) {
            messageBuilder << (inverse ? "<=" : ">") << featuresManager.GetBorders(split.FeatureId)[split.BinIdx];
        } else if (nanMode == ENanMode::Min) {
            if (split.BinIdx > 0) {
                messageBuilder << (inverse ? "<=" : ">") << featuresManager.GetBorders(split.FeatureId)[split.BinIdx - 1];
            } else {
                messageBuilder << (inverse ? "!=" : "==") << " -inf (nan)";
            }
        } else {
            CB_ENSURE(nanMode == ENanMode::Max, "Unexpected nan mode");
            if (split.BinIdx < borders.size()) {
                messageBuilder << (inverse ? "<=" : ">") << featuresManager.GetBorders(split.FeatureId)[split.BinIdx];
            } else {
                CB_ENSURE(split.BinIdx == borders.size(), "Bin index is too large");
                messageBuilder << (inverse ? "!=" : "==") << " +inf (nan)";
            }
        }

        splitTypeMessage = messageBuilder;
    }
    return splitTypeMessage;
}

void NCatboostCuda::PrintBestScore(const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
                                   const NCatboostCuda::TBinarySplit& bestSplit, double score, ui32 depth) {
    TString splitTypeMessage = SplitConditionToString(featuresManager, bestSplit);
    TStringBuilder logEntry;
    logEntry
        << "Best split for depth " << depth << ": " << bestSplit.FeatureId << " / " << bestSplit.BinIdx << " ("
        << splitTypeMessage << ")"
        << " with score " << score;
    if (featuresManager.IsCtr(bestSplit.FeatureId)) {
        logEntry
            << " tensor : " << featuresManager.GetCtr(bestSplit.FeatureId).FeatureTensor << "  (ctr type "
            << featuresManager.GetCtr(bestSplit.FeatureId).Configuration.Type << ")";
    }
    CATBOOST_INFO_LOG << logEntry << Endl;
}

NCatboostCuda::TBinarySplit NCatboostCuda::ToSplit(const NCatboostCuda::TBinarizedFeaturesManager& manager, const TBestSplitProperties& props) {
    CB_ENSURE(props.Defined(), "Need best split properties");
    if (manager.IsFeatureBundle(props.FeatureId)) {
        return manager.TranslateFeatureBundleSplitToBinarySplit(props.FeatureId, props.BinId);
    }
    TBinarySplit bestSplit;
    bestSplit.FeatureId = props.FeatureId;
    bestSplit.BinIdx = props.BinId;
    //We need to adjust binIdx. Float arithmetic could generate empty bin splits for ctrs
    if (manager.IsCat(props.FeatureId)) {
        bestSplit.SplitType = EBinSplitType::TakeBin;
        bestSplit.BinIdx = Min<ui32>(manager.GetBinCount(bestSplit.FeatureId), bestSplit.BinIdx);
    } else {
        bestSplit.SplitType = EBinSplitType::TakeGreater;
        bestSplit.BinIdx = Min<ui32>(manager.GetBorders(bestSplit.FeatureId).size() - 1, bestSplit.BinIdx);
    }
    return bestSplit;
}
