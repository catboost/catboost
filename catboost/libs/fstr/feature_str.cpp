#include "feature_str.h"

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/utility.h>
#include <util/generic/xrange.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <utility>

TString TFeature::BuildDescription(const NCB::TFeaturesLayout& layout) const {
    TStringBuilder result;
    if (Type == ESplitType::OnlineCtr) {
        result << "{";
        int feature_count = 0;
        auto proj = Ctr.Base.Projection;

        for (const int featureIdx : proj.CatFeatures) {
            if (feature_count++ > 0) {
                result << ", ";
            }
            result << BuildFeatureDescription(layout, featureIdx, EFeatureType::Categorical);
        }

        for (const auto& feature : proj.BinFeatures) {
            if (feature_count++ > 0) {
                result << ", ";
            }
            result << BuildFeatureDescription(layout, feature.FloatFeature, EFeatureType::Float);
        }

        for (const TOneHotSplit& feature : proj.OneHotFeatures) {
            if (feature_count++ > 0) {
                result << ", ";
            }
            result << BuildFeatureDescription(layout, feature.CatFeatureIdx, EFeatureType::Categorical);
        }
        result << "}";
        result << " prior_num=" << Ctr.PriorNum;
        result << " prior_denom=" << Ctr.PriorDenom;
        result << " targetborder=" << Ctr.TargetBorderIdx;
        result << " type=" << Ctr.Base.CtrType;
    } else if (Type == ESplitType::FloatFeature) {
        result << BuildFeatureDescription(layout, FeatureIdx, EFeatureType::Float);
    } else {
        Y_ASSERT(Type == ESplitType::OneHotFeature);
        result << BuildFeatureDescription(layout, FeatureIdx, EFeatureType::Categorical);
    }
    return result;
}

int GetMaxSrcFeature(const TVector<TMxTree>& trees) {
    int res = -1;
    for (const auto& tree : trees) {
        const auto& features = tree.SrcFeatures;
        for (auto f : features) {
            if (f > res) {
                res = f;
            }
        }
    }
    return res;
}

void ConvertToPercents(TVector<double>& res) {
    double total = Accumulate(res.begin(), res.end(), 0.0);
    for (auto& x : res) {
        x *= 100. / total;
    }
}

static ui64 CalcMaxBinFeaturesCount(const TVector<TVector<ui64>>& trueDocsPerFeature) {
    ui64 res = 0;
    for (const auto& x : trueDocsPerFeature) {
        res = Max(res, x.size());
    }
    return res;
}

static TVector<double> PrecalcLogFactorials(ui64 maxLog) {
    TVector<double> res(maxLog + 1);
    res[0] = 0;
    for (int i = 1; i < res.ysize(); ++i) {
        res[i] = res[i - 1] + log((double)i);
    }
    return res;
}

TVector<double> CalcFeaturesInfo(TVector<TVector<ui64>> trueDocsPerFeature,
                                 const ui64 docCount,
                                 bool symmetric) {
    ui64 maxBinFeaturesCount = CalcMaxBinFeaturesCount(trueDocsPerFeature);
    TVector<double> facLogs = PrecalcLogFactorials(docCount + maxBinFeaturesCount + 1);

    const int featuresCount = trueDocsPerFeature.size();
    TVector<double> result(featuresCount);

    for (int featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
        auto& trueDocs = trueDocsPerFeature[featureIdx];
        Sort(trueDocs.begin(), trueDocs.end());
        trueDocs.push_back(docCount);
        double infAmount = 0;

        if (symmetric) {
            ui64 docsInBucket = trueDocs[0];
            infAmount += facLogs[docsInBucket];
            for (int i = 0; i + 1 < trueDocs.ysize(); ++i) {
                docsInBucket = trueDocs[i + 1] - trueDocs[i];
                infAmount += facLogs[docsInBucket];
            }
            infAmount -= facLogs[trueDocs.back() + trueDocs.ysize() - 1];
            infAmount += facLogs[trueDocs.ysize() - 1];
        } else {
            for (int i = 0; i + 1 < trueDocs.ysize(); ++i) {
                ui64 nA = trueDocs[i], nB = trueDocs[i + 1] - trueDocs[i];
                double splitInfo = facLogs[nA] + facLogs[nB] - facLogs[nA + nB + 1];
                infAmount += splitInfo;
            }
        }

        if (infAmount != 0) {
            result[featureIdx] = -infAmount;
        }
    }
    return result;
}

TVector<double> CalculateEffectToInfoRate(const TVector<double>& effect,
                                          const TVector<double>& info) {
    Y_ASSERT(effect.size() == info.size());
    TVector<double> featuresEfficiency(effect.size());
    auto efficiencyMax = double{};
    for (const auto& index : xrange(featuresEfficiency.size())) {
        const auto efficiency = effect[index] / (info[index] + 1e-20);
        if (!IsNan(efficiency)) {
            efficiencyMax = Max(efficiencyMax, efficiency);
        }
        featuresEfficiency[index] = efficiency;
    }

    for (auto& value : featuresEfficiency) {
        value /= efficiencyMax;
    }

    return featuresEfficiency;
}

TVector<TFeaturePairInteractionInfo> CalcMostInteractingFeatures(const TVector<TMxTree>& trees,
                                                                 int topPairsCount) {
    int featureCount = GetMaxSrcFeature(trees) + 1;
    THashMap<std::pair<int, int>, double> sumInteractions;

    for (int i = 0; i < trees.ysize(); ++i) {
        const TMxTree& tree = trees[i];
        for (int f1 = 0; f1 < tree.SrcFeatures.ysize() - 1; ++f1) {
            for (int f2 = f1 + 1; f2 < tree.SrcFeatures.ysize(); ++f2) {
                int n1 = 1 << f1;
                int n2 = 1 << f2;
                double delta = 0;
                for (int leafIdx = 0; leafIdx < tree.Leaves.ysize(); ++leafIdx) {
                    int var1 = (leafIdx & n1) != 0;
                    int var2 = (leafIdx & n2) != 0;
                    int sign = (var1 ^ var2) ? 1 : -1;
                    for (int valInLeafIdx = 0; valInLeafIdx < tree.Leaves[leafIdx].Vals.ysize(); ++valInLeafIdx) {
                        delta += sign * tree.Leaves[leafIdx].Vals[valInLeafIdx];
                    }
                }
                int srcFeature1 = tree.SrcFeatures[f1];
                int srcFeature2 = tree.SrcFeatures[f2];
                if (srcFeature2 < srcFeature1) {
                    DoSwap(srcFeature1, srcFeature2);
                }
                if (srcFeature1 == srcFeature2) {
                    continue;
                }
                sumInteractions[std::make_pair(srcFeature1, srcFeature2)] += fabs(delta);
            }
        }
    }

    TVector<TFeaturePairInteractionInfo> pairsInfo;

    if (topPairsCount == EXISTING_PAIRS_COUNT) {
        for (const auto& pairInteraction : sumInteractions) {
            pairsInfo.push_back(TFeaturePairInteractionInfo(sumInteractions[pairInteraction.first],
                                                            pairInteraction.first.first, pairInteraction.first.second));
        }
    } else {
        for (int f1 = 0; f1 < featureCount; ++f1) {
            for (int f2 = f1 + 1; f2 < featureCount; ++f2) {
                pairsInfo.push_back(TFeaturePairInteractionInfo(sumInteractions[std::make_pair(f1,f2)], f1, f2));
            }
        }
    }

    std::sort(pairsInfo.rbegin(), pairsInfo.rend());
    if (topPairsCount != EXISTING_PAIRS_COUNT && pairsInfo.ysize() > topPairsCount) {
        pairsInfo.resize(topPairsCount);
    }

    return pairsInfo;
}

TFeature GetFeature(const TModelSplit& split) {
    TFeature result;
    result.Type = split.Type;
    switch(result.Type) {
        case ESplitType::FloatFeature:
            result.FeatureIdx = split.FloatFeature.FloatFeature;
            break;
        case ESplitType::OneHotFeature:
            result.FeatureIdx = split.OneHotFeature.CatFeatureIdx;
            break;
        case ESplitType::OnlineCtr:
            result.Ctr = split.OnlineCtr.Ctr;
            break;
        case ESplitType::EstimatedFeature:
            CB_ENSURE(false, "Text features is not supported in fstr mode yet");
            break;
    }
    return result;
}
