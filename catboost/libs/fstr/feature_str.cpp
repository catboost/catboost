#include "feature_str.h"

#include <util/stream/output.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>
#include <util/generic/xrange.h>
#include <util/generic/hash.h>
#include <library/containers/2d_array/2d_array.h>

static int GetMaxSrcFeature(const TVector<TMxTree>& trees) {
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

static void ConvertToPercents(TVector<double>& res) {
    double total = Accumulate(res.begin(), res.end(), 0.0);
    for (auto& x : res) {
        x *= 100. / total;
    }
}

TVector<double> CalcEffect(const TVector<TMxTree>& trees,
                           const TVector<TVector<ui64>>& docCountInLeaf) {
    TVector<double> res;
    int featureCount = GetMaxSrcFeature(trees) + 1;
    res.resize(featureCount);

    for (int treeIdx = 0; treeIdx < trees.ysize(); treeIdx++) {
        const auto& tree = trees[treeIdx];
        for (int feature = 0; feature < tree.SrcFeatures.ysize(); feature++) {
            int srcIdx = tree.SrcFeatures[feature];

            for (int leafIdx = 0; leafIdx < tree.Leafs.ysize(); ++leafIdx) {
                int inverted = leafIdx ^ (1 << feature);
                if (inverted < leafIdx) {
                    continue;
                }

                double count1 = docCountInLeaf[treeIdx][leafIdx];
                double count2 = docCountInLeaf[treeIdx][inverted];
                if (count1 == 0 || count2 == 0) {
                    continue;
                }

                for (int valInLeafIdx = 0; valInLeafIdx < tree.Leafs[leafIdx].Vals.ysize(); ++valInLeafIdx) {
                    double val1 = tree.Leafs[leafIdx].Vals[valInLeafIdx];
                    double val2 = tree.Leafs[inverted].Vals[valInLeafIdx];

                    double avrg = (val1 * count1 + val2 * count2) / (count1 + count2);
                    double dif = Sqr(val1 - avrg) * count1 + Sqr(val2 - avrg) * count2;

                    res[srcIdx] += dif;
                }
            }
        }
    }
    ConvertToPercents(res);
    return res;
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
                for (int leafIdx = 0; leafIdx < tree.Leafs.ysize(); ++leafIdx) {
                    int var1 = (leafIdx & n1) != 0;
                    int var2 = (leafIdx & n2) != 0;
                    int sign = (var1 ^ var2) ? 1 : -1;
                    for (int valInLeafIdx = 0; valInLeafIdx < tree.Leafs[leafIdx].Vals.ysize(); ++valInLeafIdx) {
                        delta += sign * tree.Leafs[leafIdx].Vals[valInLeafIdx];
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
