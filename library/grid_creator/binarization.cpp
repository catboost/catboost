#include "binarization.h"

#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/generic/queue.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>

yhash_set<float> BestSplit(yvector<float>& featureVals,
                           int bordersCount,
                           EBorderSelectionType type,
                           bool nanValuesIsInfty) {
    std::unique_ptr<NSplitSelection::IBinarizer> binarizer;
    switch (type) {
        case (EBorderSelectionType::UniformAndQuantiles):
            binarizer.reset(new NSplitSelection::TMedianPlusUniformBinarizer());
            break;
        case (EBorderSelectionType::MinEntropy):
            binarizer.reset(new NSplitSelection::TMinEntropyBinarizer());
            break;
        case (EBorderSelectionType::MaxLogSum):
            binarizer.reset(new NSplitSelection::TMaxSumLogBinarizer());
            break;
        case (EBorderSelectionType::Median):
            binarizer.reset(new NSplitSelection::TMedianBinarizer());
            break;
        case (EBorderSelectionType::GreedyLogSum):
            binarizer.reset(new NSplitSelection::TMedianInBinBinarizer());
            break;
        case (EBorderSelectionType::Uniform):
            binarizer.reset(new NSplitSelection::TUniformBinarizer());
            break;
        default:
            Y_VERIFY(false);
    }

    Sort(featureVals.begin(), featureVals.end());
    if (nanValuesIsInfty) {
        featureVals.erase(std::remove_if(featureVals.begin(), featureVals.end(), [](auto v) { return std::isnan(v); }), featureVals.end());
    }

    return binarizer->BestSplit(featureVals, bordersCount, true);
}

namespace {
enum ESF {
    E_Base,       // Base dp solution
    E_Old_Linear, // First linear dp version
    E_Base2,      // Full dp solution with assumptions that at least on value in each bin. All other modes will be under same assumption.
    E_Linear,     // Similar to E_Old_Linear
    E_Linear_2L,  // Run 2 loops and choose best one
    E_Safe,       // Correct solution in good time
    E_RLM,        // Recursive linear model
    E_RLM2,       // Almost always O(wsize * bins) time
    E_DaC,        // Guaranteed O(wsize * log(wsize) * bins) time
    E_SF_End
};
} // namespace

enum class EPenaltyType {
    MinEntropy,
    MaxSumLog,
    W2
};

static double Penalty(double weight, double expected_weight, EPenaltyType type) {
    (void)expected_weight;
    if (type == EPenaltyType::MinEntropy) {
        return weight * log(weight + 1e-8);
    } else if (type == EPenaltyType::MaxSumLog) {
        return -log(weight + 1e-8);
    } else if (type == EPenaltyType::W2) {
        return weight * weight;
    } else {
        Y_VERIFY(false);
    }
    // return fabs(weight - expected_weight); // Module
    // return weight*weight; // Square
}

template <typename TWeightType>
static void BestSplit(const yvector<TWeightType>& weights,
                      size_t bordersCount,
                      yvector<size_t>& thresholds,
                      EPenaltyType type,
                      ESF mode) {
    size_t bins = bordersCount + 1;
    // Safety checks
    if (bins <= 1 || weights.empty()) {
        return;
    }
    if (thresholds.size() != bins - 1) {
        thresholds.resize(bins - 1);
    }
    size_t wsize = weights.size();
    if (wsize <= bins) {
        for (size_t i = 0; i + 1 < wsize; ++i) {
            thresholds[i] = i;
        }
        for (size_t i = wsize - 1; i + 1 < bins; ++i) {
            thresholds[i] = wsize - 1;
        }
        return;
    }

    // Initialize
    const double Eps = 1e-12;
    yvector<TWeightType> sweights(weights);
    for (size_t i = 1; i < wsize; ++i) {
        sweights[i] += sweights[i - 1];
    }
    double expected = double(sweights[wsize - 1]) / bins;
    size_t dsize = ((mode == E_Base) || (mode == E_Old_Linear)) ? wsize : (wsize - bins + 1);
    yvector<yvector<size_t>> bestSolutions(bins - 2, yvector<size_t>(dsize));
    yvector<double> current_error(dsize), prevError(dsize);
    for (size_t i = 0; i < dsize; ++i) {
        current_error[i] = Penalty(double(sweights[i]), expected, type);
    }
    // For 2 loops runs:
    yvector<size_t> bs1(dsize), bs2(dsize);
    yvector<double> e1(dsize), e2(dsize);

    // Main loop
    for (size_t l = 0; l < bins - 2; ++l) {
        current_error.swap(prevError);
        if (mode == E_Base) {
            for (size_t j = 0; j < wsize; ++j) {
                size_t bestIndex = 0;
                double bestError = prevError[0] + Penalty(double(sweights[j] - sweights[0]), expected, type);
                for (size_t i = 1; i <= j; ++i) {
                    double newError = prevError[i] + Penalty(double(sweights[j] - sweights[i]), expected, type);
                    if (newError <= bestError) {
                        bestError = newError;
                        bestIndex = i;
                    }
                }
                bestSolutions[l][j] = bestIndex;
                current_error[j] = bestError;
            }
        } else if (mode == E_Old_Linear) {
            size_t i = 0;
            for (size_t j = 0; j < wsize; ++j) {
                double bestError = prevError[i] + Penalty(double(sweights[j] - sweights[i]), expected, type);
                for (++i; i < j; ++i) {
                    double newError = prevError[i] + Penalty(double(sweights[j] - sweights[i]), expected, type);
                    if (newError > bestError + Eps) {
                        break;
                    }
                    bestError = newError;
                }
                --i;
                bestSolutions[l][j] = i;
                current_error[j] = bestError;
            }
        } else if (mode == E_Base2) {
            for (size_t j = 0; j < dsize; ++j) {
                size_t bestIndex = 0;
                double bestError = prevError[0] + Penalty(double(sweights[l + j + 1] - sweights[l]), expected, type);
                for (size_t i = 1; i <= j; ++i) {
                    double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                    if (newError <= bestError) {
                        bestError = newError;
                        bestIndex = i;
                    }
                }
                bestSolutions[l][j] = bestIndex;
                current_error[j] = bestError;
            }
        } else if (mode == E_Linear) {
            size_t i = 0;
            for (size_t j = 0; j < dsize; ++j) {
                double bestError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                for (++i; i <= j; ++i) {
                    double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                    if (newError > bestError + Eps) {
                        break;
                    }
                    bestError = newError;
                }
                --i;
                bestSolutions[l][j] = i;
                current_error[j] = bestError;
            }
        } else if ((mode == E_Linear_2L) || (mode == E_Safe) || (mode == E_RLM)) {
            // First loop
            size_t left = 0;
            for (size_t j = 0; j < dsize; ++j) {
                double bestError = prevError[left] + Penalty(double(sweights[l + j + 1] - sweights[l + left]), expected, type);
                for (++left; left <= j; ++left) {
                    double newError = prevError[left] + Penalty(double(sweights[l + j + 1] - sweights[l + left]), expected, type);
                    if (newError > bestError + Eps) {
                        break;
                    }
                    bestError = newError;
                }
                --left;
                bs1[j] = left;
                e1[j] = bestError;
            }

            // Second loop (inverted)
            left = 0;
            for (size_t j = 0; j < dsize; ++j) {
                left = Max(left, j);
                double bestError = prevError[dsize - left - 1] + Penalty(double(sweights[l + dsize - j] - sweights[l + dsize - left - 1]), expected, type);
                for (++left; left < dsize; ++left) {
                    double newError = prevError[dsize - left - 1] + Penalty(double(sweights[l + dsize - j] - sweights[l + dsize - left - 1]), expected, type);
                    if (newError > bestError + Eps) {
                        break;
                    }
                    bestError = newError;
                }
                --left;
                bs2[dsize - j - 1] = dsize - left - 1;
                e2[dsize - j - 1] = bestError;
            }

            // Set best
            if (mode == E_Linear_2L) {
                for (size_t j = 0; j < dsize; ++j) {
                    if (e1[j] < e2[j] + Eps) {
                        bestSolutions[l][j] = bs1[j];
                        current_error[j] = e1[j];
                    } else {
                        bestSolutions[l][j] = bs2[j];
                        current_error[j] = e2[j];
                    }
                }
            } else if (mode == E_Safe) {
                for (size_t j = 0; j < dsize; ++j) {
                    double bestError = e1[j];
                    size_t bestIndex = bs1[j];
                    for (size_t i = bs1[j] + 1; i <= bs2[j]; ++i) {
                        double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                        if (newError <= bestError) {
                            bestError = newError;
                            bestIndex = i;
                        }
                    }
                    bestSolutions[l][j] = bestIndex;
                    current_error[j] = bestError;
                }
            } else if (mode == E_RLM) {
                for (size_t j = 0; j < dsize; ++j) {
                    if (bs1[j] + 1 >= bs2[j]) {
                        // Everything is fine!
                        bestSolutions[l][j] = bs1[j];
                        current_error[j] = e1[j];
                    } else {
                        // Rebuild 2L solutions.
                        bool rebuild_required_1, rebuild_required_2;
                        // Step 1. Calc borders to rebuild.
                        size_t j1 = j, j2 = j + 1;
                        for (; j2 < dsize; ++j2) {
                            if (bs1[j2] + 1 >= bs2[j2]) {
                                break;
                            }
                        }
                        --j2;
                        // Step 2. Find ideal for j1
                        {
                            double bestError = e1[j1];
                            size_t bestIndex = bs1[j1];
                            for (size_t i = bs1[j1] + 2; i <= bs2[j1]; ++i) // We can start from bs1[j1] + 2 because bs1[j1] + 1 was checked before
                            {
                                double newError = prevError[i] + Penalty(double(sweights[l + j1 + 1] - sweights[l + i]), expected, type);
                                if (newError < bestError - Eps) {
                                    bestError = newError;
                                    bestIndex = i;
                                }
                            }
                            rebuild_required_1 = (bs1[j1] != bestIndex);
                            bestSolutions[l][j1] = bs1[j1] = bs2[j1] = bestIndex;
                            current_error[j1] = e1[j1] = e2[j1] = bestError;
                        }
                        if (j2 > j1 + 1) {
                            // Step 3. Rebuild first loop
                            if (rebuild_required_1) {
                                size_t i = bs1[j1];
                                for (size_t ji = j1 + 1; ji <= j2; ++ji) {
                                    double bestError = prevError[i] + Penalty(double(sweights[l + ji + 1] - sweights[l + i]), expected, type);
                                    for (++i; i <= ji; ++i) {
                                        double newError = prevError[i] + Penalty(double(sweights[l + ji + 1] - sweights[l + i]), expected, type);
                                        if (newError > bestError + Eps) {
                                            break;
                                        }
                                        bestError = newError;
                                    }
                                    --i;
                                    bs1[ji] = i;
                                    e1[ji] = bestError;
                                }
                            }

                            // Step 4. Find ideal for j2
                            {
                                double bestError = e2[j2];
                                size_t bestIndex = bs2[j2];
                                for (size_t i = bs1[j2]; i < bs2[j2] - 1; ++i) // bs2[j2]-1 was checked before
                                {
                                    double newError = prevError[i] + Penalty(double(sweights[l + j2 + 1] - sweights[l + i]), expected, type);
                                    if (newError < bestError - Eps) {
                                        bestError = newError;
                                        bestIndex = i;
                                    }
                                }
                                rebuild_required_2 = (bs2[j2] != bestIndex);
                                bestSolutions[l][j2] = bs1[j2] = bs2[j2] = bestIndex;
                                current_error[j2] = e1[j2] = e2[j2] = bestError;
                            }

                            // Step 5. Rebuild second loop
                            if (rebuild_required_2) {
                                size_t i = dsize - bs2[j2] - 1;
                                for (size_t ji = dsize - j2; ji < dsize - j1 - 1; ++ji) {
                                    i = Max(i, ji);
                                    double bestError = prevError[dsize - i - 1] + Penalty(double(sweights[l + dsize - ji] - sweights[l + dsize - i - 1]), expected, type);
                                    for (++i; i < dsize; ++i) {
                                        double newError = prevError[dsize - i - 1] + Penalty(double(sweights[l + dsize - ji] - sweights[l + dsize - i - 1]), expected, type);
                                        if (newError > bestError + Eps) {
                                            break;
                                        }
                                        bestError = newError;
                                    }
                                    --i;
                                    bs2[dsize - ji - 1] = dsize - i - 1;
                                    e2[dsize - ji - 1] = bestError;
                                }
                            }
                        }
                    }
                }
            }
        } else if (mode == E_RLM2) {
            // First forward loop
            {
                size_t i = 0;
                for (size_t j = 0; j < dsize; ++j) {
                    double bestError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                    for (++i; i <= j; ++i) {
                        double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                        if (newError > bestError + Eps) {
                            break;
                        }
                        bestError = newError;
                    }
                    --i;
                    bs1[j] = i;
                    e1[j] = bestError;
                }
            }

            // First inverted loop
            {
                size_t i = 0;
                for (size_t j = 0; j < dsize; ++j) {
                    i = Max(i, j);
                    const size_t maxi = dsize - bs1[dsize - j - 1] - 1;
                    if (i + 1 >= maxi) {
                        bs2[dsize - j - 1] = bs1[dsize - j - 1];
                        e2[dsize - j - 1] = e1[dsize - j - 1];
                        i = maxi;
                        continue;
                    }
                    double bestError = e1[dsize - j - 1];
                    for (; i + 1 < maxi; ++i) {
                        double newError = prevError[dsize - i - 1] + Penalty(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]), expected, type);
                        if (newError + Eps < bestError) {
                            bestError = newError;
                            break;
                        }
                    }
                    if (i + 1 >= maxi) {
                        i = maxi;
                    } else {
                        for (++i; i + 1 < maxi; ++i) {
                            double newError = prevError[dsize - i - 1] + Penalty(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]), expected, type);
                            if (newError > bestError + Eps) {
                                break;
                            }
                            bestError = newError;
                        }
                        --i;
                    }
                    bs2[dsize - j - 1] = dsize - i - 1;
                    e2[dsize - j - 1] = bestError;
                }
            }

            for (size_t k = 0; k < dsize; ++k) {
                while (bs1[k] + 1 < bs2[k]) {
                    // Rebuild required
                    size_t maxj = dsize;

                    // Forward loop
                    {
                        size_t i = bs1[k] + 2;
                        for (size_t j = k; j < maxj; ++j) {
                            if (i <= bs1[j]) {
                                maxj = j;
                                break;
                            }
                            const size_t maxi = bs2[j];
                            if (i + 1 >= maxi) {
                                i = maxi;
                                bs1[j] = i;
                                e1[j] = e2[j];
                                continue;
                            }
                            double bestError = e2[j];
                            for (; i + 1 < maxi; ++i) {
                                double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                                if (newError + Eps < bestError) {
                                    bestError = newError;
                                    break;
                                }
                            }
                            if (i + 1 >= maxi) {
                                i = maxi;
                            } else {
                                for (++i; i + 1 < maxi; ++i) {
                                    double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                                    if (newError > bestError + Eps) {
                                        break;
                                    }
                                    bestError = newError;
                                }
                                --i;
                            }
                            bs1[j] = i;
                            e1[j] = bestError;
                        }
                    }

                    // Inverted loop
                    {
                        size_t j1 = dsize - maxj;
                        size_t j2 = dsize - k;
                        size_t i = dsize - bs2[dsize - j1 - 1] - 1 + 2;
                        for (size_t j = j1; j < j2; ++j) {
                            const size_t maxi = dsize - bs1[dsize - j - 1] - 1;
                            if (i + 1 >= maxi) {
                                bs2[dsize - j - 1] = bs1[dsize - j - 1];
                                e2[dsize - j - 1] = e1[dsize - j - 1];
                                i = maxi;
                                continue;
                            }
                            double bestError = e1[dsize - j - 1];
                            for (; i + 1 < maxi; ++i) {
                                double newError = prevError[dsize - i - 1] + Penalty(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]), expected, type);
                                if (newError + Eps < bestError) {
                                    bestError = newError;
                                    break;
                                }
                            }
                            if (i + 1 >= maxi) {
                                i = maxi;
                            } else {
                                for (++i; i + 1 < maxi; ++i) {
                                    double newError = prevError[dsize - i - 1] + Penalty(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]), expected, type);
                                    if (newError > bestError + Eps) {
                                        break;
                                    }
                                    bestError = newError;
                                }
                                --i;
                            }
                            bs2[dsize - j - 1] = dsize - i - 1;
                            e2[dsize - j - 1] = bestError;
                        }
                    }
                }
                // Everything is fine now!
                bestSolutions[l][k] = bs1[k];
                current_error[k] = e1[k];
            }
        } else if (mode == E_DaC)
        {
            typedef std::tuple<size_t, size_t, size_t, size_t> t4;
            yqueue<t4> qr;
            qr.push(std::make_tuple(0, dsize, 0, dsize));
            while (!qr.empty())
            {
                size_t jbegin, jend, ibegin, iend;
                std::tie(jbegin, jend, ibegin, iend) = qr.front();
                qr.pop();
                if (jbegin >= jend)
                {
                    // empty box
                }
                else if (iend - ibegin == 1)
                {
                    // i is already fixed
                    for (size_t j = jbegin; j < jend; ++j)
                    {
                        bestSolutions[l][j] = ibegin;
                        current_error[j] = prevError[ibegin] + Penalty(double(sweights[l + j + 1] - sweights[l + ibegin]), expected, type);
                    }
                }
                else
                {
                    size_t j = (jbegin + jend) / 2;
                    size_t bestIndex = ibegin;
                    double bestError = prevError[ibegin] + Penalty(double(sweights[l + j + 1] - sweights[l + ibegin]), expected, type);
                    size_t iend2 = Min(iend, j + 1);
                    for (size_t i = ibegin + 1; i < iend2; ++i)
                    {
                        double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                        if (newError <= bestError)
                        {
                            bestError = newError;
                            bestIndex = i;
                        }
                    }
                    bestSolutions[l][j] = bestIndex;
                    current_error[j] = bestError;
                    qr.push(std::make_tuple(jbegin, j, ibegin, bestIndex + 1));
                    qr.push(std::make_tuple(j + 1, jend, bestIndex, iend));
                }
            }
        }
    }

    // Last match
    {
        current_error.swap(prevError);
        size_t l = bins - 2;
        size_t j = dsize - 1; // we don't care about other
        size_t bestIndex = 0;
        if ((mode == E_Base) || (mode == E_Old_Linear)) {
            double bestError = prevError[0] + Penalty(double(sweights[j] - sweights[0]), expected, type);
            for (size_t i = 1; i <= j; ++i) {
                double newError = prevError[i] + Penalty(double(sweights[j] - sweights[i]), expected, type);
                if (newError < bestError) {
                    bestError = newError;
                    bestIndex = i;
                }
            }
        } else {
            double bestError = prevError[0] + Penalty(double(sweights[l + j + 1] - sweights[l]), expected, type);
            for (size_t i = 1; i <= j; ++i) {
                double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                if (newError < bestError) {
                    bestError = newError;
                    bestIndex = i;
                }
            }
        }
        // Now we are ready to fill answer
        thresholds[bins - 2] = bestIndex;
        for (; l > 0; --l) {
            bestIndex = bestSolutions[l - 1][bestIndex];
            thresholds[l - 1] = bestIndex;
        }
        // Adjust
        if (!((mode == E_Base) || (mode == E_Old_Linear))) {
            for (l = 0; l < thresholds.size(); ++l) {
                thresholds[l] += l;
            }
        }
    }
}

// Border before element with value "border"
static float RegularBorder(float border, const yvector<float>& sortedValues) {
    yvector<float>::const_iterator lowerBound = LowerBound(sortedValues.begin(), sortedValues.end(), border);

    if (lowerBound == sortedValues.end()) // binarizing to always false
        return Max(2.f * sortedValues.back(), sortedValues.back() + 1.f);

    if (lowerBound == sortedValues.begin()) // binarizing to always true
        return Min(.5f * sortedValues.front(), 2.f * sortedValues.front());

    float res = (lowerBound[0] + lowerBound[-1]) * .5f;
    if (res == lowerBound[0]) // wrong side rounding (should be very scarce)
        res = lowerBound[-1];

    return res;
}


static yhash_set<float> BestSplit(const yvector<float>& values,
                           const yvector<float>& weight,
                           size_t bordersCount,
                           EPenaltyType type) {
    // Positions after which threshold should be inserted.
    yvector<size_t> thresholds;
    BestSplit(weight, bordersCount, thresholds, type, E_RLM2);

    yhash_set<float> borders;
    for (auto t : thresholds) {
        if (t + 1 != values.size()) {
            borders.insert((values[t] + values[t + 1]) / 2);
        }
    }
    return borders;
}

static yhash_set<float> SplitWithGuaranteedOptimum(
        yvector<float>& featureValues,
        int bordersCount,
        EPenaltyType type,
        bool isSorted) {
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }

    yvector<float> features;
    yvector<float> weights;
    for (auto f : featureValues) {
        if (features.empty() || features.back() != f) {
            features.push_back(f);
            weights.push_back(1);
        } else {
            weights.back()++;
        }
    }
    return BestSplit(features, weights, bordersCount, type);
}

static yhash_set<float> GenerateMedianBorders(
        const yvector<float>& featureValues, int bordersCount) {
    yhash_set<float> result;
    ui64 total = featureValues.size();
    if (total == 0 || featureValues.front() == featureValues.back()) {
        return result;
    }

    for (int i = 0; i < bordersCount; ++i) {
        ui64 i1 = (i + 1) * total / (bordersCount + 1);
        i1 = Min(i1, total - 1);
        float val1 = featureValues[i1];
        if (val1 != featureValues[0]) {
            result.insert(RegularBorder(val1, featureValues));
        }
    }
    return result;
}

namespace NSplitSelection {

yhash_set<float> TMinEntropyBinarizer::BestSplit(
        yvector<float>& featureValues, int bordersCount, bool isSorted) const {
    return SplitWithGuaranteedOptimum(featureValues, bordersCount, EPenaltyType::MinEntropy, isSorted);
}

yhash_set<float> TMaxSumLogBinarizer::BestSplit(
        yvector<float>& featureValues, int bordersCount, bool isSorted) const {
    return SplitWithGuaranteedOptimum(featureValues, bordersCount, EPenaltyType::MaxSumLog, isSorted);
}

yhash_set<float> TMedianBinarizer::BestSplit(
        yvector<float>& featureValues, int bordersCount, bool isSorted) const {
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }
    return GenerateMedianBorders(featureValues, bordersCount);
}

yhash_set<float> TMedianPlusUniformBinarizer::BestSplit(
        yvector<float>& featureValues, int bordersCount, bool isSorted) const {

    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }

    if (featureValues.empty() || featureValues.front() == featureValues.back()) {
        return yhash_set<float>();
    }

    int halfBorders = bordersCount / 2;
    yhash_set<float> borders = GenerateMedianBorders(featureValues, bordersCount - halfBorders);

    // works better on rel approximation with quadratic loss
    float minValue = featureValues.front();
    float maxValue = featureValues.back();

    for (int i = 0; i < halfBorders; ++i) {
        float val = minValue + (i + 1) * (maxValue - minValue) / (halfBorders + 1);
        borders.insert(RegularBorder(val, featureValues));
    }

    return borders;
}

yhash_set<float> TUniformBinarizer::BestSplit(yvector<float>& featureValues,
                                              int bordersCount,
                                              bool isSorted) const {
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }

    if (featureValues.empty() || featureValues.front() == featureValues.back()) {
        return yhash_set<float>();
    }

    float minValue = featureValues.front();
    float maxValue = featureValues.back();

    yhash_set<float> borders;
    for (int i = 0; i < bordersCount; ++i) {
        borders.insert(minValue + (i + 1) * (maxValue - minValue) / (bordersCount + 1));
    }

    return borders;
}

}  // namespace NSplitSelection
