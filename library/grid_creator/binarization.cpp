#include "binarization.h"

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/queue.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/generic/ymath.h>

using NSplitSelection::IBinarizer;

namespace {
    class TMedianInBinBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                    int maxBordersCount,
                                    bool isSorted) const override;
    };

    class TMedianPlusUniformBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                    int maxBordersCount,
                                    bool isSorted) const override;
    };

    class TMinEntropyBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                    int maxBordersCount,
                                    bool isSorted) const override;
    };

    class TMaxSumLogBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                    int maxBordersCount,
                                    bool isSorted) const override;
    };

    // Works in O(binCount * log(n)) + O(nlogn) for sorting.
    class TMedianBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                    int maxBordersCount,
                                    bool isSorted) const override;
    };

    class TUniformBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                    int maxBordersCount,
                                    bool isSorted) const override;
    };
}

namespace NSplitSelection {
    THolder<IBinarizer> MakeBinarizer(const EBorderSelectionType type) {
        switch (type) {
            case EBorderSelectionType::UniformAndQuantiles:
                return MakeHolder<TMedianPlusUniformBinarizer>();
            case EBorderSelectionType::GreedyLogSum:
                return MakeHolder<TMedianInBinBinarizer>();
            case EBorderSelectionType::MinEntropy:
                return MakeHolder<TMinEntropyBinarizer>();
            case EBorderSelectionType::MaxLogSum:
                return MakeHolder<TMaxSumLogBinarizer>();
            case EBorderSelectionType::Median:
                return MakeHolder<TMedianBinarizer>();
            case EBorderSelectionType::Uniform:
                return MakeHolder<TUniformBinarizer>();
        }

        ythrow yexception() << "got invalid enum value: " << static_cast<int>(type);
    }
}

// TODO(yazevnul): fix memory use estimation
size_t CalcMemoryForFindBestSplit(int maxBordersCount, size_t docsCount, EBorderSelectionType type) {
    size_t bestSplitSize = docsCount * ((maxBordersCount + 2) * sizeof(size_t) + 4 * sizeof(double));
    if (type == EBorderSelectionType::MinEntropy || type == EBorderSelectionType::MaxLogSum) {
        bestSplitSize += docsCount * 3 * sizeof(float);
    }
    return bestSplitSize;
}

THashSet<float> BestSplit(TVector<float>& features,
                          int maxBordersCount,
                          EBorderSelectionType type,
                          bool nanValueIsInfty,
                          bool featuresAreSorted) {
    if (nanValueIsInfty) {
        features.erase(std::remove_if(features.begin(), features.end(), [](auto v) { return std::isnan(v); }), features.end());
    }

    if (features.empty()) {
        return {};
    }

    if (!featuresAreSorted) {
        Sort(features.begin(), features.end());
    }

    const auto binarizer = NSplitSelection::MakeBinarizer(type);
    return binarizer->BestSplit(features, maxBordersCount, true);
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

    enum class EPenaltyType {
        MinEntropy,
        MaxSumLog,
        W2
    };
}

static double Penalty(double weight, double, EPenaltyType type) {
    switch (type) {
        case EPenaltyType::MinEntropy:
            return weight * log(weight + 1e-8);
        case EPenaltyType::MaxSumLog:
            return -log(weight + 1e-8);
        case EPenaltyType::W2:
            return weight * weight;
        default:
            Y_VERIFY(false);
    }
}

template <typename TWeightType>
static void BestSplit(const TVector<TWeightType>& weights,
                      size_t maxBordersCount,
                      TVector<size_t>& thresholds,
                      EPenaltyType type,
                      ESF mode) {
    size_t bins = maxBordersCount + 1;
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
    TVector<TWeightType> sweights(weights);
    for (size_t i = 1; i < wsize; ++i) {
        sweights[i] += sweights[i - 1];
    }
    double expected = double(sweights[wsize - 1]) / bins;
    size_t dsize = ((mode == E_Base) || (mode == E_Old_Linear)) ? wsize : (wsize - bins + 1);
    TVector<size_t> bestSolutionsBuffer((bins - 2) * dsize);
    TVector<TArrayRef<size_t>> bestSolutions(bins - 2);
    for (size_t i = 0; i < bestSolutions.size(); ++i) {
        bestSolutions[i] = MakeArrayRef(bestSolutionsBuffer.data() + i * dsize, dsize);
    }

    TVector<double> current_error(dsize), prevError(dsize);
    for (size_t i = 0; i < dsize; ++i) {
        current_error[i] = Penalty(double(sweights[i]), expected, type);
    }
    // For 2 loops runs:
    TVector<size_t> bs1(dsize), bs2(dsize);
    TVector<double> e1(dsize), e2(dsize);

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
        } else if (mode == E_DaC) {
            typedef std::tuple<size_t, size_t, size_t, size_t> t4;
            TQueue<t4> qr;
            qr.push(std::make_tuple(0, dsize, 0, dsize));
            while (!qr.empty()) {
                size_t jbegin, jend, ibegin, iend;
                std::tie(jbegin, jend, ibegin, iend) = qr.front();
                qr.pop();
                if (jbegin >= jend) {
                    // empty box
                } else if (iend - ibegin == 1) {
                    // i is already fixed
                    for (size_t j = jbegin; j < jend; ++j) {
                        bestSolutions[l][j] = ibegin;
                        current_error[j] = prevError[ibegin] + Penalty(double(sweights[l + j + 1] - sweights[l + ibegin]), expected, type);
                    }
                } else {
                    size_t j = (jbegin + jend) / 2;
                    size_t bestIndex = ibegin;
                    double bestError = prevError[ibegin] + Penalty(double(sweights[l + j + 1] - sweights[l + ibegin]), expected, type);
                    size_t iend2 = Min(iend, j + 1);
                    for (size_t i = ibegin + 1; i < iend2; ++i) {
                        double newError = prevError[i] + Penalty(double(sweights[l + j + 1] - sweights[l + i]), expected, type);
                        if (newError <= bestError) {
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
static float RegularBorder(float border, const TVector<float>& sortedValues) {
    TVector<float>::const_iterator lowerBound = LowerBound(sortedValues.begin(), sortedValues.end(), border);

    if (lowerBound == sortedValues.end()) // binarizing to always false
        return Max(2.f * sortedValues.back(), sortedValues.back() + 1.f);

    if (lowerBound == sortedValues.begin()) // binarizing to always true
        return Min(.5f * sortedValues.front(), 2.f * sortedValues.front());

    float res = (lowerBound[0] + lowerBound[-1]) * .5f;
    if (res == lowerBound[0]) // wrong side rounding (should be very scarce)
        res = lowerBound[-1];

    return res;
}

static THashSet<float> BestSplit(const TVector<float>& values,
                                 const TVector<float>& weight,
                                 size_t maxBordersCount,
                                 EPenaltyType type) {
    // Positions after which threshold should be inserted.
    TVector<size_t> thresholds;
    thresholds.reserve(maxBordersCount);
    BestSplit(weight, maxBordersCount, thresholds, type, E_RLM2);

    THashSet<float> borders;
    borders.reserve(thresholds.size());
    for (auto t : thresholds) {
        if (t + 1 != values.size()) {
            borders.insert((values[t] + values[t + 1]) / 2);
        }
    }
    return borders;
}

static THashSet<float> SplitWithGuaranteedOptimum(
    TVector<float>& featureValues,
    int maxBordersCount,
    EPenaltyType type,
    bool isSorted) {
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }

    // TODO(yazevnul): rewrite this (well, actually, most of the code in this file) there is a lot
    // of place to save memory, here are first two that came to mind:
    // - reuse `featureValues`, no need to allocate `features`
    // - use `ui32` instead of `size_t` for indices

    TVector<float> features;
    features.reserve(featureValues.size());
    TVector<float> weights;
    weights.reserve(featureValues.size());
    for (auto f : featureValues) {
        if (features.empty() || features.back() != f) {
            features.push_back(f);
            weights.push_back(1);
        } else {
            weights.back()++;
        }
    }
    return BestSplit(features, weights, maxBordersCount, type);
}

static THashSet<float> GenerateMedianBorders(
    const TVector<float>& featureValues, int maxBordersCount) {
    THashSet<float> result;
    ui64 total = featureValues.size();
    if (total == 0 || featureValues.front() == featureValues.back()) {
        return result;
    }

    for (int i = 0; i < maxBordersCount; ++i) {
        ui64 i1 = (i + 1) * total / (maxBordersCount + 1);
        i1 = Min(i1, total - 1);
        float val1 = featureValues[i1];
        if (val1 != featureValues[0]) {
            result.insert(RegularBorder(val1, featureValues));
        }
    }
    return result;
}

THashSet<float> TMinEntropyBinarizer::BestSplit(
    TVector<float>& featureValues, int maxBordersCount, bool isSorted) const {
    return SplitWithGuaranteedOptimum(featureValues, maxBordersCount, EPenaltyType::MinEntropy, isSorted);
}

THashSet<float> TMaxSumLogBinarizer::BestSplit(
    TVector<float>& featureValues, int maxBordersCount, bool isSorted) const {
    return SplitWithGuaranteedOptimum(featureValues, maxBordersCount, EPenaltyType::MaxSumLog, isSorted);
}

THashSet<float> TMedianBinarizer::BestSplit(
    TVector<float>& featureValues, int maxBordersCount, bool isSorted) const {
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }
    return GenerateMedianBorders(featureValues, maxBordersCount);
}

THashSet<float> TMedianPlusUniformBinarizer::BestSplit(
    TVector<float>& featureValues, int maxBordersCount, bool isSorted) const {
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }

    if (featureValues.empty() || featureValues.front() == featureValues.back()) {
        return THashSet<float>();
    }

    int halfBorders = maxBordersCount / 2;
    THashSet<float> borders = GenerateMedianBorders(featureValues, maxBordersCount - halfBorders);

    // works better on rel approximation with quadratic loss
    float minValue = featureValues.front();
    float maxValue = featureValues.back();

    for (int i = 0; i < halfBorders; ++i) {
        float val = minValue + (i + 1) * (maxValue - minValue) / (halfBorders + 1);
        borders.insert(RegularBorder(val, featureValues));
    }

    return borders;
}

THashSet<float> TUniformBinarizer::BestSplit(TVector<float>& featureValues,
                                                int maxBordersCount,
                                                bool isSorted) const {
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }

    if (featureValues.empty() || featureValues.front() == featureValues.back()) {
        return THashSet<float>();
    }

    float minValue = featureValues.front();
    float maxValue = featureValues.back();

    THashSet<float> borders;
    for (int i = 0; i < maxBordersCount; ++i) {
        borders.insert(minValue + (i + 1) * (maxValue - minValue) / (maxBordersCount + 1));
    }

    return borders;
}

namespace {
    class TFeatureBin {
    private:
        ui32 BinStart;
        ui32 BinEnd;
        TVector<float>::const_iterator FeaturesStart;
        TVector<float>::const_iterator FeaturesEnd;

        ui32 BestSplit;
        double BestScore;

        inline void UpdateBestSplitProperties() {
            const int mid = (BinStart + BinEnd) / 2;
            float midValue = *(FeaturesStart + mid);

            ui32 lb = (ui32)(LowerBound(FeaturesStart + BinStart, FeaturesStart + mid, midValue) - FeaturesStart);
            ui32 up = (ui32)(UpperBound(FeaturesStart + mid, FeaturesStart + BinEnd, midValue) - FeaturesStart);

            const double scoreLeft = lb != BinStart ? log((double)(lb - BinStart)) + log((double)(BinEnd - lb)) : 0.0;
            const double scoreRight = up != BinEnd ? log((double)(up - BinStart)) + log((double)(BinEnd - up)) : 0.0;
            BestSplit = scoreLeft >= scoreRight ? lb : up;
            BestScore = BestSplit == lb ? scoreLeft : scoreRight;
        }

    public:
        TFeatureBin(ui32 binStart, ui32 binEnd, const TVector<float>::const_iterator featuresStart, const TVector<float>::const_iterator featuresEnd)
            : BinStart(binStart)
            , BinEnd(binEnd)
            , FeaturesStart(featuresStart)
            , FeaturesEnd(featuresEnd)
            , BestSplit(BinStart)
            , BestScore(0.0)
        {
            UpdateBestSplitProperties();
        }

        ui32 Size() const {
            return BinEnd - BinStart;
        }

        bool operator<(const TFeatureBin& bf) const {
            return Score() < bf.Score();
        }

        double Score() const {
            return BestScore;
        }

        TFeatureBin Split() {
            if (!CanSplit()) {
                throw yexception() << "Can't add new split";
            }
            TFeatureBin left = TFeatureBin(BinStart, BestSplit, FeaturesStart, FeaturesEnd);
            BinStart = BestSplit;
            UpdateBestSplitProperties();
            return left;
        }

        bool CanSplit() const {
            return (BinStart != BestSplit && BinEnd != BestSplit);
        }

        float Border() const {
            Y_ASSERT(BinStart < BinEnd);
            float borderValue = 0.5f * (*(FeaturesStart + BinEnd - 1));
            const float nextValue = ((FeaturesStart + BinEnd) < FeaturesEnd)
                                        ? (*(FeaturesStart + BinEnd))
                                        : (*(FeaturesStart + BinEnd - 1));
            borderValue += 0.5f * nextValue;
            return borderValue;
        }

        bool IsLast() const {
            return BinEnd == (FeaturesEnd - FeaturesStart);
        }
    };
}

THashSet<float> TMedianInBinBinarizer::BestSplit(TVector<float>& featureValues,
                                                 int maxBordersCount, bool isSorted) const {
    THashSet<float> borders;
    if (featureValues.size()) {
        if (!isSorted) {
            Sort(featureValues.begin(), featureValues.end());
        }

        std::priority_queue<TFeatureBin> splits;
        splits.push(TFeatureBin(0, (ui32) featureValues.size(), featureValues.begin(), featureValues.end()));

        while (splits.size() <= (ui32) maxBordersCount && splits.top().CanSplit()) {
            TFeatureBin top = splits.top();
            splits.pop();
            splits.push(top.Split());
            splits.push(top);
        }

        while (splits.size()) {
            if (!splits.top().IsLast())
                borders.insert(splits.top().Border());
            splits.pop();
        }
    }
    return borders;
}
