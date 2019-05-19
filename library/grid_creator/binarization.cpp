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
#include <util/generic/serialized_enum.h>

using NSplitSelection::IBinarizer;

namespace {
    template <EPenaltyType PenaltyType>
    class TGreedyBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(
            TVector<float>& featureValues,
            int maxBordersCount,
            bool isSorted) const override;
    };

    template <EPenaltyType PenaltyType>
    class TExactBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(
            TVector<float>& featureValues,
            int maxBordersCount,
            bool isSorted) const override;
    };

    class TMedianPlusUniformBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(
            TVector<float>& featureValues,
            int maxBordersCount,
            bool isSorted) const override;
    };

    // Works in O(binCount * log(n)) + O(n * log(n)) for sorting.
    // It's possible to implement O(n * log(binCount)) version.
    class TMedianBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(
            TVector<float>& featureValues,
            int maxBordersCount,
            bool isSorted) const override;
    };

    class TUniformBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(
            TVector<float>& featureValues,
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
                return MakeHolder<TGreedyBinarizer<EPenaltyType::MaxSumLog>>();
            case EBorderSelectionType::GreedyMinEntropy:
                return MakeHolder<TGreedyBinarizer<EPenaltyType::MinEntropy>>();
            case EBorderSelectionType::MaxLogSum:
                return MakeHolder<TExactBinarizer<EPenaltyType::MaxSumLog>>();
            case EBorderSelectionType::MinEntropy:
                return MakeHolder<TExactBinarizer<EPenaltyType::MinEntropy>>();
            case EBorderSelectionType::Median:
                return MakeHolder<TMedianBinarizer>();
            case EBorderSelectionType::Uniform:
                return MakeHolder<TUniformBinarizer>();
        }

        ythrow yexception() << "got invalid enum value: " << static_cast<int>(type);
    }
}

THashSet<float> BestSplit(
    TVector<float>& features,
    int maxBordersCount,
    EBorderSelectionType type,
    bool filterNans,
    bool featuresAreSorted
) {
    auto firstNanPos = std::remove_if(features.begin(), features.end(), IsNan);
    if (firstNanPos != features.end()) {
        if (filterNans) {
            features.erase(firstNanPos, features.end());
        } else {
            throw (yexception() << "Unexpected Nan value.");
        }
    }

    if (features.empty()) {
        return {};
    }

    const auto binarizer = NSplitSelection::MakeBinarizer(type);
    return binarizer->BestSplit(features, maxBordersCount, featuresAreSorted);
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
}

template <>
double Penalty<EPenaltyType::MinEntropy>(double weight) {
    return weight * log(weight + 1e-8);
}
template <>
double Penalty<EPenaltyType::MaxSumLog>(double weight) {
    return -log(weight + 1e-8);
}

template <>
double Penalty<EPenaltyType::W2>(double weight) {
    return weight * weight;
}

template <typename TWeightType, EPenaltyType type>
static void BestSplit(
    const TVector<TWeightType>& weights,
    size_t maxBordersCount,
    TVector<size_t>& thresholds,
    ESF mode
) {
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
    size_t dsize = ((mode == E_Base) || (mode == E_Old_Linear)) ? wsize : (wsize - bins + 1);
    TVector<size_t> bestSolutionsBuffer((bins - 2) * dsize);
    TVector<TArrayRef<size_t>> bestSolutions(bins - 2);
    for (size_t i = 0; i < bestSolutions.size(); ++i) {
        bestSolutions[i] = MakeArrayRef(bestSolutionsBuffer.data() + i * dsize, dsize);
    }

    TVector<double> current_error(dsize), prevError(dsize);
    for (size_t i = 0; i < dsize; ++i) {
        current_error[i] = Penalty<type>(double(sweights[i]));
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
                double bestError = prevError[0] + Penalty<type>(double(sweights[j] - sweights[0]));
                for (size_t i = 1; i <= j; ++i) {
                    double newError = prevError[i] + Penalty<type>(double(sweights[j] - sweights[i]));
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
                double bestError = prevError[i] + Penalty<type>(double(sweights[j] - sweights[i]));
                for (++i; i < j; ++i) {
                    double newError = prevError[i] + Penalty<type>(double(sweights[j] - sweights[i]));
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
                double bestError = prevError[0] + Penalty<type>(double(sweights[l + j + 1] - sweights[l]));
                for (size_t i = 1; i <= j; ++i) {
                    double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
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
                double bestError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
                for (++i; i <= j; ++i) {
                    double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
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
                double bestError = prevError[left] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + left]));
                for (++left; left <= j; ++left) {
                    double newError = prevError[left] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + left]));
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
                double bestError = prevError[dsize - left - 1] + Penalty<type>(double(sweights[l + dsize - j] - sweights[l + dsize - left - 1]));
                for (++left; left < dsize; ++left) {
                    double newError = prevError[dsize - left - 1] + Penalty<type>(double(sweights[l + dsize - j] - sweights[l + dsize - left - 1]));
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
                        double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
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
                                double newError = prevError[i] + Penalty<type>(double(sweights[l + j1 + 1] - sweights[l + i]));
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
                                    double bestError = prevError[i] + Penalty<type>(double(sweights[l + ji + 1] - sweights[l + i]));
                                    for (++i; i <= ji; ++i) {
                                        double newError = prevError[i] + Penalty<type>(double(sweights[l + ji + 1] - sweights[l + i]));
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
                                    double newError = prevError[i] + Penalty<type>(double(sweights[l + j2 + 1] - sweights[l + i]));
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
                                    double bestError = prevError[dsize - i - 1] + Penalty<type>(double(sweights[l + dsize - ji] - sweights[l + dsize - i - 1]));
                                    for (++i; i < dsize; ++i) {
                                        double newError = prevError[dsize - i - 1] + Penalty<type>(double(sweights[l + dsize - ji] - sweights[l + dsize - i - 1]));
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
                    double bestError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
                    for (++i; i <= j; ++i) {
                        double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
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
                        double newError = prevError[dsize - i - 1] + Penalty<type>(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]));
                        if (newError + Eps < bestError) {
                            bestError = newError;
                            break;
                        }
                    }
                    if (i + 1 >= maxi) {
                        i = maxi;
                    } else {
                        for (++i; i + 1 < maxi; ++i) {
                            double newError = prevError[dsize - i - 1] + Penalty<type>(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]));
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
                                double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
                                if (newError + Eps < bestError) {
                                    bestError = newError;
                                    break;
                                }
                            }
                            if (i + 1 >= maxi) {
                                i = maxi;
                            } else {
                                for (++i; i + 1 < maxi; ++i) {
                                    double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
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
                                double newError = prevError[dsize - i - 1] + Penalty<type>(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]));
                                if (newError + Eps < bestError) {
                                    bestError = newError;
                                    break;
                                }
                            }
                            if (i + 1 >= maxi) {
                                i = maxi;
                            } else {
                                for (++i; i + 1 < maxi; ++i) {
                                    double newError = prevError[dsize - i - 1] + Penalty<type>(double(sweights[l + dsize - j] - sweights[l + dsize - i - 1]));
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
                        current_error[j] = prevError[ibegin] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + ibegin]));
                    }
                } else {
                    size_t j = (jbegin + jend) / 2;
                    size_t bestIndex = ibegin;
                    double bestError = prevError[ibegin] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + ibegin]));
                    size_t iend2 = Min(iend, j + 1);
                    for (size_t i = ibegin + 1; i < iend2; ++i) {
                        double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
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
            double bestError = prevError[0] + Penalty<type>(double(sweights[j] - sweights[0]));
            for (size_t i = 1; i <= j; ++i) {
                double newError = prevError[i] + Penalty<type>(double(sweights[j] - sweights[i]));
                if (newError < bestError) {
                    bestError = newError;
                    bestIndex = i;
                }
            }
        } else {
            double bestError = prevError[0] + Penalty<type>(double(sweights[l + j + 1] - sweights[l]));
            for (size_t i = 1; i <= j; ++i) {
                double newError = prevError[i] + Penalty<type>(double(sweights[l + j + 1] - sweights[l + i]));
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

template <EPenaltyType type>
static THashSet<float> BestSplit(const TVector<float>& values,
                                 const TVector<float>& weight,
                                 size_t maxBordersCount) {
    // Positions after which threshold should be inserted.
    TVector<size_t> thresholds;
    thresholds.reserve(maxBordersCount);
    BestSplit<float, type>(weight, maxBordersCount, thresholds, E_RLM2);

    THashSet<float> borders;
    borders.reserve(thresholds.size());
    for (auto t : thresholds) {
        if (t + 1 != values.size()) {
            borders.insert((values[t] + values[t + 1]) / 2);
        }
    }
    return borders;
}

// Border before element with value "border"
static float RegularBorder(float border, const TVector<float>& sortedValues) {
    TVector<float>::const_iterator lowerBound = LowerBound(sortedValues.begin(), sortedValues.end(), border);

    if (lowerBound == sortedValues.end()) // binarizing to always false
        return Max(2.f * sortedValues.back(), sortedValues.back() + 1.f);

    if (lowerBound == sortedValues.begin()) // binarizing to always true
        return Min(.5f * sortedValues.front(), 2.f * sortedValues.front());

    float res = (lowerBound[0] + lowerBound[-1]) * .5f;
    if (res == lowerBound[0]) { // wrong side rounding (should be very scarce)
        res = lowerBound[-1];
    }
    return res;
}


namespace {
    template <typename T>
    class TRepeatIterator {
    private:
        T Value;
    public:
        TRepeatIterator(T value) : Value(value) {}
        TRepeatIterator& operator++() { return *this; }
        TRepeatIterator& operator++(int) { return *this; }
        T operator*() const { return Value; }
    };

    template <typename TWeightType>
    static inline bool ShouldBeSkipped(float value, TWeightType weight, bool filterNans) {
        if (weight <= 0) {
            return true;
        }
        if (IsNan(value)) {
            Y_ENSURE(filterNans, "Nan value occurred");
            return true;
        }
        return false;
    }

    template <class TWeightIteratorType>
    std::pair<TVector<float>, TVector<float>> GroupAndSortWeighedValuesImpl(
        const TVector<float>& featureValues,
        TWeightIteratorType weightsIterator,
        bool filterNans,
        bool isSorted,
        bool normalizeWeights = false
    ) {
        TVector<float> uniqueFeatureValues; // is it worth to make a reserve?
        TVector<float> uniqueValueWeights;
        size_t valueCount = 0;
        double totalWeight = 0.0f;
        if (isSorted) {
            for (auto value : featureValues) {
                auto weight = *weightsIterator++;
                if (ShouldBeSkipped(value, weight, filterNans)) {
                    continue;
                }
                ++valueCount;
                totalWeight += weight;
                if (uniqueFeatureValues.empty() || uniqueFeatureValues.back() != value) {
                    uniqueFeatureValues.push_back(value);
                    uniqueValueWeights.push_back(weight);
                } else {
                    uniqueValueWeights.back() += weight;
                }
            }
        } else {
            THashMap<float, float> groupedValues;
            for (auto value : featureValues) {
                auto weight = *weightsIterator++;
                if (ShouldBeSkipped(value, weight, filterNans)) {
                    continue;
                }
                ++valueCount;
                totalWeight += weight;
                if (groupedValues.contains(value)) {
                    groupedValues.at(value) += weight;
                } else {
                    groupedValues.emplace(value, weight);
                    uniqueFeatureValues.push_back(value);
                }
            }
            Sort(uniqueFeatureValues.begin(), uniqueFeatureValues.end());
            uniqueValueWeights.reserve(uniqueFeatureValues.size());
            for (auto value : uniqueFeatureValues) {
                uniqueValueWeights.push_back(groupedValues.at(value));
            }
        }
        if (normalizeWeights && valueCount > 0) {
            const double weightMultiplier = static_cast<double>(valueCount) / totalWeight;
            for (float& weight : uniqueValueWeights) {
                weight *= weightMultiplier;
            }
        }
        return {std::move(uniqueFeatureValues), std::move(uniqueValueWeights)};
    }
}

std::pair<TVector<float>, TVector<float>> GroupAndSortWeighedValues(
        const TVector<float>& featureValues,
        const TVector<float>& weights,
        bool filterNans, bool isSorted) {
    Y_ENSURE(featureValues.size() == weights.size());
    return GroupAndSortWeighedValuesImpl(featureValues, weights.begin(), filterNans, isSorted, true);
}

std::pair<TVector<float>, TVector<float>> GroupAndSortValues(
        const TVector<float>& featureValues, bool filterNans, bool isSorted) {
    return GroupAndSortWeighedValuesImpl(
        featureValues, TRepeatIterator<float>(1.0f), filterNans, isSorted);
}

template <EPenaltyType type>
static THashSet<float> SplitWithGuaranteedOptimum(TVector<float>& featureValues,
                                                  int maxBordersCount,
                                                  bool isSorted) {
    const auto [uniqueFeatureValues, uniqueValueWeights] = GroupAndSortValues(
        featureValues, false, isSorted);
    return BestSplit<type>(uniqueFeatureValues, uniqueValueWeights, maxBordersCount);
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

template <EPenaltyType PenaltyType>
THashSet<float> TExactBinarizer<PenaltyType>::BestSplit(
    TVector<float>& featureValues, int maxBordersCount, bool isSorted) const {
    return SplitWithGuaranteedOptimum<PenaltyType>(featureValues, maxBordersCount, isSorted);
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

THashSet<float> TUniformBinarizer::BestSplit(
    TVector<float>& featureValues,
    int maxBordersCount,
    bool isSorted
) const {
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
    class IFeatureBin {
    protected:
        ui32 BinStart;
        ui32 BinEnd;
        TVector<float>::const_iterator FeaturesStart;

        ui32 BestSplit;
        double BestScore;

    public:
        using TFeatureItreator = typename TVector<float>::const_iterator;

        IFeatureBin(ui32 binStart, ui32 binEnd, TFeatureItreator featuresStart)
        : BinStart(binStart)
        , BinEnd(binEnd)
        , FeaturesStart(featuresStart)
        , BestSplit(binStart)
        , BestScore(0.0) {
            Y_ASSERT(BinStart < BinEnd);
        }

        ui32 Size() const {
            return BinEnd - BinStart;
        }

        bool operator<(const IFeatureBin& bf) const {
            return Score() < bf.Score();
        }

        double Score() const {
            return BestScore;
        }

        bool CanSplit() const {
            return (BinStart != BestSplit && BinEnd != BestSplit);
        }

        float LeftBorder() const {
            Y_ASSERT(BinStart < BinEnd);
            if (IsFirst()) {
                return *FeaturesStart;
            }
            float borderValue = 0.5f * (*(FeaturesStart + BinStart - 1));
            borderValue += 0.5f * (*(FeaturesStart + BinStart));
            return borderValue;
        }

        bool IsFirst() const {
            return BinStart == 0;
        }
    };

    template<EPenaltyType PenaltyType>
    class TFeatureBin : public IFeatureBin {
    public:
        TFeatureBin(ui32 binStart, ui32 binEnd, TFeatureItreator featuresStart)
            : IFeatureBin(binStart, binEnd, featuresStart)
        {
            UpdateBestSplitProperties();
        }

        TFeatureBin Split() {
            if (!CanSplit()) {
                throw yexception() << "Can't add new split";
            }
            TFeatureBin left = TFeatureBin(BinStart, BestSplit, FeaturesStart);
            BinStart = BestSplit;
            UpdateBestSplitProperties();
            return left;
        }

    private:
        double CalcSplitScore(ui32 splitPos) const {
            Y_ASSERT(splitPos >= BinStart && splitPos <= BinEnd);
            if (splitPos == BinStart || splitPos == BinEnd) {
                return -std::numeric_limits<double>::infinity();
            }
            const double leftPartScore = -Penalty<PenaltyType>(static_cast<double>(splitPos - BinStart));
            const double rightPartScore = -Penalty<PenaltyType>(static_cast<double>(BinEnd - splitPos));
            const double currBinScore = -Penalty<PenaltyType>(static_cast<double>(BinEnd - BinStart));
            return leftPartScore + rightPartScore - currBinScore;
        }

        inline void UpdateBestSplitProperties() {
            const int mid = BinStart + (BinEnd - BinStart) / 2;
            float midValue = *(FeaturesStart + mid);

            const ui32 lb = static_cast<ui32>(
                LowerBound(FeaturesStart + BinStart, FeaturesStart + mid, midValue) - FeaturesStart
            );
            const ui32 ub = static_cast<ui32>(
                UpperBound(FeaturesStart + mid, FeaturesStart + BinEnd, midValue) - FeaturesStart
            );

            const double scoreLeft = CalcSplitScore(lb);
            const double scoreRight = CalcSplitScore(ub);
            BestSplit = scoreLeft >= scoreRight ? lb : ub;
            BestScore = BestSplit == lb ? scoreLeft : scoreRight;
        }
    };

    template<typename TWeightType, EPenaltyType penaltyType>
    class TWeightedFeatureBin : public IFeatureBin {
    public:
        using TWeightsIterator = typename TVector<TWeightType>::const_iterator;

        TWeightedFeatureBin(
            ui32 binStart,
            ui32 binEnd,
            TFeatureItreator featuresStart,
            TWeightsIterator cumulativeWeightsStart
        )
            : IFeatureBin(binStart, binEnd, featuresStart) , CumulativeWeightsStart(cumulativeWeightsStart)
        {
            UpdateBestSplitProperties();
        }

        TWeightedFeatureBin Split() {
            if (!CanSplit()) {
                throw yexception() << "Can't add new split";
            }
            TWeightedFeatureBin left(BinStart, BestSplit, FeaturesStart, CumulativeWeightsStart);
            BinStart = BestSplit;
            UpdateBestSplitProperties();
            return left;
        }

    private:
        double CalcSplitScore(ui32 splitPos) const {
            Y_ASSERT(splitPos >= BinStart && splitPos <= BinEnd);
            if (splitPos == BinStart || splitPos == BinEnd) {
                return -std::numeric_limits<double>::infinity();
            }
            const TWeightType leftBinsWeight = (
                    BinStart == 0 ? (TWeightType) 0 : *(CumulativeWeightsStart + BinStart - 1)
            );
            const TWeightType leftPartWeight = *(CumulativeWeightsStart + splitPos - 1) - leftBinsWeight;
            const TWeightType rightPartWeight = (
                *(CumulativeWeightsStart + BinEnd - 1) - *(CumulativeWeightsStart + splitPos - 1)
            );
            const double currBinScore = -Penalty<penaltyType>(leftPartWeight + rightPartWeight);
            const double newBinsScore = -(
                Penalty<penaltyType>(leftPartWeight) + Penalty<penaltyType>(rightPartWeight)
            );
            return newBinsScore - currBinScore;
        }

        void UpdateBestSplitProperties() {
            Y_ASSERT(BinStart < BinEnd);
            const TWeightType leftBinsWeight = (
                BinStart == 0 ? (TWeightType) 0 : *(CumulativeWeightsStart + BinStart - 1)
            );
            const double midCumulativeWeight =
                0.5 * (leftBinsWeight + *(CumulativeWeightsStart + BinEnd - 1));

            const ui32 lb = LowerBound(
                CumulativeWeightsStart + BinStart,
                CumulativeWeightsStart + BinEnd,
                midCumulativeWeight) - CumulativeWeightsStart;
            Y_ASSERT(lb < BinEnd); // weights are (strictly) positive hence ub > BinStart
            const ui32 ub = lb + 1;

            const double scoreLeft = CalcSplitScore(lb);
            const double scoreRight = CalcSplitScore(ub);

            BestSplit = scoreLeft >= scoreRight ? lb : ub;
            BestScore = BestSplit == lb ? scoreLeft : scoreRight;
        }

    private:
        TWeightsIterator CumulativeWeightsStart;
    };

    template<class TBinType>
    THashSet<float> GreedySplit(const TBinType& initialBin, int maxBordersCount) {
        std::priority_queue<TBinType> splits;
        splits.push(initialBin);

        while (splits.size() <= (ui32) maxBordersCount && splits.top().CanSplit()) {
            auto top = splits.top();
            splits.pop();
            auto left = top.Split();
            splits.push(left);
            splits.push(top);
        }

        THashSet<float> borders;
        borders.reserve(splits.size() - 1);
        while (!splits.empty()) {
            if (!splits.top().IsFirst())
                borders.insert(splits.top().LeftBorder());
            splits.pop();
        }
        return borders;
    }

    template<EPenaltyType penaltyType, class TWeightIteratorType>
    THashSet<float> BestWeightedSplitImpl(
        const TVector<float>& featureValues,
        TWeightIteratorType weightsIterator,
        int maxBordersCount,
        EOptimizationType optimizationType,
        bool filterNans,
        bool featuresAreSorted,
        bool normalizeWeights = true
    ) {
        auto[uniqueFeatureValues, uniqueValueWeights] = GroupAndSortWeighedValuesImpl(
            featureValues, weightsIterator, filterNans, featuresAreSorted, normalizeWeights);
        if (uniqueFeatureValues.empty()) {
            return {};
        }
        switch (optimizationType) {
            case EOptimizationType::Exact:
                return BestSplit<penaltyType>(uniqueFeatureValues, uniqueValueWeights, maxBordersCount);
            case EOptimizationType::Greedy: {
                for (size_t i = 0; i + 1 < uniqueValueWeights.size(); ++i) {
                    uniqueValueWeights[i + 1] += uniqueValueWeights[i];
                }
                TWeightedFeatureBin<float, penaltyType> initialBin(
                    0, uniqueFeatureValues.size(), uniqueFeatureValues.begin(), uniqueValueWeights.begin());
                return GreedySplit(initialBin, maxBordersCount);
            }
            default:
                throw (yexception() << "Invalid Optimization type.");
        }
    }
}

template <EPenaltyType penaltyType>
Y_NO_INLINE THashSet<float> BestWeightedSplit(
    const TVector<float>& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EOptimizationType optimizationType,
    bool filterNans,
    bool featuresAreSorted
) {
    Y_ENSURE(featureValues.size() == weights.size(), "weights and features should have equal size.");
    return BestWeightedSplitImpl<penaltyType>(
        featureValues, weights.begin(), maxBordersCount, optimizationType, filterNans, featuresAreSorted);
}

template<>
Y_NO_INLINE THashSet<float> BestWeightedSplit<EPenaltyType::W2>(
    const TVector<float>& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EOptimizationType optimizationType,
    bool filterNans,
    bool featuresAreSorted
) {
    Y_ENSURE(featureValues.size() == weights.size(), "weights and features should have equal size.");
    return BestWeightedSplitImpl<EPenaltyType::W2>(
        featureValues, weights.begin(), maxBordersCount, optimizationType, filterNans, featuresAreSorted);
}

THashSet<float> BestWeightedSplit(
    const TVector<float>& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EBorderSelectionType borderSelectionType,
    bool filterNans,
    bool featuresAreSorted
) {
    switch (borderSelectionType) {
        case EBorderSelectionType::MinEntropy:
            return BestWeightedSplit<EPenaltyType::MinEntropy>(featureValues, weights, maxBordersCount,
                EOptimizationType::Exact, filterNans, featuresAreSorted);
        case EBorderSelectionType::MaxLogSum:
            return BestWeightedSplit<EPenaltyType::MaxSumLog>(featureValues, weights, maxBordersCount,
                EOptimizationType::Exact, filterNans, featuresAreSorted);
        case EBorderSelectionType ::GreedyLogSum:
            return BestWeightedSplit<EPenaltyType::MaxSumLog>(featureValues, weights, maxBordersCount,
                EOptimizationType::Greedy, filterNans, featuresAreSorted);
        case EBorderSelectionType ::GreedyMinEntropy:
            return BestWeightedSplit<EPenaltyType::MinEntropy>(featureValues, weights, maxBordersCount,
                EOptimizationType::Greedy, filterNans, featuresAreSorted);
        default:
            const auto borderSelectionTypeName = GetEnumNames<EBorderSelectionType>().at(borderSelectionType);
            throw (yexception() << "Weights are unsupported for " << borderSelectionTypeName <<
                                " border selection type.");
    }
}

template <EPenaltyType PenaltyType>
THashSet<float> TGreedyBinarizer<PenaltyType>::BestSplit(
    TVector<float>& featureValues,
    int maxBordersCount,
    bool isSorted
) const {
    if (featureValues.empty()) {
        return {};
    }
    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }
    TFeatureBin<PenaltyType> initialBin(0, featureValues.size(), featureValues.cbegin());
    return GreedySplit(initialBin, maxBordersCount);
}

template <typename TKey, typename TValue>
size_t EstimateHashMapMemoryUsage(size_t hashMapSize) {
    size_t powTwoUpRoundedSize = (1 << static_cast<size_t>(log2(hashMapSize + 2) + 1));
    return 2 * sizeof(std::pair<TKey, TValue>) * powTwoUpRoundedSize;
}

size_t CalcMemoryForFindBestSplit(int maxBordersCount, size_t docsCount, EBorderSelectionType type) {
    switch (type) {
        case EBorderSelectionType::Median:
        case EBorderSelectionType::UniformAndQuantiles:
        case EBorderSelectionType::Uniform:
            return maxBordersCount * sizeof(float);
        case EBorderSelectionType::GreedyLogSum:
        case EBorderSelectionType::GreedyMinEntropy:
            // 4 stands for priority_queue and THashSet memory overhead
            return 4 * maxBordersCount * (sizeof(TFeatureBin<EPenaltyType::MaxSumLog>) + sizeof(float));
        case EBorderSelectionType::MinEntropy:
        case EBorderSelectionType::MaxLogSum:
            return docsCount * ((maxBordersCount + 2) * sizeof(size_t) + 4 * sizeof(double)) + docsCount * 3 * sizeof(float);
        default:
            Y_UNREACHABLE();
    }
}
