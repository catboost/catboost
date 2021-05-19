#include "binarization.h"

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/queue.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>
#include <util/generic/ymath.h>
#include <util/generic/serialized_enum.h>
#include <util/stream/labeled.h>

#include <algorithm>

using namespace NSplitSelection;
using namespace NSplitSelection::NImpl;

namespace {
    template <EPenaltyType PenaltyType>
    class TGreedyBinarizer: public IBinarizer {
    public:
        TQuantization BestSplit(
            TFeatureValues&& features,
            int maxBordersCount,
            TMaybe<float> quantizedDefaultBinFraction = Nothing(),
            const TMaybe<TVector<float>>& initialBorders = Nothing()) const override;
    };

    template <EPenaltyType PenaltyType>
    class TExactBinarizer: public IBinarizer {
    public:
        TQuantization BestSplit(
            TFeatureValues&& features,
            int maxBordersCount,
            TMaybe<float> quantizedDefaultBinFraction = Nothing(),
            const TMaybe<TVector<float>>& initialBorders = Nothing()) const override;
    };

    class TMedianPlusUniformBinarizer: public IBinarizer {
    public:
        TQuantization BestSplit(
            TFeatureValues&& features,
            int maxBordersCount,
            TMaybe<float> quantizedDefaultBinFraction = Nothing(),
            const TMaybe<TVector<float>>& initialBorders = Nothing()) const override;
    };

    // Works in O(binCount * log(n)) + O(n * log(n)) for sorting.
    // It's possible to implement O(n * log(binCount)) version.
    class TMedianBinarizer: public IBinarizer {
    public:
        TQuantization BestSplit(
            TFeatureValues&& features,
            int maxBordersCount,
            TMaybe<float> quantizedDefaultBinFraction = Nothing(),
            const TMaybe<TVector<float>>& initialBorders = Nothing()) const override;
    };

    class TUniformBinarizer: public IBinarizer {
    public:
        TQuantization BestSplit(
            TFeatureValues&& features,
            int maxBordersCount,
            TMaybe<float> quantizedDefaultBinFraction = Nothing(),
            const TMaybe<TVector<float>>& initialBorders = Nothing()) const override;
    };
}

namespace NSplitSelection {

    TQuantization BestSplit(
        TFeatureValues&& features,
        bool featureValuesMayContainNans,
        int maxBordersCount,
        EBorderSelectionType type,
        TMaybe<float> quantizedDefaultBinFraction,
        const TMaybe<TVector<float>>& initialBorders
    ) {
        if (features.DefaultValue && IsNan(features.DefaultValue->Value)) {
            if (featureValuesMayContainNans) {
                features.DefaultValue = Nothing();
            } else {
                throw (yexception() << "Unexpected Nan value.");
            }
        }

        auto firstNanPos = std::remove_if(features.Values.begin(), features.Values.end(), IsNan);
        if (firstNanPos != features.Values.end()) {
            if (featureValuesMayContainNans) {
                features.Values.erase(firstNanPos, features.Values.end());
            } else {
                throw (yexception() << "Unexpected Nan value.");
            }
        }

        if (quantizedDefaultBinFraction) {
            Y_ENSURE(
                (*quantizedDefaultBinFraction >= 0.0f) && (*quantizedDefaultBinFraction < 1.0f),
                LabeledOutput(quantizedDefaultBinFraction) << " is not in required [0, 1) bounds"
            );
        }

        if (features.Values.empty()) {
            return {};
        }

        const auto binarizer = MakeBinarizer(type);
        return binarizer->BestSplit(std::move(features), maxBordersCount, quantizedDefaultBinFraction, initialBorders);
    }

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
    bool featuresAreSorted,
    const TMaybe<TVector<float>>& initialBorders
) {
    const TQuantization quantization = NSplitSelection::BestSplit(
        TFeatureValues(std::move(features), featuresAreSorted),
        filterNans,
        maxBordersCount,
        type,
        /*quantizedDefaultBinFraction=*/Nothing(),
        initialBorders);

    return THashSet<float>(quantization.Borders.begin(), quantization.Borders.end());
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

namespace NSplitSelection {

    namespace NImpl {

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

    }
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
                                 const TMaybe<TVector<float>>& initialBorders,
                                 size_t maxBordersCount) {
    // Positions after which threshold should be inserted.
    TVector<size_t> thresholds;
    thresholds.reserve(maxBordersCount);
    BestSplit<float, type>(weight, maxBordersCount, thresholds, E_RLM2);

    THashSet<float> borders;
    borders.reserve(thresholds.size());
    for (auto t : thresholds) {
        if (t + 1 != values.size()) {
            if (initialBorders) {
                const auto possibleBorder = LowerBound(initialBorders->begin(), initialBorders->end(), values[t]);
                if (possibleBorder != initialBorders->end() && (*possibleBorder) <= values[t + 1]) {
                    borders.insert(*possibleBorder);
                    continue;
                }
            }
            borders.insert((values[t] + values[t + 1]) / 2);
        }
    }
    return borders;
}

// Border before element with value "border"
static float RegularBorder(float border, const TVector<float>& sortedValues, const TMaybe<TVector<float>>& initialBorders) {
    TVector<float>::const_iterator lowerBound = LowerBound(sortedValues.begin(), sortedValues.end(), border);

    if (lowerBound == sortedValues.end()) { // binarizing to always false
        if (initialBorders && !initialBorders->empty()) {
            if (sortedValues.back() < initialBorders->back()) {
                return initialBorders->back();
            }
        }
        return Max(2.f * sortedValues.back(), sortedValues.back() + 1.f);
    }

    if (lowerBound == sortedValues.begin()) {// binarizing to always true
        if (initialBorders && !initialBorders->empty()) {
            if ((*initialBorders)[0] <= sortedValues.back()) {
                return (*initialBorders)[0];
            }
        }
        return Min(.5f * sortedValues.front(), 2.f * sortedValues.front());
    }

    if (initialBorders) {
        const auto possibleBorder = UpperBound(initialBorders->begin(), initialBorders->end(), lowerBound[-1]);
        if (possibleBorder != initialBorders->end() && (*possibleBorder) <= lowerBound[0]) {
            return (*possibleBorder);
        }
    }

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

    /*
     * returns non-default weight for a single value and default weight otherwise
     * useful when dealing with default values
     */
    template <typename T>
    class TSingleValueWeightedIterator {
    public:
    private:
        size_t CurrentPosition;
        size_t SingleValuePosition;
        T DefaultWeight;
        T SingleValueWeight;
    public:
        TSingleValueWeightedIterator(T defaultWeight, size_t singleValuePosition, T singleValueWeight)
            : CurrentPosition(0)
            , SingleValuePosition(singleValuePosition)
            , DefaultWeight(defaultWeight)
            , SingleValueWeight(singleValueWeight)
        {}
        TSingleValueWeightedIterator& operator++() {
            ++CurrentPosition;
            return *this;
        }
        TSingleValueWeightedIterator operator++(int) {
            TSingleValueWeightedIterator result(*this);
            ++(*this);
            return result;
        }
        T operator*() const {
            return (CurrentPosition == SingleValuePosition) ? SingleValueWeight  : DefaultWeight;
        }
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
        TVector<float>&& featureValues,
        TWeightIteratorType weightsIterator,
        bool filterNans,
        bool isSorted,
        bool normalizeWeights = false,
        bool cumulativeWeights = false
    ) {
        TVector<float> uniqueFeatureValues; // featureValues' memory will be moved here
        TVector<float> uniqueValueWeights;
        size_t valueCount = 0;
        double totalWeight = 0.0f;
        if (isSorted) {
            auto srcIter = featureValues.begin();
            auto srcEndIter = featureValues.end();
            auto dstIter = featureValues.begin();

            for (; srcIter != srcEndIter; ++srcIter) {
                auto weight = *weightsIterator++;
                if (ShouldBeSkipped(*srcIter, weight, filterNans)) {
                    continue;
                }
                ++valueCount;
                totalWeight += weight;

                if (uniqueValueWeights.empty()) {
                    *dstIter++ = *srcIter;
                    uniqueValueWeights.push_back(weight);
                } else if (*(dstIter - 1) != *srcIter) {
                    *dstIter++ = *srcIter;
                    if (cumulativeWeights) {
                        weight += uniqueValueWeights.back();
                    }
                    uniqueValueWeights.push_back(weight);
                } else {
                    uniqueValueWeights.back() += weight;
                }
            }
            featureValues.resize(dstIter - featureValues.begin());
            featureValues.shrink_to_fit();
            uniqueFeatureValues = std::move(featureValues);
            if (normalizeWeights && uniqueValueWeights.size() > 0) {
                const double weightMultiplier = static_cast<double>(valueCount) / totalWeight;
                for (float& weight : uniqueValueWeights) {
                    weight *= weightMultiplier;
                }
            }
        } else {
            THashMap<float, float> groupedValues;
            THashMap<float, float>::insert_ctx insertCtx;

            auto srcIter = featureValues.begin();
            auto srcEndIter = featureValues.end();
            auto dstIter = featureValues.begin();

            for (; srcIter != srcEndIter; ++srcIter) {
                auto weight = *weightsIterator++;
                if (ShouldBeSkipped(*srcIter, weight, filterNans)) {
                    continue;
                }
                ++valueCount;
                totalWeight += weight;

                auto groupedValuesIter = groupedValues.find(*srcIter, insertCtx);
                if (groupedValuesIter == groupedValues.end()) {
                    groupedValues.emplace_direct(insertCtx, *srcIter, weight);
                    *dstIter++ = *srcIter;
                } else {
                    groupedValuesIter->second += weight;
                }
            }
            featureValues.resize(dstIter - featureValues.begin());
            featureValues.shrink_to_fit();
            uniqueFeatureValues = std::move(featureValues);
            Sort(uniqueFeatureValues.begin(), uniqueFeatureValues.end());

            uniqueValueWeights.reserve(uniqueFeatureValues.size());
            const double weightMultiplier = static_cast<double>(valueCount) / totalWeight;
            for (auto value : uniqueFeatureValues) {
                float weight = groupedValues.at(value);
                if (normalizeWeights) {
                    weight *= weightMultiplier;
                }
                if (cumulativeWeights && !uniqueValueWeights.empty()) {
                    weight += uniqueValueWeights.back();
                }
                uniqueValueWeights.push_back(weight);
            }
        }

        return {std::move(uniqueFeatureValues), std::move(uniqueValueWeights)};
    }
}


template <class TGetWeight>
static TQuantization SetQuantization(
    TConstArrayRef<float> sortedValues,

    // (sortedValuedStartIdx, sortedValuesEndIdx) -> weight for values range
    TGetWeight&& getWeight,
    float totalWeight,
    THashSet<float>&& bordersSet,
    TMaybe<float> quantizedDefaultBinFraction) {

    if (bordersSet.contains(-0.0f)) { // BestSplit might add negative zeros
        bordersSet.erase(-0.0f);
        bordersSet.insert(0.0f);
    }

    TQuantization result;
    result.Borders.assign(bordersSet.begin(), bordersSet.end());
    Sort(result.Borders);
    if (quantizedDefaultBinFraction) {
        ui32 currentBin = 0;

        auto getBin = [&] (float value) -> ui32 {
            ui32 bin = currentBin;
            while ((bin < result.Borders.size()) && (value >= result.Borders[bin])) {
                ++bin;
            }
            return bin;
        };

        size_t currentBinBeginIdx = 0;
        currentBin = getBin(sortedValues[0]);

        float maxBinWeight = 0.f;
        ui32 maxBinIdx;

        auto processCurrentBin = [&] (size_t currentBinEndIdx) {
            const float currentBinWeight = getWeight(currentBinBeginIdx, currentBinEndIdx);
            if (currentBinWeight > maxBinWeight) {
                maxBinWeight = currentBinWeight;
                maxBinIdx = currentBin;
            }
        };

        size_t i = 1;
        for (; i < sortedValues.size(); ++i) {
            auto bin = getBin(sortedValues[i]);
            if (bin != currentBin) {
                processCurrentBin(i);
                currentBin = bin;
                currentBinBeginIdx = i;
                if (currentBin == result.Borders.size()) {
                    i = sortedValues.size();
                    break;
                }
            }
        }
        processCurrentBin(i);

        float maxBinFraction = maxBinWeight / totalWeight;
        if (maxBinFraction > *quantizedDefaultBinFraction) {
            result.DefaultQuantizedBin = TDefaultQuantizedBin{maxBinIdx, maxBinFraction};
        }
    }
    return result;
}

static TQuantization SetQuantizationWithoutWeights(
    TConstArrayRef<float> sortedValues,
    THashSet<float>&& bordersSet,
    TMaybe<float> quantizedDefaultBinFraction) {

    return SetQuantization(
        sortedValues,
        [] (size_t begin, size_t end) -> float {
            return float(end - begin);
        },
        float(sortedValues.size()),
        std::move(bordersSet),
        quantizedDefaultBinFraction);
}


static TQuantization SetQuantizationWithCumulativeWeights(
    TConstArrayRef<float> sortedValues,
    TConstArrayRef<float> cumulativeWeights,
    THashSet<float>&& bordersSet,
    TMaybe<float> quantizedDefaultBinFraction) {

    return SetQuantization(
        sortedValues,
        [&] (size_t begin, size_t end) -> float {
            Y_ASSERT(end != 0);
            float weight = cumulativeWeights[end - 1];
            if (begin) {
                weight -= cumulativeWeights[begin - 1];
            }
            return weight;
        },
        cumulativeWeights.back(),
        std::move(bordersSet),
        quantizedDefaultBinFraction);
}

static TQuantization SetQuantizationWithMaybeSingleWeightedValue(
    TFeatureValues&& featureValues,
    TMaybe<size_t> maybeDefaultValueFirstPos,
    THashSet<float>&& bordersSet,
    TMaybe<float> quantizedDefaultBinFraction) {

    if (maybeDefaultValueFirstPos) {
        const size_t defaultValueFirstPos = *maybeDefaultValueFirstPos;
        const float defaultValueWeight = float(featureValues.DefaultValue->Count);
        return SetQuantization(
            featureValues.Values,
            [defaultValueFirstPos, defaultValueWeight] (size_t begin, size_t end) -> float {
                float result = float(end - begin + 1);
                if ((defaultValueFirstPos >= begin) && (defaultValueFirstPos < end)) {
                    result += (defaultValueWeight - 1.0f);
                }
                return result;
            },
            float(featureValues.Values.size() - 1) + defaultValueWeight,
            std::move(bordersSet),
            quantizedDefaultBinFraction);
    } else {
        return SetQuantizationWithoutWeights(
            featureValues.Values,
            std::move(bordersSet),
            quantizedDefaultBinFraction);
    }
}



template <EPenaltyType type>
static TQuantization SplitWithGuaranteedOptimum(
    TFeatureValues&& features,
    const TMaybe<TVector<float>>& initialBorders,
    int maxBordersCount,
    TMaybe<float> quantizedDefaultBinFraction) {

    auto [uniqueFeatureValues, uniqueValueWeights] = GroupAndSortValues(std::move(features), false);
    THashSet<float> bordersSet = BestSplit<type>(uniqueFeatureValues, uniqueValueWeights, initialBorders, maxBordersCount);

    if (quantizedDefaultBinFraction) {
        // reuse uniqueValueWeights for cumulative weights
        for (auto i : xrange<size_t>(1, uniqueValueWeights.size())) {
            uniqueValueWeights[i] += uniqueValueWeights[i - 1];
        }
    }

    return SetQuantizationWithCumulativeWeights(
        uniqueFeatureValues,
        uniqueValueWeights,
        std::move(bordersSet),
        quantizedDefaultBinFraction);

}

static THashSet<float> GenerateMedianBorders(
    const TVector<float>& featureValues, const TMaybe<TVector<float>>& initialBorders, int maxBordersCount) {
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
            result.insert(RegularBorder(val1, featureValues, initialBorders));
        }
    }
    return result;
}

static THashSet<float> GenerateMedianBordersWithDefaultValue(
    // must be sorted, featureValues must include it
    const TVector<float>& featureValues,
    const TMaybe<TVector<float>>& initialBorders,
    size_t defaultValueStartPos, // in featureValues
    const TDefaultValue<float> defaultValue,
    int maxBordersCount) {

    Y_ASSERT(featureValues[defaultValueStartPos] == defaultValue.Value);

    THashSet<float> result;
    if (maxBordersCount == 0 || featureValues.front() == featureValues.back()) {
        return result;
    }

    ui64 total = featureValues.size() + defaultValue.Count - 1;

    auto getValuesIndex = [=] (int borderIdx) -> ui64 {
        ui64 i1 = (borderIdx + 1) * total / (maxBordersCount + 1);
        return Min(i1, total - 1);
    };

    int i = 0;
    bool defaultValuePassed = false;
    do {
        if (defaultValuePassed) {
            float val1 = featureValues[getValuesIndex(i) - (defaultValue.Count - 1)];
            if (val1 != featureValues[0]) {
                result.insert(RegularBorder(val1, featureValues, initialBorders));
            }
            ++i;
        } else {
            ui64 i1 = getValuesIndex(i);
            float val1;
            if (i1 > featureValues.size()) { // default value at the end
                val1 = defaultValue.Value;
                i = maxBordersCount;
            } else {
                if (i1 >= defaultValueStartPos) {
                    defaultValuePassed = true;

                    int newI = (
                        CeilDiv((defaultValueStartPos + defaultValue.Count) * (maxBordersCount + 1), total) - 1);
                    if (newI != i) {
                        val1 = defaultValue.Value;
                        i = newI;
                    } else {
                        continue;
                    }
                } else {
                    val1 = featureValues[i1];
                    ++i;
                }
            }
            if (val1 != featureValues[0]) {
                result.insert(RegularBorder(val1, featureValues, initialBorders));
            }
        }
    } while (i < maxBordersCount);

    return result;
}

static THashSet<float> GenerateMedianBorders(
    // must be sorted, featureValues must include it
    const TVector<float>& featureValues,
    const TMaybe<TVector<float>>& initialBorders,
    const TMaybe<TDefaultValue<float>> defaultValue,
    TMaybe<size_t> defaultValueFirstPos, // pos in featureValues, defined only if defaultValue is defined
    int maxBordersCount) {

    if (defaultValue) {
        return GenerateMedianBordersWithDefaultValue(
            featureValues,
            initialBorders,
            *defaultValueFirstPos,
            *defaultValue,
            maxBordersCount);
    } else {
        return GenerateMedianBorders(featureValues, initialBorders, maxBordersCount);
    }
}



template <EPenaltyType PenaltyType>
TQuantization TExactBinarizer<PenaltyType>::BestSplit(
    TFeatureValues&& features,
    int maxBordersCount,
    TMaybe<float> quantizedDefaultBinFraction,
    const TMaybe<TVector<float>>& initialBorders) const {

    return SplitWithGuaranteedOptimum<PenaltyType>(
        std::move(features),
        initialBorders,
        maxBordersCount,
        quantizedDefaultBinFraction);
}


static void SortValuesAndInsertDefault(
    TFeatureValues& features,
    TMaybe<size_t>* defaultValueFirstPos) { // out parameter

    if (features.DefaultValue) {
        const float defaultValue = features.DefaultValue->Value;
        if (features.ValuesSorted) {
            auto defaultValueFirstPosIter = LowerBound(
                features.Values.begin(),
                features.Values.end(),
                defaultValue);

            *defaultValueFirstPos = defaultValueFirstPosIter - features.Values.begin();

            features.Values.insert(defaultValueFirstPosIter, defaultValue);
        } else {
            features.Values.push_back(defaultValue);
            Sort(features.Values);

            auto defaultValueFirstPosIter = LowerBound(
                features.Values.begin(),
                features.Values.end(),
                defaultValue);

            *defaultValueFirstPos = defaultValueFirstPosIter - features.Values.begin();
        }
    } else {
        if (!features.ValuesSorted) {
            Sort(features.Values);
        }
        *defaultValueFirstPos = Nothing();
    }
    features.ValuesSorted = true;
}


TQuantization TMedianBinarizer::BestSplit(
    TFeatureValues&& features,
    int maxBordersCount,
    TMaybe<float> quantizedDefaultBinFraction,
    const TMaybe<TVector<float>>& initialBorders) const {

    TMaybe<size_t> defaultValueFirstPos;
    SortValuesAndInsertDefault(features, &defaultValueFirstPos);

    THashSet<float> borders = GenerateMedianBorders(
        features.Values,
        initialBorders,
        features.DefaultValue,
        defaultValueFirstPos,
        maxBordersCount);

    return SetQuantizationWithMaybeSingleWeightedValue(
        std::move(features),
        defaultValueFirstPos,
        std::move(borders),
        quantizedDefaultBinFraction);
}

TQuantization TMedianPlusUniformBinarizer::BestSplit(
    TFeatureValues&& features,
    int maxBordersCount,
    TMaybe<float> quantizedDefaultBinFraction,
    const TMaybe<TVector<float>>& initialBorders) const {

    TMaybe<size_t> defaultValueFirstPos;
    SortValuesAndInsertDefault(features, &defaultValueFirstPos);

    if (features.Values.empty() || features.Values.front() == features.Values.back()) {
        return TQuantization();
    }

    int halfBorders = maxBordersCount / 2;
    THashSet<float> borders;
    borders = GenerateMedianBorders(
        features.Values,
        initialBorders,
        features.DefaultValue,
        defaultValueFirstPos,
        maxBordersCount - halfBorders);

    // works better on rel approximation with quadratic loss
    float minValue = features.Values.front();
    float maxValue = features.Values.back();

    for (int i = 0; i < halfBorders; ++i) {
        float val = minValue + (i + 1) * (maxValue - minValue) / (halfBorders + 1);
        borders.insert(RegularBorder(val, features.Values, initialBorders));
    }

    return SetQuantizationWithMaybeSingleWeightedValue(
        std::move(features),
        defaultValueFirstPos,
        std::move(borders),
        quantizedDefaultBinFraction);
}

TQuantization TUniformBinarizer::BestSplit(
    TFeatureValues&& features,
    int maxBordersCount,
    TMaybe<float> quantizedDefaultBinFraction,
    const TMaybe<TVector<float>>& initialBorders) const {

    if (features.Values.empty()) {
        return TQuantization();
    }

    auto [minIter, maxIter] = MinMaxElement(features.Values.begin(), features.Values.end());
    float minValue = *minIter;
    float maxValue = *maxIter;

    if (features.DefaultValue) {
        minValue = Min(minValue, features.DefaultValue->Value);
        maxValue = Max(maxValue, features.DefaultValue->Value);
    }

    if (minValue == maxValue) {
        return TQuantization();
    }

    TVector<float> featureValues;
    if (initialBorders) {
        featureValues = features.Values;
        if (features.DefaultValue) {
            featureValues.push_back(features.DefaultValue->Value);
        }
        Sort(featureValues.begin(), featureValues.end());
    }

    THashSet<float> borders;
    for (int i = 0; i < maxBordersCount; ++i) {
        double currentValue = minValue + (i + 1) * (maxValue - minValue) / (maxBordersCount + 1);
        if (initialBorders) {
            const auto separationPosition = LowerBound(featureValues.begin(), featureValues.end(), currentValue);
            const auto possibleBorder = UpperBound(initialBorders->begin(), initialBorders->end(), separationPosition[-1]);
            if (possibleBorder != initialBorders->end() && (*possibleBorder) <= separationPosition[0]) {
                currentValue = *possibleBorder;
            }
        }
        borders.insert(currentValue);
    }

    TMaybe<size_t> defaultValueFirstPos;
    if (quantizedDefaultBinFraction) {
        SortValuesAndInsertDefault(features, &defaultValueFirstPos);
    }

    return SetQuantizationWithMaybeSingleWeightedValue(
        std::move(features),
        defaultValueFirstPos,
        std::move(borders),
        quantizedDefaultBinFraction);
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

        float LeftBorder(const TMaybe<TVector<float>>& initialBorders) const {
            Y_ASSERT(BinStart < BinEnd);
            if (IsFirst()) {
                return *FeaturesStart;
            }
            if (initialBorders) {
                const auto possibleBorder = UpperBound(initialBorders->begin(), initialBorders->end(), (*(FeaturesStart + BinStart - 1)));
                if (possibleBorder != initialBorders->end() && (*possibleBorder) <= (*(FeaturesStart + BinStart))) {
                    return (*possibleBorder);
                }
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
    THashSet<float> GreedySplit(const TBinType& initialBin, const TMaybe<TVector<float>>& initialBorders, int maxBordersCount) {
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
                borders.insert(splits.top().LeftBorder(initialBorders));
            splits.pop();
        }
        return borders;
    }

    template<EPenaltyType penaltyType, class TWeightIteratorType>
    THashSet<float> BestWeightedSplitImpl(
        TVector<float>&& featureValues,
        TWeightIteratorType weightsIterator,
        int maxBordersCount,
        EOptimizationType optimizationType,
        bool filterNans,
        bool featuresAreSorted,
        bool normalizeWeights = true
    ) {
        auto[uniqueFeatureValues, uniqueValueWeights] = GroupAndSortWeighedValuesImpl(
            std::move(featureValues),
            weightsIterator,
            filterNans,
            featuresAreSorted,
            normalizeWeights,
            /*cumulativeWeights*/ optimizationType == EOptimizationType::Greedy);
        if (uniqueFeatureValues.empty()) {
            return {};
        }
        switch (optimizationType) {
            case EOptimizationType::Exact:
                return BestSplit<penaltyType>(uniqueFeatureValues, uniqueValueWeights, /*initialBorders=*/Nothing(), maxBordersCount);
            case EOptimizationType::Greedy: {
                TWeightedFeatureBin<float, penaltyType> initialBin(
                    0, uniqueFeatureValues.size(), uniqueFeatureValues.begin(), uniqueValueWeights.begin());
                return GreedySplit(initialBin, /*initialBorders=*/Nothing(), maxBordersCount);
            }
            default:
                throw (yexception() << "Invalid Optimization type.");
        }
    }
}


namespace NSplitSelection {

    namespace NImpl {

        template <EPenaltyType penaltyType>
        Y_NO_INLINE THashSet<float> BestWeightedSplit(
            TVector<float>&& featureValues,
            const TVector<float>& weights,
            int maxBordersCount,
            EOptimizationType optimizationType,
            bool filterNans,
            bool featuresAreSorted
        ) {
            Y_ENSURE(featureValues.size() == weights.size(), "weights and features should have equal size.");
            return BestWeightedSplitImpl<penaltyType>(
                std::move(featureValues),
                weights.begin(),
                maxBordersCount,
                optimizationType,
                filterNans,
                featuresAreSorted);
        }

        template<>
        Y_NO_INLINE THashSet<float> BestWeightedSplit<EPenaltyType::W2>(
            TVector<float>&& featureValues,
            const TVector<float>& weights,
            int maxBordersCount,
            EOptimizationType optimizationType,
            bool filterNans,
            bool featuresAreSorted
        ) {
            Y_ENSURE(featureValues.size() == weights.size(), "weights and features should have equal size.");
            return BestWeightedSplitImpl<EPenaltyType::W2>(
                std::move(featureValues),
                weights.begin(),
                maxBordersCount,
                optimizationType,
                filterNans,
                featuresAreSorted);
        }

        std::pair<TVector<float>, TVector<float>> GroupAndSortWeighedValues(
                TVector<float>&& featureValues,
                TVector<float>&& weights,
                bool filterNans,
                bool isSorted) {
            Y_ENSURE(featureValues.size() == weights.size());
            return GroupAndSortWeighedValuesImpl(
                std::move(featureValues),
                weights.begin(),
                filterNans,
                isSorted,
                true);
        }

        std::pair<TVector<float>, TVector<float>> GroupAndSortValues(
            TFeatureValues&& features,
            bool filterNans,
            bool cumulativeWeights) {

            if (features.DefaultValue) {
                features.Values.push_back(features.DefaultValue->Value);
                const size_t defaultValuePosition = features.Values.size() - 1;
                return GroupAndSortWeighedValuesImpl(
                    std::move(features.Values),
                    TSingleValueWeightedIterator<float>(
                        1.0f,
                        defaultValuePosition,
                        (float)features.DefaultValue->Count),
                    filterNans,
                    /*isSorted*/ false,
                    /*normalizeWeights*/ false,
                    cumulativeWeights);
            } else {
                return GroupAndSortWeighedValuesImpl(
                    std::move(features.Values),
                    TRepeatIterator<float>(1.0f),
                    filterNans,
                    features.ValuesSorted,
                    /*normalizeWeights*/ false,
                    cumulativeWeights);
            }
        }

    }

}


THashSet<float> BestWeightedSplit(
    TVector<float>&& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EBorderSelectionType borderSelectionType,
    bool filterNans,
    bool featuresAreSorted
) {
    switch (borderSelectionType) {
        case EBorderSelectionType::MinEntropy:
            return BestWeightedSplit<EPenaltyType::MinEntropy>(std::move(featureValues), weights, maxBordersCount,
                EOptimizationType::Exact, filterNans, featuresAreSorted);
        case EBorderSelectionType::MaxLogSum:
            return BestWeightedSplit<EPenaltyType::MaxSumLog>(std::move(featureValues), weights, maxBordersCount,
                EOptimizationType::Exact, filterNans, featuresAreSorted);
        case EBorderSelectionType ::GreedyLogSum:
            return BestWeightedSplit<EPenaltyType::MaxSumLog>(std::move(featureValues), weights, maxBordersCount,
                EOptimizationType::Greedy, filterNans, featuresAreSorted);
        case EBorderSelectionType ::GreedyMinEntropy:
            return BestWeightedSplit<EPenaltyType::MinEntropy>(std::move(featureValues), weights, maxBordersCount,
                EOptimizationType::Greedy, filterNans, featuresAreSorted);
        default:
            const auto borderSelectionTypeName = GetEnumNames<EBorderSelectionType>().at(borderSelectionType);
            throw (yexception() << "Weights are unsupported for " << borderSelectionTypeName <<
                                " border selection type.");
    }
}


template <EPenaltyType PenaltyType>
TQuantization TGreedyBinarizer<PenaltyType>::BestSplit(
    TFeatureValues&& features,
    int maxBordersCount,
    TMaybe<float> quantizedDefaultBinFraction,
    const TMaybe<TVector<float>>& initialBorders) const {

    if (features.Values.empty()) {
        return TQuantization();
    }
    if (features.DefaultValue) {
        auto [uniqueFeatureValues, uniqueValueWeights] = GroupAndSortValues(
            std::move(features),
            /*filterNans*/ false,
            /*cumulativeWeights*/ true);

        TWeightedFeatureBin<float, PenaltyType> initialBin(
            0,
            uniqueFeatureValues.size(),
            uniqueFeatureValues.begin(),
            uniqueValueWeights.begin());
        THashSet<float> bordersSet = GreedySplit(initialBin, initialBorders, maxBordersCount);
        return SetQuantizationWithCumulativeWeights(
            uniqueFeatureValues,
            uniqueValueWeights,
            std::move(bordersSet),
            quantizedDefaultBinFraction);
    } else {
        if (!features.ValuesSorted) {
            Sort(features.Values);
        }
        TFeatureBin<PenaltyType> initialBin(0, features.Values.size(), features.Values.cbegin());
        THashSet<float> bordersSet = GreedySplit(initialBin, initialBorders, maxBordersCount);
        return SetQuantizationWithoutWeights(
            features.Values,
            std::move(bordersSet),
            quantizedDefaultBinFraction);
    }
}

template <typename TKey, typename TValue>
size_t EstimateHashMapMemoryUsage(size_t hashMapSize) {
    size_t powTwoUpRoundedSize = (1ULL << static_cast<size_t>(log2(hashMapSize + 2) + 1));
    return 2 * sizeof(std::pair<TKey, TValue>) * powTwoUpRoundedSize;
}


static size_t CalcMemoryForFindBestSplitGreedyBinarizer(
    int maxBordersCount,
    size_t nonDefaultObjectCount,
    const TMaybe<TDefaultValue<float>>& defaultValue) {

    if (!defaultValue) {
        // 4 stands for priority_queue and THashSet memory overhead
        return 4 * maxBordersCount * (sizeof(TFeatureBin<EPenaltyType::MaxSumLog>) + sizeof(float));
    }

    const size_t featureValuesCount = nonDefaultObjectCount + 1;

    const size_t memoryForResizedFeaturesValues = featureValuesCount * sizeof(float);
    const size_t memoryForGroupedValues = EstimateHashMapMemoryUsage<float, float>(featureValuesCount);
    const size_t memoryForUniqueWeights = featureValuesCount * sizeof(float);

    // 4 stands for priority_queue and THashSet memory overhead
    const size_t memoryForGreedySplit
        = 4 * maxBordersCount * (
            sizeof(TWeightedFeatureBin<float, EPenaltyType::MaxSumLog>) + sizeof(float));

    return memoryForResizedFeaturesValues
        + memoryForUniqueWeights
        + Max(memoryForGroupedValues, memoryForGreedySplit);
}


/*
 * for
 *
 * template <typename TWeightType, EPenaltyType type>
 *   static void BestSplit(
 *      const TVector<TWeightType>& weights,
 *      size_t maxBordersCount,
 *      TVector<size_t>& thresholds,
 *      ESF mode
 *  ) {
 *
 * only common case: TWeightType = float, mode = E_RLM2
 */
static size_t CalcMemoryForInnerBestSplit(size_t weightsCount, size_t maxBordersCount) {
    const size_t bins = maxBordersCount + 1;
    const size_t wsize = weightsCount;

    if (wsize <= bins) {
        return 0;
    }

    const size_t dsize = wsize - bins + 1;

    return
        // sweights
        weightsCount * sizeof(float)

        // bestSolutionsBuffer
        + (bins - 2) * dsize * sizeof(size_t)

        // bestSolutions
        + (bins - 2) * sizeof(TArrayRef<size_t>)

        // current_error, prevError
        + 2 * dsize * sizeof(double)

        // bs1, bs2
        + 2 * dsize * sizeof(size_t)

        // e1, e2
        + 2 * dsize * sizeof(double);
}


static size_t CalcMemoryForFindBestSplitExactBinarizer(
    int maxBordersCount,
    size_t nonDefaultObjectCount,
    const TMaybe<TDefaultValue<float>>& defaultValue) {

    const size_t featureValuesCount = defaultValue ? (nonDefaultObjectCount + 1) : nonDefaultObjectCount;

    const size_t memoryForGroupedValues = EstimateHashMapMemoryUsage<float, float>(featureValuesCount);
    const size_t memoryForUniqueWeights = featureValuesCount * sizeof(float);

    // groupedValues + uniqueValueWeights
    const size_t memoryForGroupAndSortWeighedValuesImpl
        = memoryForGroupedValues + memoryForUniqueWeights;

    const size_t memoryForBorders = maxBordersCount * sizeof(float);

    const size_t memoryForOuterBestSplit
        = memoryForUniqueWeights

            // thresholds
            + maxBordersCount * sizeof(size_t)

            + Max(CalcMemoryForInnerBestSplit(featureValuesCount, maxBordersCount), memoryForBorders);

    return Max(memoryForGroupAndSortWeighedValuesImpl, memoryForOuterBestSplit);
}


namespace NSplitSelection {

    size_t CalcMemoryForFindBestSplit(
        int maxBordersCount,
        size_t nonDefaultObjectCount,
        const TMaybe<TDefaultValue<float>>& defaultValue,
        EBorderSelectionType type) {

        switch (type) {
            case EBorderSelectionType::Median:
            case EBorderSelectionType::UniformAndQuantiles:
                return maxBordersCount * sizeof(float)
                    + (defaultValue ? ((nonDefaultObjectCount + 1) * sizeof(float)) : 0);
            case EBorderSelectionType::Uniform:
                return maxBordersCount * sizeof(float);
            case EBorderSelectionType::GreedyLogSum:
            case EBorderSelectionType::GreedyMinEntropy:
                return CalcMemoryForFindBestSplitGreedyBinarizer(
                    maxBordersCount,
                    nonDefaultObjectCount,
                    defaultValue);

            case EBorderSelectionType::MinEntropy:
            case EBorderSelectionType::MaxLogSum:
                return CalcMemoryForFindBestSplitExactBinarizer(
                    maxBordersCount,
                    nonDefaultObjectCount,
                    defaultValue);
            default:
                Y_UNREACHABLE();
        }
    }

}
