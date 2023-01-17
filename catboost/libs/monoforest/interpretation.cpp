#include "interpretation.h"

#include <util/generic/vector.h>


namespace NMonoForest {
    static TVector<double> CalcAveragePolynomValue(const TPolynom& polynom) {
        TVector<double> value(polynom.Dimension());
        double totalWeight = 0;
        for (const auto& [structure, stat] : polynom.MonomsEnsemble) {
            for (auto dim : xrange(polynom.Dimension())) {
                value[dim] += stat.Value[dim] * stat.Weight;
            }
            if (structure.Splits.empty()) {
                totalWeight = stat.Weight;
            }
        }
        CB_ENSURE(totalWeight > 0);
        for (auto dim : xrange(polynom.Dimension())) {
            value[dim] /= totalWeight;
        }
        return value;
    }

    static double GetTotalWeight(const TPolynom& polynom) {
        for (const auto& [structure, stat] : polynom.MonomsEnsemble) {
            if (structure.Splits.empty()) {
                return stat.Weight;
            }
        }
        CB_ENSURE(false, "Failed to get total weight");
    }

    static TVector<TVector<double>> GetBorderWeights(const TPolynom& polynom, const IGrid& grid) {
        TVector<TVector<double>> weightForBorder(grid.FeatureCount());
        for (auto featureIdx : xrange(grid.FeatureCount())) {
            weightForBorder[featureIdx].resize(grid.BorderCount(featureIdx));
        }
        for (const auto& [structure, stat] : polynom.MonomsEnsemble) {
            if (structure.Splits.size() == 1) {
                const auto& split = structure.Splits[0];
                weightForBorder[split.FeatureId][split.BinIdx] = stat.Weight;
            }
        }
        return weightForBorder;
    }

    static TVector<TVector<TVector<double>>> CalcValueChangeForBorders(
        const TPolynom& polynom,
        const IGrid& grid,
        const TVector<TVector<double>>& weightForBorder)
    {
        const auto dimension = polynom.Dimension();
        TVector<TVector<TVector<double>>> valueChangeForBorder(grid.FeatureCount());
        for (auto featureIdx : xrange(grid.FeatureCount())) {
            valueChangeForBorder[featureIdx].resize(grid.BorderCount(featureIdx), TVector<double>(dimension));
        }
        for (const auto& [structure, stat] : polynom.MonomsEnsemble) {
            for (const auto& split : structure.Splits) {
                for (auto dim : xrange(dimension)) {
                    valueChangeForBorder[split.FeatureId][split.BinIdx][dim] += stat.Value[dim] * stat.Weight;
                }
            }
        }
        for (auto featureIdx : xrange(grid.FeatureCount())) {
            for (auto borderIdx : xrange(grid.BorderCount(featureIdx))) {
                if (weightForBorder[featureIdx][borderIdx] > 0) {
                    for (auto dim : xrange(dimension)) {
                        valueChangeForBorder[featureIdx][borderIdx][dim] /= weightForBorder[featureIdx][borderIdx];
                    }
                }
            }
        }
        return valueChangeForBorder;
    }

    static TFeatureExplanation ExplainFeature(
        int featureIdx,
        const IGrid& grid,
        const TVector<double>& weightForBorder,
        const TVector<TVector<double>>& valueChangeForBorder,
        const TVector<double>& averagePolynomValue,
        double totalWeight,
        int dimension)
    {
        TVector<double> bias = averagePolynomValue;
        for (auto dim : xrange(dimension)) {
            for (auto borderIdx : xrange(weightForBorder.size())) {
                bias[dim] -= valueChangeForBorder[borderIdx][dim] * weightForBorder[borderIdx] / totalWeight;
            }
        }
        TFeatureExplanation explanation;
        explanation.FeatureIdx = featureIdx;
        explanation.FeatureType = grid.FeatureType(featureIdx);
        explanation.ExpectedBias = bias;

        for (auto borderIdx : xrange(grid.BorderCount(featureIdx))) {
            if (weightForBorder[borderIdx] > 0) {
                TBorderExplanation borderExplanation;
                borderExplanation.Border = grid.Border(featureIdx, borderIdx);
                borderExplanation.ProbabilityToSatisfy = weightForBorder[borderIdx] / totalWeight;
                borderExplanation.ExpectedValueChange = valueChangeForBorder[borderIdx];
                explanation.BordersExplanations.emplace_back(std::move(borderExplanation));
            }
        }
        return explanation;
    }

    TVector<TFeatureExplanation> ExplainFeatures(const TPolynom& polynom, const IGrid& grid) {
        const auto dimension = polynom.Dimension();
        const double totalWeight = GetTotalWeight(polynom);
        const auto weightForBorder = GetBorderWeights(polynom, grid);
        const auto valueChangeForBorder = CalcValueChangeForBorders(polynom, grid, weightForBorder);
        const auto averagePolynomValue = CalcAveragePolynomValue(polynom);

        TVector<TFeatureExplanation> featuresExplanations(grid.FeatureCount());
        for (auto featureIdx : xrange(grid.FeatureCount())) {
            featuresExplanations[featureIdx] = ExplainFeature(featureIdx, grid, weightForBorder[featureIdx],
                valueChangeForBorder[featureIdx], averagePolynomValue, totalWeight, dimension);
        }

        return featuresExplanations;
    }
}
