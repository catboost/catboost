# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.libs.monoforest._monoforest cimport *


cdef extern from "catboost/python-package/catboost/monoforest_helpers.h" namespace "NMonoForest":
    cdef cppclass THumanReadableSplit:
        int FeatureIdx
        EBinSplitType SplitType
        float Border

    cdef cppclass THumanReadableMonom:
        TVector[THumanReadableSplit] Splits
        TVector[double] Value
        double Weight


cdef extern from "catboost/python-package/catboost/monoforest_helpers.h" namespace "NMonoForest":
    TString ConvertFullModelToPolynomString(const TFullModel& fullModel) except +ProcessException
    TVector[THumanReadableMonom] ConvertFullModelToPolynom(const TFullModel& fullModel) except +ProcessException
    TVector[TFeatureExplanation] ExplainFeatures(const TFullModel& fullModel) except +ProcessException


class Split:
    def __init__(self, feature_idx, split_type, border):
        self.feature_idx = feature_idx
        self.split_type = split_type
        self.border = border

    def __str__(self):
        type_sign = ">" if self.split_type == "TakeGreater" else "="
        return "[F{} {} {}]".format(self.feature_idx, type_sign, self.border)

    def __repr__(self):
        return self.__str__()


class Monom:
    def __init__(self, splits, value, weight):
        self.splits = splits
        self.value = value
        self.weight = weight

    def __str__(self):
        value_str = "({})".format(", ".join(map(str, self.value)))
        split_str = "".join(map(str, self.splits))
        if not split_str:
            return value_str
        return value_str + " * " + split_str

    def __repr__(self):
        return self.__str__() + " <weight={}>".format(self.weight)


class BorderExplanation:
    def __init__(self, border, probability, expected_value_change):
        self.border = border
        self.probability = probability
        self.expected_value_change = expected_value_change

    def __repr__(self):
        return "(border={}, probability={}, value_change={})".format(self.border, self.probability, self.expected_value_change)


class FeatureExplanation:
    def __init__(self, feature, type, expected_bias, borders_explanations):
        self.feature = feature
        self.type = type
        self.expected_bias = expected_bias
        self.borders_explanations = borders_explanations

    def __repr__(self):
        return "feature={} ({}), bias={}, borders={}".format(self.feature, self.type, self.expected_bias, self.borders_explanations)

    def dimension(self):
        return len(self.expected_bias)

    def calc_strength(self, dim=None):
        if dim is not None:
            strength = 0
            for border_expl in self.borders_explanations:
                strength += border_expl.expected_value_change[dim] * border_expl.probability
            return strength
        else:
            strength = []
            for dim in xrange(len(self.expected_bias)):
                strength.append(self.calc_strength(dim))
            return strength

    def _calc_pdp_values(self, dim):
        values = []
        if self.type == "Float":
            values.append(self.expected_bias[dim])
            for border_expl in self.borders_explanations:
                values.append(values[len(values) - 1] + border_expl.expected_value_change[dim])
        else:
            for border_expl in self.borders_explanations:
                values.append(self.expected_bias[dim] + border_expl.expected_value_change[dim])
        return values

    def calc_pdp(self, dim=None):
        borders = [border_expl.border for border_expl in self.borders_explanations]
        if dim is not None:
            values = self._calc_pdp_values(dim)
        else:
            values = [self._calc_pdp_values(dim) for dim in xrange(len(self.expected_bias))]
        return borders, values


cpdef to_polynom(model):
    cdef TVector[THumanReadableMonom] monoms = ConvertFullModelToPolynom(dereference((<_CatBoost>model).__model))
    python_monoms = []
    for monom in monoms:
        python_splits = []
        for split in monom.Splits:
            type = "TakeGreater" if split.SplitType == EBinSplitType_TakeGreater else "TakeEqual"
            python_splits.append(Split(split.FeatureIdx, type, split.Border))
        value = []
        for i in xrange(monom.Value.size()):
            value.append(monom.Value[i])
        python_monoms.append(Monom(python_splits, value, monom.Weight))
    return python_monoms


cpdef to_polynom_string(model):
    return to_str(ConvertFullModelToPolynomString(dereference((<_CatBoost>model).__model)))


cpdef explain_features(model):
    cdef TVector[TFeatureExplanation] featuresExplanations = ExplainFeatures(dereference((<_CatBoost>model).__model))
    result = []
    for featureExpl in featuresExplanations:
        borders = []
        for borderExpl in featureExpl.BordersExplanations:
            borders.append(
                BorderExplanation(
                    borderExpl.Border,
                    borderExpl.ProbabilityToSatisfy,
                    tvector_to_py(<TConstArrayRef[double]>borderExpl.ExpectedValueChange))
            )
        feature_type = "Float" if featureExpl.FeatureType == EMonoForestFeatureType_Float else "OneHot"
        result.append(
            FeatureExplanation(
                featureExpl.FeatureIdx,
                feature_type,
                tvector_to_py(<TConstArrayRef[double]>featureExpl.ExpectedBias), borders)
        )
    return result
