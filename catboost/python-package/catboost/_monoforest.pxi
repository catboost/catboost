# distutils: language = c++
# coding: utf-8
# cython: wraparound=False


cdef extern from "catboost/libs/monoforest/enums.h" namespace "NMonoForest":
    cdef cppclass EBinSplitType:
        bool_t operator==(EBinSplitType)

    cdef EBinSplitType EBinSplitType_TakeGreater "NMonoForest::EBinSplitType::TakeGreater"
    cdef EBinSplitType EFeatureType_TakeEqual "NMonoForest::EBinSplitType::TakeBin"


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
    TString ConvertFullModelToPolynomString(const TFullModel& fullModel)
    TVector[THumanReadableMonom] ConvertFullModelToPolynom(const TFullModel& fullModel)


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


cpdef to_polynom(model):
    cdef TVector[THumanReadableMonom] monoms = ConvertFullModelToPolynom(dereference((<_CatBoost>model).__model))
    python_monoms = []
    for monom in monoms:
        python_splits = []
        for split in monom.Splits:
            type = "TakeGreater" if split.SplitType == EBinSplitType_TakeGreater else "TakeEqual"
            python_splits.append(Split(split.FeatureIdx, type, split.Border))
        value = []
        for i in range(monom.Value.size()):
            value.append(monom.Value[i])
        python_monoms.append(Monom(python_splits, value, monom.Weight))
    return python_monoms


cpdef to_polynom_string(model):
    return to_native_str(ConvertFullModelToPolynomString(dereference((<_CatBoost>model).__model)))
