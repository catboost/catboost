# distutils: language = c++
# coding: utf-8
# cython: wraparound=False


cdef extern from "library/cpp/grid_creator/binarization.h":
    cdef cppclass EBorderSelectionType:
        bool_t operator==(EBorderSelectionType)

    THashSet[float] BestSplit(
        TVector[float]& features,
        int maxBordersCount,
        EBorderSelectionType borderSelectionType,
        bool_t filterNans,
        bool_t featuresAreSorted
    ) except +ProcessException


cpdef _calculate_quantization_grid(values, max_borders_count, border_type):
    cdef TVector[float] valuesVector
    cdef int values_len = len(values)
    valuesVector.reserve(values_len)

    cdef int i
    for i in xrange(values_len):
        valuesVector.push_back(float(values[i]))
    cdef EBorderSelectionType borderSelectionType
    if not TryFromString[EBorderSelectionType](to_arcadia_string(border_type), borderSelectionType):
        raise CatBoostError('Unknown border selection type {}.'.format(border_type))
    cdef THashSet[float] result
    result = BestSplit(valuesVector, int(max_borders_count), borderSelectionType, False, False)

    return sorted([x for x in result])
