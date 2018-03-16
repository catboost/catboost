#pragma once

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>

template<typename T>
static TVector<const T*> GetConstPointers(const TVector<THolder<T>>& pointers) {
    TVector<const T*> result(pointers.ysize());
    for (int i = 0; i < pointers.ysize(); ++i) {
        result[i] = pointers[i].Get();
    }
    return result;
}

template <typename T>
struct TMinMax {
    T Min;
    T Max;
};

template <typename TForwardIterator, typename T = typename std::iterator_traits<TForwardIterator>::value_type>
inline TMinMax<T> CalcMinMax(TForwardIterator begin, TForwardIterator end) {
    auto minmax = std::minmax_element(begin, end);
    Y_VERIFY(minmax.first != end);
    return {*minmax.first, *minmax.second};
}

template <typename T>
inline TMinMax<T> CalcMinMax(const TVector<T>& v) {
    return CalcMinMax(v.begin(), v.end());
}
inline bool IsConst(const TVector<float>& values) {
    if (values.empty()) {
        return true;
    }
    auto bounds = CalcMinMax(values);
    return bounds.Min == bounds.Max;
}
