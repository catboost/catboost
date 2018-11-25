#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>

#include <algorithm>


template <typename T>
inline TVector<const T*> GetConstPointers(const TVector<T>& objects) {
    TVector<const T*> result(objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
        result[i] = &objects[i];
    }
    return result;
}

template <typename T>
inline TVector<T*> GetMutablePointers(TVector<T>& objects) {
    TVector<T*> result(objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
        result[i] = &objects[i];
    }
    return result;
}

template <typename T>
inline TVector<const T*> GetConstPointers(const TVector<THolder<T>>& holders) {
    TVector<const T*> result(holders.size());
    for (size_t i = 0; i < holders.size(); ++i) {
        result[i] = holders[i].Get();
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

template <typename Int1, typename Int2, typename T>
inline void ResizeRank2(Int1 dim1, Int2 dim2, TVector<TVector<T>>& vvt) {
    vvt.resize(dim1);
    for (auto& vt : vvt) {
        vt.resize(dim2);
    }
}

template <class T>
void Assign(TConstArrayRef<T> arrayRef, TVector<T>* v) {
    v->assign(arrayRef.begin(), arrayRef.end());
}

template <class T>
bool Equal(TConstArrayRef<T> arrayRef, const TVector<T>& v) {
    return arrayRef == TConstArrayRef<T>(v);
}

template <class T>
bool ApproximatelyEqual(TConstArrayRef<T> lhs, TConstArrayRef<T> rhs, const T eps) {
    return std::equal(
        lhs.begin(),
        lhs.end(),
        rhs.begin(),
        rhs.end(),
        [eps](T lElement, T rElement) { return Abs(lElement - rElement) < eps; }
    );
}
