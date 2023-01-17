#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/mutex.h>

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
    CB_ENSURE(minmax.first != end, "Empty iterator range in CalcMinMax");
    return {*minmax.first, *minmax.second};
}

template <typename T>
inline TMinMax<T> CalcMinMax(TConstArrayRef<T> array) {
    return CalcMinMax(array.begin(), array.end());
}

template <typename T>
void GuardedUpdateMinMax(const TMinMax<T>& value, TMinMax<T> volatile* target, TMutex& guard);

inline bool IsConst(TConstArrayRef<float> array) {
    if (array.empty()) {
        return true;
    }
    return FindIf(
        array.begin() + 1,
        array.end(),
        [first = array.front()] (auto element) { return element != first; }
    ) == array.end();
}

template <typename TArrayLike>
inline TArrayRef<typename TArrayLike::value_type> GetSlice(TArrayLike& array, size_t offset, size_t count) {
    if (array.empty()) {
        return TArrayRef<typename TArrayLike::value_type>();
    }
    return TArrayRef<typename TArrayLike::value_type>(array.begin() + offset, count);
}

template <typename TArrayLike, typename TIsDefined>
inline typename TArrayLike::value_type GetIf(TIsDefined isDefined, const TArrayLike& array, size_t index, typename TArrayLike::value_type orElse) {
    return isDefined ? array.at(index) : orElse;
}

template <typename Int1, typename Int2, typename T>
inline void ResizeRank2(Int1 dim1, Int2 dim2, TVector<TVector<T>>& vvt) {
    vvt.resize(dim1);
    for (auto& vt : vvt) {
        vt.resize(dim2);
    }
}

template <typename Int1, typename Int2, typename T>
inline void AllocateRank2(Int1 dim1, Int2 dim2, TVector<TVector<T>>& vvt) {
    vvt.resize(dim1);
    for (auto& vt : vvt) {
        vt.yresize(dim2);
    }
}

template <class T1, class T2>
void Assign(TConstArrayRef<T1> arrayRef, TVector<T2>* v) {
    v->assign(arrayRef.begin(), arrayRef.end());
}

template <class T1, class T2>
inline void AssignRank2(TConstArrayRef<TConstArrayRef<T1>> src, TVector<TVector<T2>>* dst) {
    dst->resize(src.size());
    for (auto dim1 : xrange(src.size())) {
        Assign(src[dim1], &((*dst)[dim1]));
    }
}

template <typename T, typename T2DArrayLike>
inline static TVector<TConstArrayRef<T>> To2DConstArrayRef(const T2DArrayLike& array) {
    auto arrayView = TVector<TConstArrayRef<T>>();
    for (const auto& subArray : array) {
        arrayView.emplace_back(subArray);
    }
    return arrayView;
}

template <typename T, typename T2DArrayLike>
inline static TVector<TConstArrayRef<T>> To2DConstArrayRef(const T2DArrayLike& array, size_t offset, size_t count) {
    auto arrayView = TVector<TConstArrayRef<T>>();
    for (const auto& subArray : array) {
        arrayView.emplace_back(MakeArrayRef(subArray.begin() + offset, count));
    }
    return arrayView;
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

template <class T>
inline bool AreEqualTo(TConstArrayRef<T> entries, const T& value) {
    for (const auto& entry : entries) {
        if (entry != value) {
            return false;
        }
    }
    return true;
}

template <typename T>
inline void SumTransposedBlocks(
    int srcColumnBegin,
    int srcColumnEnd,
    TConstArrayRef<TVector<T>> srcA,
    TConstArrayRef<TVector<T>> srcB,
    TArrayRef<TVector<T>> dst
) {
    Y_ASSERT(srcColumnEnd - srcColumnBegin <= IntegerCast<int>(dst.size()));
    if (srcB.empty()) {
        for (int srcRowIdx : xrange(srcA.size())) {
            for (int srcColumnIdx : xrange(srcColumnBegin, srcColumnEnd)) {
                dst[srcColumnIdx - srcColumnBegin][srcRowIdx] = srcA[srcRowIdx][srcColumnIdx];
            }
        }
    } else {
        Y_ASSERT(srcA.size() == srcB.size());
        for (int srcRowIdx : xrange(srcA.size())) {
            for (int srcColumnIdx : xrange(srcColumnBegin, srcColumnEnd)) {
                dst[srcColumnIdx - srcColumnBegin][srcRowIdx] = srcA[srcRowIdx][srcColumnIdx] + srcB[srcRowIdx][srcColumnIdx];
            }
        }
    }
}

template <typename T>
inline void SumTransposedBlocks(
    int srcColumnBegin,
    int srcColumnEnd,
    TConstArrayRef<TConstArrayRef<T>> srcA,
    TConstArrayRef<TConstArrayRef<T>> srcB,
    TArrayRef<TVector<T>> dst
) {
    Y_ASSERT(srcColumnEnd - srcColumnBegin <= IntegerCast<int>(dst.size()));
    if (srcB.empty()) {
        for (int srcRowIdx : xrange(srcA.size())) {
            for (int srcColumnIdx : xrange(srcColumnBegin, srcColumnEnd)) {
                dst[srcColumnIdx - srcColumnBegin][srcRowIdx] = srcA[srcRowIdx][srcColumnIdx];
            }
        }
    } else {
        Y_ASSERT(srcA.size() == srcB.size());
        for (int srcRowIdx : xrange(srcA.size())) {
            for (int srcColumnIdx : xrange(srcColumnBegin, srcColumnEnd)) {
                dst[srcColumnIdx - srcColumnBegin][srcRowIdx] = srcA[srcRowIdx][srcColumnIdx] + srcB[srcRowIdx][srcColumnIdx];
            }
        }
    }
}

template <typename T>
TVector<size_t> GetNonEmptyElementsIndices(const TVector<TVector<T>>& data) {
    TVector<size_t> result;
    result.reserve(data.size());
    for (auto i : xrange(data.size())) {
        if (!data[i].empty()) {
            result.push_back(i);
        }
    }
    return result;
}
