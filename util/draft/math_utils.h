#pragma once

#include <algorithm>
#include <numeric>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>
#include <util/generic/yexception.h>

#include <util/random/mersenne.h>

#include <util/stream/output.h>

#include <util/system/yassert.h>

template <class T>
class TIntHistogram: public yvector<T> {
public:
    T& operator[](size_t i) {
        if (i >= yvector<T>::size())
            yvector<T>::resize(i + 1, T(0));
        return yvector<T>::operator[](i);
    }

    T operator[](size_t i) const {
        return yvector<T>::operator[](i);
    }

    void AddTo(size_t i) {
        ++(operator[](i));
    }

    void operator+=(const TIntHistogram<T>& rhs) {
        const yvector<T>& rhs1 = rhs;
        size_t addSize = Min(yvector<T>::size(), rhs1.size());
        for (size_t i = 0; i < addSize; ++i)
            (*this)[i] += rhs1[i];

        if (rhs1.size() > yvector<T>::size())
            yvector<T>::insert(yvector<T>::end(), rhs1.begin() + addSize, rhs1.end());
    }

    void Trim(size_t nMax) {
        if (yvector<T>::size() > nMax)
            yvector<T>::resize(nMax);
    }

    void Print(IOutputStream& out,
               bool vertical = true, bool printArgs = true, bool printEmpty = true) const {
        size_t nParts = yvector<T>::size();
        if (nParts == 0)
            return;

        if (vertical) {
            for (size_t i = 0; i < nParts; ++i) {
                T v = (operator[](i));
                if ((v == 0) && !printEmpty)
                    continue;
                if (printArgs)
                    out << i << '\t';
                out << v << '\n';
            }
        } else {
            if (printArgs) {
                bool first = true;
                for (size_t i = 0; i < nParts; ++i) {
                    T v = (operator[](i));
                    if ((v == 0) && !printEmpty)
                        continue;
                    if (first) {
                        first = false;
                    } else {
                        out << '\t';
                    }
                    out << i;
                }
                out << '\n';
            }
            bool first = true;
            for (size_t i = 0; i < nParts; ++i) {
                T v = (operator[](i));
                if ((v == 0) && !printEmpty)
                    continue;
                if (first) {
                    first = false;
                } else {
                    out << '\t';
                }
                out << v;
            }
            out << '\n';
        }
    }
};

template <class T>
class TPeriodHistogram: public TIntHistogram<T> {
public:
    float Period;
    float Start;

public:
    TPeriodHistogram(float period = 1.0f, float start = 0.0f)
        : Period(period)
        , Start(start)
    {
    }

    T& operator[](float arg) {
        if (arg < Start)
            ythrow yexception() << "Val " << arg << " is outside the histogram";

        size_t i = size_t((arg - Start) / Period);
        return TIntHistogram<T>::operator[](i);
    }
};

template <class T>
inline void UpdateMin(T& min, const T& v) {
    if (v < min)
        min = v;
}

template <class T>
inline void UpdateMax(T& max, const T& v) {
    if (v > max)
        max = v;
}
