#pragma once

#include <util/generic/vector.h>
#include <library/containers/2d_array/2d_array.h>

void SolveLinearSystem(const TArray2D<double>& matrix, const TVector<double>& proj, TVector<double>* res);

void SolveLinearSystemCholesky(TVector<double>* matrix, TVector<double>* target);

template <class T>
inline void MakeZeroAverage(TVector<T>* res) {
    T average = 0;
    for (int i = 0; i < res->ysize(); ++i) {
        average += (*res)[i];
    }
    average /= res->ysize();
    for (int i = 0; i < res->ysize(); ++i) {
        (*res)[i] -= average;
    }
}
