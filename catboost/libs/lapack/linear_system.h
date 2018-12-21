#pragma once

#include <util/generic/fwd.h>

template <typename T>
class TArray2D;

void SolveLinearSystem(const TArray2D<double>& matrix, const TVector<double>& proj, TVector<double>* res);

void SolveLinearSystemCholesky(TVector<double>* matrix, TVector<double>* target);
