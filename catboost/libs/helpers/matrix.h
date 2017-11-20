#pragma once

#include <util/generic/vector.h>

#include <library/containers/2d_array/2d_array.h>

void FindSomeLinearSolution(const TArray2D<double>& matrix, const TVector<double>& proj, TVector<double>* res);
