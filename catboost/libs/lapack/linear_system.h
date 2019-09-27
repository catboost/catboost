#pragma once

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>

void SolveLinearSystem(TArrayRef<double> matrix, TArrayRef<double> target);

void SolveLinearSystemCholesky(TVector<double>* matrix, TVector<double>* target);
