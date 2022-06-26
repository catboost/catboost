#include "linear_system.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/ymath.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>

#include <contrib/libs/clapack/clapack.h>


void SolveLinearSystem(TArrayRef<double> matrix, TArrayRef<double> target) {
    const auto expectedMatrixSize = target.size() * (target.size() + 1) / 2;
    CB_ENSURE_INTERNAL(
        matrix.size() == expectedMatrixSize,
        "Packed matrix size for right hand side size " << target.size()
        << " should be " << expectedMatrixSize << ", not " << matrix.size()
    );
    if (target.size() == 1) {
        target[0] /= matrix[0];
        return;
    }

    char matrixStorageType[] = {'L', '\0'};
    int systemSize = target.ysize();
    int numberOfRightHandSides = 1;
    int info = 0;

    dppsv_(matrixStorageType, &systemSize, &numberOfRightHandSides, matrix.data(), target.data(), &systemSize, &info);

    CB_ENSURE(info == 0, "System of linear equations is not positive definite");
}

void SolveLinearSystemCholesky(TVector<double>* matrix,
                               TVector<double>* target) {
    if (target->size() == 1) {
        (*target)[0] /= (*matrix)[0];
        return;
    }

    char matrixStorageType[] = {'U', '\0'};
    int systemSize = target->ysize();
    int numberOfRightHandSides = 1;

    int info = 0;
    dposv_(matrixStorageType, &systemSize, &numberOfRightHandSides, matrix->data(), &systemSize,
           target->data(), &systemSize, &info);

    CB_ENSURE(info >= 0, "LAPACK dposv_ failed with status " << info);
}
