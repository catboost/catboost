#include "matrix.h"

#include <util/generic/ymath.h>

// TODO(vitekmel): styleguide

void FindSomeLinearSolution(const TArray2D<double>& matrix, const TVector<double>& proj, TVector<double>* res) {
    size_t nSize = proj.size();
    Y_ASSERT(matrix.GetXSize() == nSize && matrix.GetYSize() == nSize);
    res->resize(nSize);

    TArray2D<double> left(matrix);
    TArray2D<double> right(nSize, nSize);
    for (size_t y = 0; y < nSize; ++y) {
        for (size_t x = 0; x < nSize; ++x) {
            right[y][x] = x == y;
        }
    }

    const double about0 = 1e-5f;
    for (size_t y = 0; y < nSize; ++y) {
        double fDiag = left[y][y], fMax = Abs(fDiag);
        size_t nBestRow = y;
        for (size_t k = y + 1; k < nSize; ++k) {
            double fTest = Abs(left[k][y]);
            if (fTest > 2 * fMax) {
                fMax = fTest;
                nBestRow = k;
            }
        }

        if (nBestRow != y) {
            double f = fMax * fDiag > 0 ? 1 : -1;
            for (size_t x = 0; x < nSize; ++x) {
                left[y][x] += f * left[nBestRow][x];
                right[y][x] += f * right[nBestRow][x];
            }
        }

        fDiag = left[y][y];
        if (Abs(fDiag) < about0) {
            ptrdiff_t h = y;
            while ((h < (ptrdiff_t)nSize - 1) && (Abs(fDiag) < about0)) {
                h++;
                if (Abs(left[h][y]) > about0) {
                    for (size_t u = 0; u < nSize; u++) {
                        left[y][u] += left[h][u];
                        right[y][u] += right[h][u];
                    }
                    fDiag = left[y][y];
                }
            }
            if (Abs(fDiag) < 1e-8) {
                continue;
            }
        }

        double fDiag1 = 1 / fDiag;
        for (size_t x = 0; x < nSize; ++x) {
            left[y][x] *= fDiag1;
            right[y][x] *= fDiag1;
        }

        size_t leftOffset = 0, rightOffset = nSize;
        while (leftOffset < nSize && left[y][leftOffset] == 0) {
            ++leftOffset;
        }
        while (rightOffset > 0 && right[y][rightOffset - 1] == 0) {
            --rightOffset;
        }

        for (size_t k = 0; k < nSize; ++k) {
            if (k == y) {
                continue;
            }
            double fK = left[k][y];
            if (fK == 0) {
                continue;
            }

            {
                double* leftPtr = &left[k][0] + leftOffset;
                const double* leftSub = &left[y][0] + leftOffset;
                double* finPtr = &left[k][0] + nSize;
                while (leftPtr < finPtr) {
                    *leftPtr++ -= *leftSub++ * fK;
                }
            }

            {
                double* rightPtr = &right[k][0];
                const double* rightSub = &right[y][0];
                double* finPtr = &right[k][0] + rightOffset;
                while (rightPtr < finPtr) {
                    *rightPtr++ -= *rightSub++ * fK;
                }
            }
        }
    }

    for (size_t y = 0; y < nSize; ++y) {
        double fRes = 0;
        for (size_t x = 0; x < nSize; ++x) {
            fRes += right[y][x] * proj[x];
        }
        (*res)[y] = fRes;
    }
}
