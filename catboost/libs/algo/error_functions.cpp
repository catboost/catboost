#include "error_functions.h"

void TCrossEntropyError::CalcFirstDerRange(int start, int count,
                                      const double* __restrict approxes, const double* __restrict approxDeltas,
                                      const float* __restrict targets, const float* __restrict weights,
                                      double* __restrict ders) const {
    if (approxDeltas != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double e = approxes[i] * approxDeltas[i];
            const double p = e / (1 + e);
            ders[i] = targets[i] - p;
        }
    } else {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double e = approxes[i];
            const double p = e / (1 + e);
            ders[i] = targets[i] - p;
        }
    }
    if (weights != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            ders[i] *= weights[i];
        }
    }
}

void TCrossEntropyError::CalcDersRange(int start, int count,
                                  const double* __restrict approxExps, const double* __restrict approxDeltas,
                                  const float* __restrict targets, const float* __restrict weights,
                                  TDer1Der2* __restrict ders) const {
    if (approxDeltas != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] * approxDeltas[i] / (1 + approxExps[i] * approxDeltas[i]);
            ders[i].Der1 = targets[i] - p;
            ders[i].Der2 = -p * (1 - p);
        }
    } else {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] / (1 + approxExps[i]);
            ders[i].Der1 = targets[i] - p;
            ders[i].Der2 = -p * (1 - p);
        }
    }
    if (weights != nullptr) {
#pragma clang loop vectorize_width(8) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            ders[i].Der1 *= weights[i];
            ders[i].Der2 *= weights[i];
        }
    }
}
