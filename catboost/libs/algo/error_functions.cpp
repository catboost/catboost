#include "error_functions.h"

void CalcSoftmax(const yvector<double>& approx, yvector<double>* softmax) {
    double maxApprox = *MaxElement(approx.begin(), approx.end());
    double sumExpApprox = 0;
    for (int dim = 0; dim < approx.ysize(); ++dim) {
        double expApprox = exp(approx[dim] - maxApprox);
        (*softmax)[dim] = expApprox;
        sumExpApprox += expApprox;
    }
    for (auto& curSoftmax : *softmax) {
        curSoftmax /= sumExpApprox;
    }
}

void TLoglossError::CalcFirstDerRange(int start, int count,
                                      const double* __restrict approxes, const double* __restrict approxDeltas,
                                      const float* __restrict targets, const float* __restrict weights,
                                      double* __restrict ders) const {
    if (approxDeltas != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double e = approxes[i] * approxDeltas[i];
            const double p = e / (1 + e);
            ders[i] = targets[i] > 0 ? (1 - p) : -p;
        }
    } else {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double e = approxes[i];
            const double p = e / (1 + e);
            ders[i] = targets[i] > 0 ? (1 - p) : -p;
        }
    }
    if (weights != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            ders[i] *= weights[i];
        }
    }
}

void TLoglossError::CalcDersRange(int start, int count,
                                  const double* __restrict approxExps, const double* __restrict approxDeltas,
                                  const float* __restrict targets, const float* __restrict weights,
                                  TDer1Der2* __restrict ders) const {
    if (approxDeltas != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] * approxDeltas[i] / (1 + approxExps[i] * approxDeltas[i]);
            ders[i].Der1 = targets[i] > 0 ? (1 - p) : -p;
            ders[i].Der2 = -p * (1 - p);
        }
    } else {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] / (1 + approxExps[i]);
            ders[i].Der1 = targets[i] > 0 ? (1 - p) : -p;
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

void TCrossEntropyError::CalcDersRange(int start, int count,
                                       const double* __restrict approxExps, const double* __restrict approxDeltas,
                                       const float* __restrict probs, const float* __restrict weights,
                                       TDer1Der2* __restrict ders) const {
    if (approxDeltas != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] * approxDeltas[i] / (1 + approxExps[i] * approxDeltas[i]);
            ders[i].Der1 = (probs[i] - (1 - probs[i]) * approxExps[i] * approxDeltas[i]) / (1 + approxExps[i] * approxDeltas[i]);
            ders[i].Der2 = -p * (1 - p);
        }
    } else {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] / (1 + approxExps[i]);
            ders[i].Der1 = (probs[i] - (1 - probs[i]) * approxExps[i]) / (1 + approxExps[i]);
            ders[i].Der2 = -p * (1 - p);
        }
    }
    if (weights != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            ders[i].Der1 *= weights[i];
            ders[i].Der2 *= weights[i];
        }
    }
}
