#include "error_functions.h"

#include <library/fast_exp/fast_exp.h>

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


void TBinclassError::CalcFirstDerRange(int count, const double* approxes, const float* targets, const float* weights,
                                       double* ders) const {
    for (int i = 0; i < count; ++i) {
        const double e = exp(approxes[i]);
        const double p = e / (1 + e);
        ders[i] = targets[i] > 0 ? (1 - p) : -p;
    }
    if (weights) {
        for (int i = 0; i < count; ++i) {
            ders[i] *= weights[i];
        }
    }
}

void TBinclassError::CalcDersRange(int count, const double* approxes, const float* targets, const float* weights,
                                   TDer1Der2* ders) const {
    yvector<double> approxExps(approxes, approxes + count);
    FastExpInplace(approxExps.data(), count);
    for (int i = 0; i < count; ++i) {
        const double p = approxExps[i] / (1 + approxExps[i]);
        ders[i].Der1 = targets[i] > 0 ? (1 - p) : -p;
        ders[i].Der2 = -p * (1 - p);
    }
    if (weights) {
        for (int i = 0; i < count; ++i) {
            ders[i].Der1 *= weights[i];
            ders[i].Der2 *= weights[i];
        }
    }
}

void TCrossEntropyError::CalcDersRange(int count, const double* approxes, const float* probs, const float* weights,
                                       TDer1Der2* ders) const {
    yvector<double> approxExps(approxes, approxes + count);
    FastExpInplace(approxExps.data(), count);
    for (int i = 0; i < count; ++i) {
        const double p = approxExps[i] / (1 + approxExps[i]);
        ders[i].Der1 = (probs[i] - (1 - probs[i]) * approxExps[i]) / (1 + approxExps[i]);
        ders[i].Der2 = -p * (1 - p);
    }
    if (weights) {
        for (int i = 0; i < count; ++i) {
            ders[i].Der1 *= weights[i];
            ders[i].Der2 *= weights[i];
        }
    }
}

