#include "model.h"
#include "scale_and_bias.h"
#include "cpu/simd.h"

void ApplyScaleAndBias(const TScaleAndBias& scaleAndBias, TArrayRef<double> data, size_t treeStart) {
    if (scaleAndBias.IsIdentity()) {
        return;
    }
    auto biasRef = scaleAndBias.GetBiasRef();
    bool isZeroBias = scaleAndBias.IsZeroBias();
    if (biasRef.size() == 1 || isZeroBias) {
        double bias = scaleAndBias.GetOneDimensionalBiasOrZero();
        if (scaleAndBias.Scale == 1.0 && (isZeroBias || treeStart > 0)) {
            return;
        }
        if (data.size() >= 128) {
            if (auto kernel = NCB::NModelEvaluation::NDetail::GetWideScaleBias()) {
                kernel(data.data(), data.size(), scaleAndBias.Scale, bias, !isZeroBias && treeStart == 0);
                return;
            }
        }
        if (scaleAndBias.Scale != 1 || isZeroBias) {
            if (isZeroBias || treeStart > 0) {
                for (size_t idx = 0; idx < data.size(); ++idx) {
                    data[idx] *= scaleAndBias.Scale;
                }
            } else {
                for (size_t idx = 0; idx < data.size(); ++idx) {
                    data[idx] = scaleAndBias.Scale * data[idx] + bias;
                }
            }
        } else if (treeStart == 0) { // bias != 0
            for (size_t idx = 0; idx < data.size(); ++idx) {
                data[idx] += bias;
            }
        }
    } else { // isZeroBias = false && biasRef.size() > 1
        if (scaleAndBias.Scale != 1) {
            if (treeStart > 0) {
                for (size_t idx = 0; idx < data.size();) {
                    for (size_t dim = 0; dim < biasRef.size(); ++dim, ++idx) {
                        data[idx] = scaleAndBias.Scale * data[idx];
                    }
                }
            } else {
                for (size_t idx = 0; idx < data.size();) {
                    for (size_t dim = 0; dim < biasRef.size(); ++dim, ++idx) {
                        data[idx] = scaleAndBias.Scale * data[idx] + biasRef[dim];
                    }
                }
            }
        } else if (treeStart == 0) {
            for (size_t idx = 0; idx < data.size();) {
                for (size_t dim = 0; dim < biasRef.size(); ++dim, ++idx) {
                    data[idx] += biasRef[dim];
                }
            }
        }
    }
}
