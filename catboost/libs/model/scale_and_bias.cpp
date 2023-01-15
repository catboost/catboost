#include  "model.h"
#include  "scale_and_bias.h"

void ApplyScaleAndBias(const TScaleAndBias& scaleAndBias, TArrayRef<double> data, size_t treeStart) {
    if (scaleAndBias.IsIdentity()) {
        return;
    }
    if (scaleAndBias.Scale != 1) {
        if (scaleAndBias.Bias == 0 || treeStart > 0) {
            for (size_t idx = 0; idx < data.size(); ++idx) {
                data[idx] *= scaleAndBias.Scale;
            }
        } else {
            for (size_t idx = 0; idx < data.size(); ++idx) {
                data[idx] = scaleAndBias.Scale * data[idx] + scaleAndBias.Bias;
            }
        }
    } else if (treeStart == 0 && scaleAndBias.Bias != 0) {
        for (size_t idx = 0; idx < data.size(); ++idx) {
            data[idx] += scaleAndBias.Bias;
        }
    }
}
