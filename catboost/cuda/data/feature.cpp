#include "feature.h"

#include <util/stream/output.h>

using namespace NCatboostCuda;

template <>
void Out<TFeatureTensor>(IOutputStream& out, const TFeatureTensor& featureTensor) {
    out << "[";
    if (featureTensor.GetSplits().size()) {
        out << "binary splits: ";
        for (auto& split : featureTensor.GetSplits()) {
            out << split.FeatureId << " / " << split.BinIdx << " " << split.SplitType << "; ";
        }
    }
    if (featureTensor.GetCatFeatures().size()) {
        out << "cat: ";
        for (auto& catFeature : featureTensor.GetCatFeatures()) {
            out << catFeature << "; ";
        }
    }
    out << "]";
}
