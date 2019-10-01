#include "feature_index.h"

#include <util/stream/output.h>

// Impossible to do one templated Out because function template partial specialization is not allowed

template <>
void Out<NCB::TFloatFeatureIdx>(IOutputStream& output, const NCB::TFloatFeatureIdx& floatFeatureIdx) {
    output << "float feature #" << *floatFeatureIdx;
}

template <>
void Out<NCB::TCatFeatureIdx>(IOutputStream& output, const NCB::TCatFeatureIdx& catFeatureIdx) {
    output << "categorical feature #" << *catFeatureIdx;
}

template <>
void Out<NCB::TTextFeatureIdx>(IOutputStream& output, const NCB::TTextFeatureIdx& textFeatureIdx) {
    output << "text feature #" << *textFeatureIdx;
}
