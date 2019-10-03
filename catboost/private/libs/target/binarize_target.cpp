#include "binarize_target.h"

void PrepareTargetBinary(TConstArrayRef<float> srcTarget, float border, TVector<float>* dstTarget) {
    if (srcTarget.data() != dstTarget->data()) {
        dstTarget->yresize(srcTarget.size());
    }
    for (size_t i = 0; i < srcTarget.size(); ++i) {
        (*dstTarget)[i] = srcTarget[i] > border;
    }
}
