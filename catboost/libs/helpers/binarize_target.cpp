#include "binarize_target.h"

void PrepareTargetBinary(float border, TVector<float>* target) {
    for (int i = 0; i < (*target).ysize(); ++i) {
        (*target)[i] = ((*target)[i] > border);
    }
}
