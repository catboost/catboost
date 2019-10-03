#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

// srcTarget can be the same array as dstTarget
void PrepareTargetBinary(TConstArrayRef<float> srcTarget, float border, TVector<float>* dstTarget);

