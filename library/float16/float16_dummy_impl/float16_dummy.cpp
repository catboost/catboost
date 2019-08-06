#include <library/float16/float16.h>

#include <util/system/yassert.h>

bool NFloat16Ops::IsIntrisincsAvailableOnHost() {
    return false;
}

void NFloat16Ops::UnpackFloat16SequenceIntrisincs(const TFloat16*, float*, size_t) {
    Y_FAIL("NFloat16Ops::UnpackFloat16SequenceIntrisincs() is not implemented on this platform");
}

float NFloat16Ops::DotProductOnFloatIntrisincs(const float*, const TFloat16*, size_t) {
    Y_FAIL("NFloat16Ops::DotProductOnFloatIntrisincs() is not implemented on this platform");
}
