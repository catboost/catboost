#include <library/cpp/float16/float16.h>

#include <util/system/yassert.h>

bool NFloat16Impl::AreConversionIntrinsicsAvailableOnHost() {
    return false;
}

ui16 NFloat16Impl::ConvertFloat32IntoFloat16Intrinsics(float) {
    Y_ABORT("NFloat16Impl::ConvertFloat32IntoFloat16Intrinsics() is not implemented on this platform");
}

float NFloat16Impl::ConvertFloat16IntoFloat32Intrinsics(ui16) {
    Y_ABORT("NFloat16Ops::ConvertFloat16IntoFloat32Intrinsics() is not implemented on this platform");
}

bool NFloat16Ops::AreIntrinsicsAvailableOnHost() {
    return false;
}

void NFloat16Ops::UnpackFloat16SequenceIntrisincs(const TFloat16*, float*, size_t) {
    Y_ABORT("NFloat16Ops::UnpackFloat16SequenceIntrisincs() is not implemented on this platform");
}

float NFloat16Ops::DotProductOnFloatIntrisincs(const float*, const TFloat16*, size_t) {
    Y_ABORT("NFloat16Ops::DotProductOnFloatIntrisincs() is not implemented on this platform");
}

void NFloat16Ops::PackFloat16SequenceIntrisincs(const float*, TFloat16*, size_t) {
    Y_ABORT("NFloat16Ops::PackFloat16SequenceIntrisincs() is not implemented on this platform");
}
