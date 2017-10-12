#pragma once

#include <util/system/types.h>

template <ui32 BITS_PER_FEATURE,
          ui32 FEATURES_PER_BYTE>
struct TGridPolicy {

    static constexpr ui32 BitsPerFeature() {
        return BITS_PER_FEATURE;
    }

    static constexpr ui32 BinCount() {
        return 1 << BitsPerFeature();
    }

    static constexpr ui32 FeaturesPerByte() {
        return FEATURES_PER_BYTE;
    }

    static constexpr ui32 CanProceed(ui32 binCount) {
        return (binCount <= BinCount()) && binCount > 1;
    }
};

template <class TGridPolicy>
class TCompressedIndexHelper {
public:
    static constexpr ui32 Mask() {
        const ui32 mask = (1ULL << TGridPolicy::BitsPerFeature()) - 1;
        return mask;
    }

    static constexpr ui32 MaxFolds() {
        return (1ULL << TGridPolicy::BitsPerFeature()) - 1;
    }

    static constexpr ui32 ShiftedMask(ui32 featureId) {
        return Mask() << Shift(featureId);
    }

    static constexpr ui32 Shift(ui32 featureId) {
        const ui32 entriesPerInt = 4 * TGridPolicy::FeaturesPerByte();
        const ui32 localId = featureId % (entriesPerInt);
        const ui32 byteId = localId / TGridPolicy::FeaturesPerByte();
        const ui32 shiftInByte = (TGridPolicy::FeaturesPerByte() - localId % TGridPolicy::FeaturesPerByte() - 1) * TGridPolicy::BitsPerFeature();
        const ui32 shift = (4 - byteId - 1) * 8 + shiftInByte;
        return shift;
    }

    static constexpr ui32 FeaturesPerInt() {
        return (4 * TGridPolicy::FeaturesPerByte());
    }
};
