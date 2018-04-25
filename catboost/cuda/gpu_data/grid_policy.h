#pragma once

#include <util/system/types.h>
#include <util/generic/vector.h>
#include <catboost/libs/helpers/exception.h>

namespace NCatboostCuda {
    enum EFeaturesGroupingPolicy {
        BinaryFeatures,
        HalfByteFeatures,
        OneByteFeatures
    };

    inline TVector<EFeaturesGroupingPolicy> GetAllGroupingPolicies() {
        return {EFeaturesGroupingPolicy::BinaryFeatures, EFeaturesGroupingPolicy::HalfByteFeatures, EFeaturesGroupingPolicy::OneByteFeatures};
    }

    template <EFeaturesGroupingPolicy>
    struct TBitsPerFeature;

    template <EFeaturesGroupingPolicy>
    struct TFeaturesPerByte;

    template <>
    struct TBitsPerFeature<EFeaturesGroupingPolicy::BinaryFeatures> {
        static constexpr ui32 BitsPerFeature() {
            return 1u;
        }
    };

    template <>
    struct TBitsPerFeature<EFeaturesGroupingPolicy::HalfByteFeatures> {
        static constexpr ui32 BitsPerFeature() {
            return 4u;
        }
    };

    template <>
    struct TBitsPerFeature<EFeaturesGroupingPolicy::OneByteFeatures> {
        static constexpr ui32 BitsPerFeature() {
            return 8u;
        }
    };

    template <>
    struct TFeaturesPerByte<EFeaturesGroupingPolicy::OneByteFeatures> {
        static constexpr ui32 FeaturesPerByte() {
            return 1u;
        }
    };

    template <>
    struct TFeaturesPerByte<EFeaturesGroupingPolicy::HalfByteFeatures> {
        static constexpr ui32 FeaturesPerByte() {
            return 2u;
        }
    };

    template <>
    struct TFeaturesPerByte<EFeaturesGroupingPolicy::BinaryFeatures> {
        static constexpr ui32 FeaturesPerByte() {
            return 8u;
        }
    };

    template <EFeaturesGroupingPolicy Policy>
    class TCompressedIndexHelper {
    public:
        static ui32 Mask() {
            const ui32 mask = (1ULL << TBitsPerFeature<Policy>::BitsPerFeature()) - 1;
            return mask;
        }

        static ui32 BinCount() {
            return 1 << TBitsPerFeature<Policy>::BitsPerFeature();
        }

        static ui32 MaxFolds() {
            return (1ULL << TBitsPerFeature<Policy>::BitsPerFeature()) - 1;
        }

        static ui32 ShiftedMask(ui32 featureId) {
            return Mask() << Shift(featureId);
        }

        static ui32 Shift(ui32 featureId) {
            const ui32 entriesPerInt = 4 * TFeaturesPerByte<Policy>::FeaturesPerByte();
            const ui32 localId = featureId % (entriesPerInt);
            const ui32 byteId = localId / TFeaturesPerByte<Policy>::FeaturesPerByte();
            const ui32 shiftInByte = (TFeaturesPerByte<Policy>::FeaturesPerByte() - localId % TFeaturesPerByte<Policy>::FeaturesPerByte() - 1) *
                                     TBitsPerFeature<Policy>::BitsPerFeature();
            const ui32 shift = (4 - byteId - 1) * 8 + shiftInByte;
            return shift;
        }

        static ui32 FeaturesPerInt() {
            return (4 * TFeaturesPerByte<Policy>::FeaturesPerByte());
        }
    };

    inline ui32 GetFeaturesPerInt(EFeaturesGroupingPolicy policy) {
        switch (policy) {
            case EFeaturesGroupingPolicy::BinaryFeatures: {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::BinaryFeatures>::FeaturesPerInt();
            }
            case EFeaturesGroupingPolicy::HalfByteFeatures: {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::HalfByteFeatures>::FeaturesPerInt();
            }
            case EFeaturesGroupingPolicy::OneByteFeatures: {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::OneByteFeatures>::FeaturesPerInt();
            }
            default: {
                ythrow TCatboostException() << "Unknown policy " << policy;
            }
        }
    }
}
