#pragma once

#include <util/system/types.h>
#include <util/generic/vector.h>
#include <catboost/libs/helpers/exception.h>

namespace NCatboostCuda {
    enum class EFeaturesGroupingPolicy {
        BinaryFeatures,
        HalfByteFeatures,
        OneByteFeatures
    };

    template <EFeaturesGroupingPolicy>
    struct TFeaturePolicyTraits;

    template <>
    struct TFeaturePolicyTraits<EFeaturesGroupingPolicy::BinaryFeatures> {
        static constexpr ui32 BitsPerFeature() {
            return 1u;
        }

        static constexpr ui32 FeaturesPerInt() {
            return 32 / BitsPerFeature();
        }
    };

    template <>
    struct TFeaturePolicyTraits<EFeaturesGroupingPolicy::HalfByteFeatures> {
        static constexpr ui32 BitsPerFeature() {
            return 4u;
        }

        static constexpr ui32 FeaturesPerInt() {
            return 32 / BitsPerFeature();
        }
    };

    template <>
    struct TFeaturePolicyTraits<EFeaturesGroupingPolicy::OneByteFeatures> {
        static constexpr ui32 BitsPerFeature() {
            return 8u;
        }

        static constexpr ui32 FeaturesPerInt() {
            return 32 / BitsPerFeature();
        }
    };

    template <EFeaturesGroupingPolicy Policy>
    class TCompressedIndexHelper {
    public:
        static ui32 Mask() {
            const ui32 mask = (1ULL << TFeaturePolicyTraits<Policy>::BitsPerFeature()) - 1;
            return mask;
        }

        static ui32 BinCount() {
            return 1 << TFeaturePolicyTraits<Policy>::BitsPerFeature();
        }

        static ui32 MaxFolds() {
            return (1ULL << TFeaturePolicyTraits<Policy>::BitsPerFeature()) - 1;
        }

        static ui32 ShiftedMask(ui32 featureId) {
            return Mask() << Shift(featureId);
        }

        static ui32 Shift(ui32 featureId) {
            const ui32 localId = featureId % TFeaturePolicyTraits<Policy>::FeaturesPerInt();
            return 32 - (1 + localId) * TFeaturePolicyTraits<Policy>::BitsPerFeature();
        }

        static ui32 FeaturesPerInt() {
            return TFeaturePolicyTraits<Policy>::FeaturesPerInt();
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
                ythrow TCatBoostException() << "Unknown policy " << policy;
            }
        }
    }

    inline ui32 GetShift(EFeaturesGroupingPolicy policy, ui32 fid) {
        switch (policy) {
            case EFeaturesGroupingPolicy::BinaryFeatures: {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::BinaryFeatures>::Shift(fid);
            }
            case EFeaturesGroupingPolicy::HalfByteFeatures: {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::HalfByteFeatures>::Shift(fid);
            }
            case EFeaturesGroupingPolicy::OneByteFeatures: {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::OneByteFeatures>::Shift(fid);
            }
            default: {
                ythrow TCatBoostException() << "Unknown policy " << policy;
            }
        }
    }
}
