#pragma once

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

#include <array>
#include <cmath>

namespace NCB {
    // CUDA's configured grid policy, before actual CTR borders can shrink.
    // Values follow EFeaturesGroupingPolicy's ordered TMap traversal.
    enum class EMetalMetaL2Policy : ui8 { Binary = 0, HalfByte = 1, OneByte = 2 };

    struct TMetalMetaL2Feature {
        ui32 RuntimeFeature = 0;
        EMetalMetaL2Policy Policy = EMetalMetaL2Policy::Binary;
    };

    struct TMetalMetaL2DataSet {
        // Already consumed from the shared trainer stream. This helper never
        // advances that stream or owns a second snapshot cursor.
        ui64 ScoreSeed = 0;
        // Includes configured nonempty policies even if their corresponding
        // native columns currently have no useful split candidates.
        ui8 PolicyMask = 0;
        TVector<TMetalMetaL2Feature> Features;
    };

    inline double MetalMetaL2Uniform(ui64 seed) {
        // random_gen.cuh::NextUniform: one multiply-with-carry step, including
        // uint32 overflow before conversion and the inclusive-one endpoint.
        ui32 v = ui32(seed >> 32);
        ui32 u = ui32(seed);
        v = 36969u * (v & 0xffffu) + (v >> 16);
        u = 18000u * (u & 0xffffu) + (u >> 16);
        const ui32 value = (v << 16) + u;
        return value * 2.328306435996595e-10;
    }

    inline std::array<float, 3> MetalMetaL2PolicyExponents(
            ui64 datasetSeed, ui8 policyMask, double exponent, double frequency) {
        CB_ENSURE((policyMask & ~ui8(7)) == 0 && std::isfinite(exponent) &&
            std::isfinite(float(exponent)) && std::isfinite(frequency),
            "Invalid Metal meta-L2 policy or parameters");
        std::array<float, 3> result{1, 1, 1};
        TRandom policies(datasetSeed);
        for (ui32 policy = 0; policy < result.size(); ++policy) {
            if (policyMask & (1u << policy)) {
                const ui64 seed = policies.NextUniformL();
                result[policy] = MetalMetaL2Uniform(seed) >= frequency ? 1.0f : float(exponent);
            }
        }
        return result;
    }

    inline ui64 MetalMetaL2TreeDataSetSeed(ui64 visitorSeed, ui32 device, ui64 baseTensorHash) {
        // tree_ctr_datasets_visitor.cpp derives one local seed per device,
        // then adds the base tensor hash before the policy-local expansion.
        TRandom devices(visitorSeed);
        devices.Advance(device);
        return devices.NextUniformL() + baseTensorHash;
    }

    inline TVector<float> BuildMetalMetaL2FeatureExponents(ui32 runtimeFeatures,
            double exponent, double frequency, TConstArrayRef<TMetalMetaL2DataSet> datasets) {
        // Unscored/inactive columns keep exponent one. A caller must identify
        // separate static, dependent and dynamic tensor datasets explicitly;
        // feature IDs alone cannot recover their independent score seeds.
        CB_ENSURE(std::isfinite(exponent) && std::isfinite(float(exponent)) && std::isfinite(frequency),
            "Invalid Metal meta-L2 parameters");
        TVector<float> result(runtimeFeatures, 1.0f);
        TVector<ui8> assigned(runtimeFeatures, 0);
        for (const auto& dataset : datasets) {
            const auto selected = MetalMetaL2PolicyExponents(
                dataset.ScoreSeed, dataset.PolicyMask, exponent, frequency);
            for (const auto& feature : dataset.Features) {
                const ui32 policy = ui32(feature.Policy);
                CB_ENSURE(policy < 3 && (dataset.PolicyMask & (1u << policy)) &&
                    feature.RuntimeFeature < runtimeFeatures && !assigned[feature.RuntimeFeature],
                    "Invalid or repeated Metal meta-L2 runtime feature mapping");
                assigned[feature.RuntimeFeature] = 1;
                result[feature.RuntimeFeature] = selected[policy];
            }
        }
        return result;
    }

    inline TVector<ui8> BuildMetalMetaL2FeatureChoices(ui32 runtimeFeatures,
            double exponent, double frequency, TConstArrayRef<TMetalMetaL2DataSet> datasets) {
        CB_ENSURE(std::isfinite(exponent) && std::isfinite(float(exponent)) && std::isfinite(frequency),
            "Invalid Metal meta-L2 parameters");
        TVector<ui8> result(runtimeFeatures, 0);
        for (const auto& dataset : datasets) {
            const auto selected = MetalMetaL2PolicyExponents(dataset.ScoreSeed, dataset.PolicyMask, exponent, frequency);
            for (const auto& feature : dataset.Features) {
                const ui32 policy = ui32(feature.Policy);
                CB_ENSURE(policy < 3 && (dataset.PolicyMask & (1u << policy)) && feature.RuntimeFeature < runtimeFeatures,
                    "Invalid Metal meta-L2 runtime feature mapping");
                // The same CTR may be scored from several base-tensor packs.
                // Preserve both complete candidate scores in that situation.
                result[feature.RuntimeFeature] |= selected[policy] == 1.0f ? 1u : 2u;
            }
        }
        for (auto& value : result) if (!value) value = 1;
        return result;
    }
}
