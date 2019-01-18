#pragma once

#include <util/system/types.h>
#include <util/digest/multi.h>
#include <util/generic/hash.h>

namespace NCatboostCuda {
    enum class ELoadFromCompressedIndexPolicy {
        GatherBins,     /* for multiclass */
        LoadByIndexBins /* for 2 stat trees */
    };

    struct TComputeByBlocksConfig {
        double SampleRate = 1.0;
        bool ForceOneBlockPerPolicy = false;
        ui32 StreamCount = 1;
        //TODO(noxoomo): get rid of this, used in tests only currently
        ELoadFromCompressedIndexPolicy LoadPolicyAfterSplit = ELoadFromCompressedIndexPolicy::GatherBins;
        ELoadFromCompressedIndexPolicy LoadPolicyFromScratch = ELoadFromCompressedIndexPolicy::LoadByIndexBins;

        bool operator==(const TComputeByBlocksConfig& rhs) const {
            return std::tie(SampleRate, LoadPolicyAfterSplit, LoadPolicyFromScratch,
                            ForceOneBlockPerPolicy) ==
                   std::tie(rhs.SampleRate, rhs.LoadPolicyAfterSplit, rhs.LoadPolicyFromScratch,
                            rhs.ForceOneBlockPerPolicy);
        }

        bool operator!=(const TComputeByBlocksConfig& rhs) const {
            return !(rhs == *this);
        }

        size_t GetHash() const {
            return MultiHash(SampleRate, LoadPolicyAfterSplit, LoadPolicyFromScratch, ForceOneBlockPerPolicy);
        }
    };

    inline constexpr ui32 ComputeHistogramsStreamCount() {
        return 2;
    }
}

template <>
struct THash<NCatboostCuda::TComputeByBlocksConfig> {
    inline size_t operator()(const NCatboostCuda::TComputeByBlocksConfig& value) const {
        return value.GetHash();
    }
};
