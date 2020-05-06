#pragma once

#include "types.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/bitops.h>

namespace NTextProcessing::NDictionary {
    constexpr ui32 MAX_BUCKET_SEARCH_STEPS = 1000;
    constexpr ui64 MAX_SEED_CHOICE_COUNT = 10;

    struct TBucket {
        static constexpr ui64 InvalidHashValue = 0xffffffffffffffffull;
        using THashType = ui64;

        THashType Hash = InvalidHashValue;
        TTokenId TokenId = 0;

        bool operator==(const TBucket& other) const {
            return std::tie(Hash, TokenId) == std::tie(other.Hash, other.TokenId);
        }
    };

    inline ui64 GetBucketIndex(
        ui64 hash,
        TConstArrayRef<TBucket> buckets,
        ui32* bucketSearchStepCountPtr = nullptr
    ) {
        Y_ENSURE(!buckets.empty(), "Bucket vector is empty!");
        ui32 bucketSearchStepCount = 0;
        const ui64 hashMask = buckets.size() - 1;
        ui64 index = hash & hashMask;
        for (; buckets[index].Hash != TBucket::InvalidHashValue; index = (index + 1) & hashMask) {
            if (buckets[index].Hash == hash) {
                break;
            }
            ++bucketSearchStepCount;
        }

        if (bucketSearchStepCountPtr) {
            *bucketSearchStepCountPtr = bucketSearchStepCount;
        }

        return index;
    }

    inline ui32 ComputeCorrectBucketCount(ui32 elementCount) {
        return elementCount != 0 ? FastClp2(elementCount) * 2 : 1;
    }

    template <typename TElementRange, typename TExctractor>
    void BuildBuckets(
        const TElementRange& range,
        const TExctractor& getTokenInfo,
        TVector<TBucket>* buckets,
        ui64* seed
    ) {
        const ui32 bucketCount = ComputeCorrectBucketCount(range.size());
        *seed = 0;
        buckets->yresize(bucketCount);
        for (; *seed < MAX_SEED_CHOICE_COUNT; ++(*seed)) {
            bool isBucketSearchStepLimitExceeded = false;
            std::fill(buckets->begin(), buckets->end(), TBucket());
            for (const auto& element : range) {
                const auto hashAndTokenId = getTokenInfo(element, *seed);
                ui32 bucketSearchStepCount;
                auto index = GetBucketIndex(hashAndTokenId.first, *buckets, &bucketSearchStepCount);
                (*buckets)[index] = {hashAndTokenId.first, hashAndTokenId.second};
                isBucketSearchStepLimitExceeded |= bucketSearchStepCount > MAX_BUCKET_SEARCH_STEPS;
            }

            if (!isBucketSearchStepLimitExceeded) {
                return;
            }
        }

        Y_ENSURE(false, "Couldn't find a mapping without collisions.");
    }
}
