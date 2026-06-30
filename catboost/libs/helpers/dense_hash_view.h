#pragma once

#include "exception.h"

#include <util/digest/numeric.h>
#include <util/generic/array_ref.h>
#include <util/generic/algorithm.h>
#include <util/generic/bitops.h>

namespace NCatboost {

#pragma pack(push, 1)
    struct TBucket {
        static constexpr ui64 InvalidHashValue = 0xffffffffffffffffull;
        using THashType = ui64;

    public:
        THashType Hash;
        ui32 IndexValue;

    public:
        bool operator==(const TBucket& other) const {
            return std::tie(Hash, IndexValue) == std::tie(other.Hash, other.IndexValue);
        }
    };
#pragma pack(pop)

    // Special optimized classes for building and applying non resizeable indexes [ui64] -> [ui32]
    // It's only use case - to be stored on disk as part of catboost flatbuffer model
    class TDenseIndexHashView {
    public:
        static_assert(sizeof(TBucket) == 12, "Expected sizeof(TBucket) == 12 bytes");
        static constexpr ui32 NotFoundIndex = 0xffffffffu;
        static_assert(std::is_pod<TBucket>::value, "must be pod");

    public:
        explicit TDenseIndexHashView(TConstArrayRef<TBucket> bucketsRef)
            : HashMask(bucketsRef.size() - 1)
            , Buckets(bucketsRef)
        {
            CB_ENSURE(IsPowerOf2(bucketsRef.size()), "Dense hash view must have 2^k buckets");
        }

        size_t GetBucketCount() const {
            return Buckets.size();
        }

        ui32 GetIndex(ui64 idx) const {
            for (ui64 zz = idx & HashMask;
                 Buckets[zz].Hash != TBucket::InvalidHashValue;
                 zz = (zz + 1) & HashMask)
            {
                if (Buckets[zz].Hash == idx) {
                    return Buckets[zz].IndexValue;
                }
            }
            return NotFoundIndex;
        }

        size_t CountNonEmptyBuckets() const {
            return CountIf(
                Buckets,
                [](const TBucket& bucket) { return bucket.Hash != TBucket::InvalidHashValue; });
        }

        const TConstArrayRef<TBucket> GetBuckets() const {
            return Buckets;
        }
    private:
        ui64 HashMask = 0;
        TConstArrayRef<TBucket> Buckets;
    };

    class TDenseIndexHashBuilder {
    public:
        static_assert(sizeof(TBucket) == 12, "Expected sizeof(TBucket) == 12 bytes");

    public:
        explicit TDenseIndexHashBuilder(TArrayRef<TBucket> bucketsRef)
            : HashMask(bucketsRef.size() - 1)
            , Buckets(bucketsRef)
        {
            CB_ENSURE(IsPowerOf2(bucketsRef.size()), "Dense hash view must have 2^k buckets");
            TBucket emptyBucket = {TBucket::InvalidHashValue, 0};
            std::fill(bucketsRef.begin(), bucketsRef.end(), emptyBucket);
        }

        ui32 AddIndex(ui64 hash) {
            ui64 zz = hash & HashMask;
            for (; Buckets[zz].Hash != TBucket::InvalidHashValue; zz = (zz + 1) & HashMask) {
                if (Buckets[zz].Hash == hash) {
                    return Buckets[zz].IndexValue;
                }
            }
            Buckets[zz].Hash = hash;
            Buckets[zz].IndexValue = BinCount++;
            return BinCount - 1;
        }

        void SetIndex(ui64 hash, ui32 index) {
            ui64 zz = hash & HashMask;
            for (; Buckets[zz].Hash != TBucket::InvalidHashValue; zz = (zz + 1) & HashMask) {
                if (Buckets[zz].Hash == hash) {
                    Y_ASSERT(Buckets[zz].IndexValue == index);
                    return;
                }
            }
            Buckets[zz].Hash = hash;
            Buckets[zz].IndexValue = index;
            BinCount = Max(BinCount, index );
        }
        static size_t GetProperBucketsCount(size_t uniqueElementsCount, float loadFactor = 0.5f) {
            if (uniqueElementsCount == 0) {
                return 2;
            }
            return FastClp2(static_cast<size_t>(uniqueElementsCount / loadFactor));
        }

    private:
        ui64 HashMask = 0;
        ui32 BinCount = 0;
        TArrayRef<TBucket> Buckets;
    };
}
