#pragma once

#include "data_provider.h"
#include "data_utils.h"
#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/random/shuffle.h>
#include <numeric>

namespace NCatboostCuda {
    class TDataPermutation {
    private:
        const TDataProvider* DataProvider;
        ui32 Index;
        ui32 BlockSize;

    public:
        TDataPermutation(const TDataProvider& dataProvider,
                         ui32 index,
                         ui32 blockSize)
            : DataProvider(&dataProvider)
            , Index(index)
            , BlockSize(blockSize)
        {
        }

        TDataPermutation(const TDataPermutation& other) = default;
        TDataPermutation(TDataPermutation&& other) = default;

        void FillOrder(TVector<ui32>& order) const {
            if (Index != IdentityPermutationId()) {
                const auto seed = 1664525 * GetPermutationId() + 1013904223 + BlockSize;
                if (DataProvider->HasQueries()) {
                    QueryConsistentShuffle(seed, BlockSize, DataProvider->GetQueryIds(), &order);
                } else {
                    Shuffle(seed, BlockSize, DataProvider->GetSampleCount(), &order);
                }
            } else {
                order.resize(DataProvider->GetSampleCount());
                std::iota(order.begin(), order.end(), 0);
            }
        }

        template <class T>
        TVector<T> Gather(const TVector<T>& src) const {
            TVector<T> result;
            result.resize(src.size());

            TVector<ui32> order;
            FillOrder(order);
            for (ui32 i = 0; i < order.size(); ++i) {
                result[i] = src[order[i]];
            }
            return result;
        }

        void FillInversePermutation(TVector<ui32>& permutation) const {
            TVector<ui32> order;
            FillOrder(order);
            permutation.resize(order.size());
            for (ui32 i = 0; i < order.size(); ++i) {
                permutation[order[i]] = i;
            }
        }

        template <class TBuffer>
        void WriteOrder(TBuffer& dst) const {
            TVector<ui32> order;
            FillOrder(order);
            dst.Write(order);
        }

        template <class TBuffer>
        void WriteInversePermutation(TBuffer& dst) const {
            TVector<ui32> order;
            FillInversePermutation(order);
            dst.Write(order);
        }

        bool IsIdentity() const {
            return Index == IdentityPermutationId();
        }

        static constexpr ui32 IdentityPermutationId() {
            return 0;
        }

        ui32 GetPermutationId() const {
            return Index;
        }

        ui64 GetDocCount() const {
            return DataProvider->GetSampleCount();
        }
    };

    inline TDataPermutation GetPermutation(const TDataProvider& dataProvider,
                                           ui32 permutationId,
                                           ui32 blockSize = 1) {
        return TDataPermutation(dataProvider,
                                permutationId,
                                blockSize);
    }

    inline TDataPermutation GetIdentityPermutation(const TDataProvider& dataProvider) {
        return GetPermutation(dataProvider,
                              TDataPermutation::IdentityPermutationId(),
                              1);
    }

}
