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

        void FillOrder(TVector<ui32>& order) const;
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

        void FillInversePermutation(TVector<ui32>& permutation) const;

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


    template <class T>
    inline TVector<T> Scatter(const TVector<ui32>& indices, const TVector<T>& src)  {
        CB_ENSURE(indices.size() == src.size(), "THIS IS A BUG, report to catboost team: Scatter indices count should be equal to scattered data size");

        TVector<T> result;
        result.resize(src.size());

        for (ui32 i = 0; i < indices.size(); ++i) {
            result[indices[i]] = src[i];
        }
        return result;
    }

}
