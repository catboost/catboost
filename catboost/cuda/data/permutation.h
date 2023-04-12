#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/exception.h>

#include <util/system/types.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    class TDataPermutation {
    private:
        const NCB::TTrainingDataProvider* DataProvider;
        ui32 Index;
        ui32 BlockSize;

    public:
        TDataPermutation(const NCB::TTrainingDataProvider& dataProvider,
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
        void FillGroupOrder(TVector<ui32>& queryOrder) const;
        template <class T>
        TVector<T> Gather(TConstArrayRef<T> src) const {
            TVector<T> result;
            result.resize(src.size());

            TVector<ui32> order;
            FillOrder(order);
            for (ui32 i = 0; i < order.size(); ++i) {
                result[i] = src[order[i]];
            }
            return result;
        }

        template <class T>
        TVector<TVector<T>> Gather2D(TConstArrayRef<TConstArrayRef<T>> src) const {
            TVector<TVector<T>> result;
            result.resize(src.size());

            TVector<ui32> order;
            FillOrder(order);
            for (ui32 j = 0; j < src.size(); ++j) {
                result[j].resize(order.size());
                for (ui32 i = 0; i < order.size(); ++i) {
                    result[j][i] = src[j][order[i]];
                }
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
            return DataProvider->GetObjectCount();
        }
    private:
        ui64 GetSeed() const {
            return 1664525 * GetPermutationId() + 1013904223 + BlockSize;
        }
    };

    inline TDataPermutation GetPermutation(const NCB::TTrainingDataProvider& dataProvider,
                                           ui32 permutationId,
                                           ui32 blockSize = 1) {
        return TDataPermutation(dataProvider,
                                permutationId,
                                blockSize);
    }

    inline TDataPermutation GetIdentityPermutation(const NCB::TTrainingDataProvider& dataProvider) {
        return GetPermutation(dataProvider,
                              TDataPermutation::IdentityPermutationId(),
                              1);
    }

    template <class T>
    inline TVector<T> Scatter(const TVector<ui32>& indices, const TVector<T>& src) {
        CB_ENSURE(indices.size() == src.size(), "THIS IS A BUG, report to catboost team: Scatter indices count should be equal to scattered data size");

        TVector<T> result;
        result.resize(src.size());

        for (ui32 i = 0; i < indices.size(); ++i) {
            result[indices[i]] = src[i];
        }
        return result;
    }

}
