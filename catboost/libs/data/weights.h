#pragma once

#include "util.h"

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/serialization.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/system/types.h>
#include <util/system/yassert.h>


namespace NCB {


    template <class T = float>
    void CheckWeights(
        TConstArrayRef<T> weights,
        ui32 objectCount,
        const TStringBuf weightTypeName,
        bool dataCanBeEmpty = false,
        bool allWeightsCanBeZero = false
    ) {
        CheckDataSize(weights.size(), (size_t)objectCount, weightTypeName, dataCanBeEmpty);

        if ((objectCount > 0) && (weights.size() > 0)) {
            bool hasNonZeroWeights = false;
            for (auto i : xrange(weights.size())) {
                if (weights[i] > T(0)) {
                    hasNonZeroWeights = true;
                } else {
                    CB_ENSURE(weights[i] >= T(0), "" << weightTypeName << '[' << i << "] is negative");
                }
            }
            CB_ENSURE(
                allWeightsCanBeZero || hasNonZeroWeights,
                "All data in " << weightTypeName << " is 0"
            );
        }
    }

    // empty data is assumed trivial
    template <class T = float>
    bool AreWeightsTrivial(TConstArrayRef<T> weights) {
        return FindIf(weights, [](T value) { return value != T(1); }) == weights.end();
    }

    // needed for optimization - we don't want to store all weights if all of them are equal to 1

    template <class T=float>
    class TWeights : public TThrRefBase {
    public:
        // need postponed init in some cases
        TWeights()
            : Size(0)
            , Weights(TMaybeOwningArrayHolder<T>::CreateNonOwning(TArrayRef<T>()))
        {}

        // pass empty array in weights argument if weights are trivial
        explicit TWeights(
            ui32 size,
            TMaybeOwningArrayHolder<T> weights = TMaybeOwningArrayHolder<T>::CreateNonOwning(TArrayRef<T>()),
            bool skipCheck = false,
            const TStringBuf weightTypeName = "Weight", // for error messages in check
            bool allWeightsCanBeZero = false
        )
            : Size(size)
            , Weights(std::move(weights))
        {
            if (!skipCheck) {
                CheckWeights(TConstArrayRef<T>(*Weights), Size, weightTypeName, true, allWeightsCanBeZero);
            }
        }

        /* use this constructor when weights are non-trivial,
         * empty 'weights' argument's size means empty data size, not trivial data weights!
         */
        explicit TWeights(
            TVector<T>&& weights,
            const TStringBuf weightTypeName = "Weight",
            bool allWeightsCanBeZero = false
        )
            : Size(weights.size())
            , Weights(TMaybeOwningArrayHolder<T>::CreateOwning(std::move(weights)))
        {
            CheckWeights(TConstArrayRef<T>(*Weights), Size, weightTypeName, false, allWeightsCanBeZero);
        }


        T operator[] (ui32 idx) const {
            Y_ASSERT(idx < Size);
            return IsTrivial() ? T(1) : Weights[idx];
        }

        bool operator ==(const TWeights& lhs) const {
            if (IsTrivial()) {
                if (lhs.IsTrivial()) {
                    return Size == lhs.Size;
                }
                return AreWeightsTrivial(lhs.GetNonTrivialData());
            }
            if (lhs.IsTrivial()) {
                return AreWeightsTrivial(GetNonTrivialData());
            }
            return GetNonTrivialData() == lhs.GetNonTrivialData();
        }

        SAVELOAD(Size, Weights);

        ui32 GetSize() const noexcept {
            return Size;
        }

        // sometimes it's more optimal to check once per all array instead of repeated checks in operator[]
        bool IsTrivial() const noexcept {
            return (*Weights).empty();
        }


        // sometimes it's more optimal to check once per all array instead of repeated checks in operator[]
        // make sure to call IsTrivial first
        TConstArrayRef<T> GetNonTrivialData() const {
            CB_ENSURE(!IsTrivial(), "Data is trivial");
            return *Weights;
        }

        TWeights GetSubset(
            const TArraySubsetIndexing<ui32>& subset,
            NPar::ILocalExecutor* localExecutor
        ) const {
            if (IsTrivial()) {
                return TWeights(subset.Size());
            } else {
                return TWeights(
                    subset.Size(),
                    TMaybeOwningArrayHolder<T>::CreateOwning(
                        NCB::GetSubset<T>(*Weights, subset, localExecutor)
                    ),
                    true
                );
            }
        }

    private:
        ui32 Size;
        TMaybeOwningArrayHolder<T> Weights; // if empty - means weights are trivial
    };

    template <class T=float>
    using TSharedWeights = TIntrusivePtr<TWeights<T>>;

}
