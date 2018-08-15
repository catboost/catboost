#pragma once

#include "exception.h"
#include "index_range.h"

#include <util/generic/array_ref.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <cstdlib>

namespace NCB {

    template <class TSize>
    struct TRangesSubset {
        using TRanges = TVector<TIndexRange<TSize>>;

        TSize Size;
        TRanges Ranges;

    public:
        TRangesSubset(TSize size, TRanges&& ranges)
            : Size(size)
            , Ranges(std::move(ranges))
        {
            Y_ASSERT(CalcSize(Ranges) == Size);
        }

        explicit TRangesSubset(TRanges&& ranges)
            : TRangesSubset(CalcSize(ranges), std::move(ranges))
        {}

    private:
        static TSize CalcSize(const TRanges& ranges) {
            TSize size = 0;
            for (const auto& range : ranges) {
                size += range.Size();
            }
            return size;
        }
    };

    template <class TSize>
    using TIndexedSubset = TVector<TSize>; // index in src data

    template <class TSize>
    struct TFullSubset {
        TSize Size;

    public:
        explicit TFullSubset(TSize size)
            : Size(size)
        {}
    };

    template <class TSize>
    class TArraySubsetIndexing
        : public TVariant<TFullSubset<TSize>, TRangesSubset<TSize>, TIndexedSubset<TSize>>
    {
        using TBase = TVariant<TFullSubset<TSize>, TRangesSubset<TSize>, TIndexedSubset<TSize>>;

    public:
        template<class T>
        explicit TArraySubsetIndexing(T&& subsetIndexingVariant)
            : TBase(std::move(subsetIndexingVariant))
        {}

        TSize Size() const {
            switch (TBase::Index()) {
                case TBase::template TagOf<TFullSubset<TSize>>():
                    return Get<TFullSubset<TSize>>().Size;
                case TBase::template TagOf<TRangesSubset<TSize>>():
                    return Get<TRangesSubset<TSize>>().Size;
                case TBase::template TagOf<TIndexedSubset<TSize>>():
                    return static_cast<TSize>(Get<TIndexedSubset<TSize>>().size());
            }
            return 0; // just to silence compiler warnings
        }

        // Had to redefine Get because automatic resolution does not work with current TVariant implementation
        template <class T>
        decltype(auto) Get() {
            return ::Get<T>((TBase&)*this);
        }

        template <class T>
        decltype(auto) Get() const {
            return ::Get<T>((const TBase&)*this);
        }
    };


    // TArrayLike must have O(1) random-access operator[].
    template <class TArrayLike, class TSize = size_t>
    class TArraySubset {
    public:
        TArraySubset(TArrayLike* src, const TArraySubsetIndexing<TSize>* subsetIndexing)
            : Src(src)
            , SubsetIndexing(subsetIndexing)
        {
            CB_ENSURE(Src, "TArraySubset constructor: src argument is nullptr");
            CB_ENSURE(SubsetIndexing, "TArraySubset constructor: subsetIndexing argument is nullptr");
        }

        TSize Size() const {
            return SubsetIndexing->Size();
        }

        // f is a visitor function that will be repeatedly called with (index, element) arguments
        template<class F>
        void ForEach(F&& f) {
            switch (SubsetIndexing->Index()) {
                case TArraySubsetIndexing<TSize>::template TagOf<TFullSubset<TSize>>():
                    {
                        for (TSize index : xrange<TSize>(SubsetIndexing->template Get<TFullSubset<TSize>>().Size)) {
                            f(index, (*Src)[index]);
                        }
                    }
                    break;
                case TArraySubsetIndexing<TSize>::template TagOf<TRangesSubset<TSize>>():
                    {
                        TSize index = 0;
                        const auto& ranges = SubsetIndexing->template Get<TRangesSubset<TSize>>().Ranges;
                        for (const auto& range : ranges) {
                            for (TSize srcIndex = range.Begin; srcIndex != range.End; ++srcIndex, ++index) {
                                f(index, (*Src)[srcIndex]);
                            }
                        }
                    }
                    break;
                case TArraySubsetIndexing<TSize>::template TagOf<TIndexedSubset<TSize>>():
                    {
                        const auto& indexView = SubsetIndexing->template Get<TIndexedSubset<TSize>>();
                        for (TSize index : xrange<TSize>(indexView.size())) {
                            f(index, (*Src)[indexView[index]]);
                        }
                    }
                    break;
            }
        }
    private:
        TArrayLike* Src;
        const TArraySubsetIndexing<TSize>* SubsetIndexing;
    };
}

