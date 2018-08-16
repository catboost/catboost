#pragma once

#include "exception.h"

#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

namespace NCB {

    // represents index range to process: [Begin, End)
    template <class TSize>
    struct TIndexRange {
        TSize Begin;
        TSize End;

        explicit TIndexRange(TSize end)
            : TIndexRange(TSize(0), end)
        {}

        TIndexRange(TSize begin, TSize end)
            : Begin(begin)
            , End(end)
        {
            CB_ENSURE(End >= Begin,
                      "TIndexRange::TIndexRange : begin (" << begin << ") > end (" << end << ")");
        }

        bool Empty() const {
            return Begin == End;
        }

        TSize Size() const {
            return End - Begin;
        }

        // support for range-based for loop
        constexpr auto Iter() const {
            return xrange(Begin, End);
        }
    };

    template <class TSize>
    struct IIndexRangesGenerator {
        virtual ~IIndexRangesGenerator() = default;

        virtual TSize RangesCount() const = 0;

        virtual NCB::TIndexRange<TSize> GetRange(TSize idx) const = 0;
    };

    template <class TSize>
    class TSimpleIndexRangesGenerator : public IIndexRangesGenerator<TSize> {
    public:
        TSimpleIndexRangesGenerator(NCB::TIndexRange<TSize> fullRange, TSize blockSize)
            : FullRange(fullRange)
            , BlockSize(blockSize)
        {}

        TSize RangesCount() const override {
            return CeilDiv(FullRange.Size(), BlockSize);
        }

        NCB::TIndexRange<TSize> GetRange(TSize idx) const override {
            Y_ASSERT(idx < RangesCount());
            TSize blockBeginIdx = FullRange.Begin + idx*BlockSize;
            TSize blockEndIdx = Min(blockBeginIdx + BlockSize, FullRange.End);
            return NCB::TIndexRange<TSize>(blockBeginIdx, blockEndIdx);
        }

    private:
        NCB::TIndexRange<TSize> FullRange;
        TSize BlockSize;
    };

    template <class TSize>
    class TSavedIndexRanges : public NCB::IIndexRangesGenerator<TSize> {
    public:
        explicit TSavedIndexRanges(TVector<NCB::TIndexRange<TSize>>&& indexRanges)
            : IndexRanges(std::move(indexRanges))
        {}

        TSize RangesCount() const override {
            return (TSize)IndexRanges.size();
        }

        NCB::TIndexRange<TSize> GetRange(TSize idx) const override {
            return IndexRanges[idx];
        }

    private:
        TVector<NCB::TIndexRange<TSize>> IndexRanges;
    };

}
