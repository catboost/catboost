#pragma once

#include "exception.h"

#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

namespace NCB {

    // represents index range to process: [Begin, End)
    struct TIndexRange {
        int Begin;
        int End;

        explicit TIndexRange(int end)
            : TIndexRange(0, end)
        {}

        TIndexRange(int begin, int end)
            : Begin(begin)
            , End(end)
        {
            CB_ENSURE(End >= Begin,
                      "TIndexRange::TIndexRange : begin (" << begin << ") > end (" << end << ")");
        }

        bool Empty() const {
            return Begin == End;
        }

        int Size() const {
            return End - Begin;
        }

        // support for range-based for loop
        constexpr auto Iter() const {
            return xrange(Begin, End);
        }
    };

    struct IIndexRangesGenerator {
        virtual ~IIndexRangesGenerator() = default;

        virtual int RangesCount() const = 0;

        virtual NCB::TIndexRange GetRange(int idx) const = 0;
    };

    class TSimpleIndexRangesGenerator : public IIndexRangesGenerator {
    public:
        TSimpleIndexRangesGenerator(NCB::TIndexRange fullRange, int blockSize)
            : FullRange(fullRange)
            , BlockSize(blockSize)
        {}

        int RangesCount() const override {
            return CeilDiv(FullRange.Size(), BlockSize);
        }

        NCB::TIndexRange GetRange(int idx) const override {
            Y_ASSERT(idx < RangesCount());
            int blockBeginIdx = FullRange.Begin + idx*BlockSize;
            int blockEndIdx = Min(blockBeginIdx + BlockSize, FullRange.End);
            return NCB::TIndexRange(blockBeginIdx, blockEndIdx);
        }

    private:
        NCB::TIndexRange FullRange;
        int BlockSize;
    };

}
