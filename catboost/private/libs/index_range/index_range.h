#pragma once

#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>

#include <library/cpp/binsaver/bin_saver.h>

// TODO(akhropov): move back to libs/helpers when circular dependencies with private/libs/data_types are resolved

namespace NCB {

    // represents index range to process: [Begin, End)
    template <class TSize>
    struct TIndexRange {
        TSize Begin = 0;
        TSize End = 0;

    public:
        // for BinSaver
        TIndexRange() = default;

        explicit TIndexRange(TSize end)
            : TIndexRange(TSize(0), end)
        {}

        TIndexRange(TSize begin, TSize end)
            : Begin(begin)
            , End(end)
        {
            Y_ASSERT(End >= Begin);
        }

        bool Empty() const {
            return Begin == End;
        }

        TSize GetSize() const {
            Y_ASSERT(End >= Begin);
            return End - Begin;
        }

        bool operator==(const TIndexRange& rhs) const {
            return (Begin == rhs.Begin) && (End == rhs.End);
        }

        // support for range-based for loop
        constexpr auto Iter() const {
            return xrange(Begin, End);
        }

        bool Contains(TSize idx) const {
            return (idx >= Begin) && (idx < End);
        }

        void ConvexHull(TIndexRange rhs) {
            if (rhs.Empty()) {
                return;
            } else if (Empty()) {
                Begin = rhs.Begin;
                End = rhs.End;
            } else {
                Begin = Min(Begin, rhs.Begin);
                End = Max(End, rhs.End);
            }
        }

        SAVELOAD(Begin, End);
        Y_SAVELOAD_DEFINE(Begin, End);
    };

    template <class TSize>
    static inline IOutputStream& operator<<(IOutputStream& o, const TIndexRange<TSize>& indexRange) {
        o << '[' << indexRange.Begin << ',' << indexRange.End << ')';
        return o;
    }


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
            return CeilDiv(FullRange.GetSize(), BlockSize);
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

    template <class TSize>
    class TEqualRangesGenerator : public IIndexRangesGenerator<TSize> {
    public:
        TEqualRangesGenerator(NCB::TIndexRange<TSize> fullRange, TSize blockCount) {
            TSize begin = fullRange.Begin;
            const TSize size = fullRange.GetSize();
            IndexRanges.reserve(blockCount);
            for (TSize i = 0; i < blockCount; ++i) {
                const TSize currentSize = (size / blockCount) + (i < (size % blockCount));
                IndexRanges.emplace_back(begin, begin + currentSize);
                begin += currentSize;
            }
        }

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
