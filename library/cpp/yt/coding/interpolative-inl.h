#ifndef INTERPOLATIVE_INL_H_
#error "Direct inclusion of this file is not allowed, include interpolative.h"
// For the sake of sane code completion.
#include "interpolative.h"
#endif

#include "bit_io.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/memory/range.h>

#include <util/system/compiler.h>

#include <array>
#include <bit>
#include <concepts>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NInterpolativeCodingDetail {

// Truncated-binary (minimal) code for a value in [0, rangeSize): the first
// Cutoff values take LowWidth bits, the rest one more.
struct TTruncatedBinaryParams
{
    int LowWidth;   // floor(log2(rangeSize))
    ui32 Cutoff;    // 2^(LowWidth + 1) - rangeSize
};

Y_FORCE_INLINE TTruncatedBinaryParams GetTruncatedBinaryParams(ui32 rangeSize)
{
    int lowWidth = std::bit_width(rangeSize) - 1;
    ui32 cutoff = (2u << lowWidth) - rangeSize;
    return {lowWidth, cutoff};
}

// Writes #value in [0, #rangeSize) with the entropy-optimal integer code for a
// uniform value. rangeSize == 1 yields lowWidth 0 / cutoff 1 and emits a
// zero-width code (a no-op), so the singleton case needs no branch.
Y_FORCE_INLINE void WriteTruncatedBinary(TBitWriter* writer, ui32 value, ui32 rangeSize)
{
    auto [lowWidth, cutoff] = GetTruncatedBinaryParams(rangeSize);
    // Branchless: the (value >= cutoff) predicate would mispredict on nearly
    // every element, so fold it into the emitted codeword and width instead.
    ui32 isLong = value >= cutoff ? 1 : 0;
    writer->WriteBits(value + (cutoff & (0u - isLong)), lowWidth + static_cast<int>(isLong));
}

Y_FORCE_INLINE ui32 ReadTruncatedBinary(TBitReader* reader, ui32 rangeSize)
{
    auto [lowWidth, cutoff] = GetTruncatedBinaryParams(rangeSize);
    ui32 high = reader->ReadBits(lowWidth);
    if (high < cutoff) {
        return high;
    }
    ui32 low = reader->ReadBits(1);
    return ((high << 1) | low) - cutoff;
}

// A subrange [BeginIndex, EndIndex) of the value array together with the
// inclusive value bounds [Lo, Hi] that its elements must fall in. The median
// values[m] is the only array element touched per node, which matters because
// the traversal walks a large sequence in a cache-unfriendly tree order.
struct TInterpolativeFrame
{
    int BeginIndex;
    int EndIndex;
    ui32 Lo;
    ui32 Hi;
};

} // namespace NInterpolativeCodingDetail

////////////////////////////////////////////////////////////////////////////////

inline size_t GetInterpolativeMaxByteSize(int count, ui32 lo, ui32 hi)
{
    // Each element is coded in at most ceil(log2(hi - lo + 1)) bits; the trailing
    // word covers the writer's 4-byte flush store.
    int maxBitWidth = std::bit_width(hi - lo);
    return (static_cast<size_t>(count) * maxBitWidth + 7) / 8 + sizeof(ui32);
}

////////////////////////////////////////////////////////////////////////////////

// Both traversals descend the left child in place and stack only the (non-empty)
// right child, so the stack sees ~count/2 pushes instead of ~count. Left-first
// order matches between encoder and decoder.
template <std::unsigned_integral T>
void InterpolativeEncode(TBitWriter* writer, TRange<T> values, ui32 lo, ui32 hi)
{
    using namespace NInterpolativeCodingDetail;

    int count = std::ssize(values);
    if (count == 0) {
        return;
    }

    // Right children stack up along the leftmost path => depth <= ceil(log2(count)).
    std::array<TInterpolativeFrame, 48> stack;
    int top = 0;
    int beginIndex = 0;
    int endIndex = count;
    ui32 l = lo;
    ui32 h = hi;
    for (;;) {
        // beginIndex is invariant while descending left; only endIndex/l/h change.
        while (beginIndex < endIndex) {
            int half = (endIndex - beginIndex) / 2;
            int m = beginIndex + half;
            // The next descent step reads values[beginIndex + half/2]; prefetch it
            // to hide the cache-scattered tree walk on large sequences.
            Y_PREFETCH_READ(values.data() + beginIndex + (half >> 1), 0);
            ui32 value = static_cast<ui32>(values[m]);
            // rangeSize = (upperBound - lowerBound + 1) simplifies to this;
            // lowerBound = l + half.
            ui32 rangeSize = (h - l) - static_cast<ui32>(endIndex - beginIndex) + 2;
            WriteTruncatedBinary(writer, value - l - static_cast<ui32>(half), rangeSize);
            if (m + 1 < endIndex) {
                YT_ASSERT(top < std::ssize(stack));
                stack[top++] = {m + 1, endIndex, value + 1, h};
            }
            endIndex = m;
            h = value - 1;
        }
        if (top == 0) {
            break;
        }
        auto f = stack[--top];
        beginIndex = f.BeginIndex;
        endIndex = f.EndIndex;
        l = f.Lo;
        h = f.Hi;
    }
}

template <std::unsigned_integral T>
void InterpolativeDecode(TBitReader* reader, TMutableRange<T> values, ui32 lo, ui32 hi)
{
    using namespace NInterpolativeCodingDetail;

    int count = std::ssize(values);
    if (count == 0) {
        return;
    }

    std::array<TInterpolativeFrame, 48> stack;
    int top = 0;
    int beginIndex = 0;
    int endIndex = count;
    ui32 l = lo;
    ui32 h = hi;
    for (;;) {
        while (beginIndex < endIndex) {
            int half = (endIndex - beginIndex) / 2;
            int m = beginIndex + half;
            ui32 rangeSize = (h - l) - static_cast<ui32>(endIndex - beginIndex) + 2;
            ui32 value = l + static_cast<ui32>(half) + ReadTruncatedBinary(reader, rangeSize);
            values[m] = static_cast<T>(value);
            if (m + 1 < endIndex) {
                stack[top++] = {m + 1, endIndex, value + 1, h};
            }
            endIndex = m;
            h = value - 1;
        }
        if (top == 0) {
            break;
        }
        auto f = stack[--top];
        beginIndex = f.BeginIndex;
        endIndex = f.EndIndex;
        l = f.Lo;
        h = f.Hi;
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
