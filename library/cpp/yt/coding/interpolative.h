#pragma once

#include "bit_io.h"

#include <library/cpp/yt/memory/range.h>

#include <util/system/types.h>

#include <concepts>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////
//
// Binary interpolative coding for a sorted, strictly increasing sequence of
// integers drawn from a known range [lo, hi]. The value domain is 32-bit: lo, hi
// and hence every value fit in ui32, independent of the element type T (which is
// merely the container's width).
//
// It recursively encodes the median element within the range implied by its
// position and its already-coded neighbors, so clustered sequences compress far
// below a flat log2 per element and no per-element headers are needed. Each
// element is stored with a truncated-binary (minimal) code, which spends the
// fractional part of log2(range) instead of rounding every element up to a whole
// bit.
//
// The bit stream is MSB-first (see bit_io.h).
//
////////////////////////////////////////////////////////////////////////////////

//! Encodes #values, which must be strictly increasing and all within [#lo, #hi],
//! with binary interpolative coding. An empty range emits nothing; the length is
//! not stored and must be conveyed out of band (e.g. as a varint prefix).
template <std::unsigned_integral T>
void InterpolativeEncode(TBitWriter* writer, TRange<T> values, ui32 lo, ui32 hi);

//! Decodes a sequence written by #InterpolativeEncode into #values, whose size
//! must equal the encoded element count. #lo and #hi must match the encoder.
template <std::unsigned_integral T>
void InterpolativeDecode(TBitReader* reader, TMutableRange<T> values, ui32 lo, ui32 hi);

//! An upper bound on the buffer size #InterpolativeEncode needs to encode #count
//! values over [#lo, #hi], including the slack the writer requires.
size_t GetInterpolativeMaxByteSize(int count, ui32 lo, ui32 hi);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define INTERPOLATIVE_INL_H_
#include "interpolative-inl.h"
#undef INTERPOLATIVE_INL_H_
