#pragma once

#include <util/system/defaults.h>

#include <util/stream/input.h>
#include <util/stream/output.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

constexpr size_t MaxVarInt64Size = (8 * sizeof(ui64) - 1) / 7 + 1;
constexpr size_t MaxVarUint64Size = (8 * sizeof(ui64) - 1) / 7 + 1;

constexpr size_t MaxVarInt32Size = (8 * sizeof(ui32) - 1) / 7 + 1;
constexpr size_t MaxVarUint32Size = (8 * sizeof(ui32) - 1) / 7 + 1;

////////////////////////////////////////////////////////////////////////////////

// Various functions to read/write varints.

// Return the number of bytes written.
int WriteVarUint64(IOutputStream* output, ui64 value);
int WriteVarUint32(IOutputStream* output, ui32 value);
int WriteVarInt32(IOutputStream* output, i32 value);
int WriteVarInt64(IOutputStream* output, i64 value);

// Assume that #output has enough capacity.
int WriteVarUint64(char* output, ui64 value);
int WriteVarUint32(char* output, ui32 value);
int WriteVarInt32(char* output, i32 value);
int WriteVarInt64(char* output, i64 value);

// Return the number of bytes read.
int ReadVarUint64(IInputStream* input, ui64* value);
int ReadVarUint32(IInputStream* input, ui32* value);
int ReadVarInt32(IInputStream* input, i32* value);
int ReadVarInt64(IInputStream* input, i64* value);

// Assume that #input contains a valid varint.
int ReadVarUint64(const char* input, ui64* value);
int ReadVarUint32(const char* input, ui32* value);
int ReadVarInt32(const char* input, i32* value);
int ReadVarInt64(const char* input, i64* value);

int ReadVarUint64(const char* input, const char* end, ui64* value);
int ReadVarUint32(const char* input, const char* end, ui32* value);
int ReadVarInt32(const char* input, const char* end, i32* value);
int ReadVarInt64(const char* input, const char* end, i64* value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define VARINT_INL_H_
#include "varint-inl.h"
#undef VARINT_INL_H_
