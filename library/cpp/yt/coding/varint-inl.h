#ifndef VARINT_INL_H_
#error "Direct inclusion of this file is not allowed, include varint.h"
// For the sake of sane code completion.
#include "varint.h"
#endif

#include "zig_zag.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class TWriteCallback>
Y_FORCE_INLINE int WriteVarUint64Impl(TWriteCallback doWrite, ui64 value)
{
    bool stop = false;
    int bytesWritten = 0;
    while (!stop) {
        ++bytesWritten;
        ui8 byte = static_cast<ui8>(value | 0x80);
        value >>= 7;
        if (value == 0) {
            stop = true;
            byte &= 0x7F;
        }
        doWrite(byte);
    }
    return bytesWritten;
}

// These are optimized versions of these Read/Write functions in protobuf/io/coded_stream.cc.
Y_FORCE_INLINE int WriteVarUint64(IOutputStream* output, ui64 value)
{
    return WriteVarUint64Impl([&] (ui8 byte) {
        output->Write(byte);
    }, value);
}

Y_FORCE_INLINE int WriteVarUint64(char* output, ui64 value)
{
    return WriteVarUint64Impl([&] (ui8 byte) {
        *output++ = byte;
    }, value);
}

////////////////////////////////////////////////////////////////////////////////

template <class TOutput>
Y_FORCE_INLINE int WriteVarUint32Impl(TOutput output, ui32 value)
{
    return WriteVarUint64(output, static_cast<ui64>(value));
}

Y_FORCE_INLINE int WriteVarUint32(IOutputStream* output, ui32 value)
{
    return WriteVarUint32Impl(output, value);
}

Y_FORCE_INLINE int WriteVarUint32(char* output, ui32 value)
{
    return WriteVarUint32Impl(output, value);
}

////////////////////////////////////////////////////////////////////////////////

template <class TOutput>
Y_FORCE_INLINE int WriteVarInt32Impl(TOutput output, i32 value)
{
    return WriteVarUint64(output, static_cast<ui64>(ZigZagEncode32(value)));
}

Y_FORCE_INLINE int WriteVarInt32(IOutputStream* output, i32 value)
{
    return WriteVarInt32Impl(output, value);
}

Y_FORCE_INLINE int WriteVarInt32(char* output, i32 value)
{
    return WriteVarInt32Impl(output, value);
}

////////////////////////////////////////////////////////////////////////////////

template <class TOutput>
Y_FORCE_INLINE int WriteVarInt64Impl(TOutput output, i64 value)
{
    return WriteVarUint64(output, static_cast<ui64>(ZigZagEncode64(value)));
}

Y_FORCE_INLINE int WriteVarInt64(IOutputStream* output, i64 value)
{
    return WriteVarInt64Impl(output, value);
}

Y_FORCE_INLINE int WriteVarInt64(char* output, i64 value)
{
    return WriteVarInt64Impl(output, value);
}

////////////////////////////////////////////////////////////////////////////////

template <class TReadCallback>
Y_FORCE_INLINE int ReadVarUint64Impl(TReadCallback doRead, ui64* value)
{
    size_t count = 0;
    ui64 result = 0;

    ui8 byte;
    do {
        if (7 * count > 8 * sizeof(ui64) ) {
            throw TSimpleException("Value is too big for varuint64");
        }
        byte = doRead();
        result |= (static_cast<ui64> (byte & 0x7F)) << (7 * count);
        ++count;
    } while (byte & 0x80);

    *value = result;
    return count;
}

Y_FORCE_INLINE int ReadVarUint64(IInputStream* input, ui64* value)
{
    return ReadVarUint64Impl([&] () {
        char byte;
        if (input->Read(&byte, 1) != 1) {
            throw TSimpleException("Premature end of stream while reading varuint64");
        }
        return byte;
    }, value);
}

Y_FORCE_INLINE int ReadVarUint64(const char* input, ui64* value)
{
    return ReadVarUint64Impl([&] () {
        char byte = *input;
        ++input;
        return byte;
    }, value);
}

Y_FORCE_INLINE int ReadVarUint64(const char* input, const char* end, ui64* value)
{
    return ReadVarUint64Impl([&] () {
        if (input == end) {
            throw TSimpleException("Premature end of data while reading varuint64");
        }
        char byte = *input;
        ++input;
        return byte;
    }, value);
}

////////////////////////////////////////////////////////////////////////////////

template <class... Args>
Y_FORCE_INLINE int ReadVarUint32Impl(ui32* value, Args... args)
{
    ui64 varInt;
    int bytesRead = ReadVarUint64(args..., &varInt);
    if (varInt > std::numeric_limits<ui32>::max()) {
        throw TSimpleException("Value is too big for varuint32");
    }
    *value = static_cast<ui32>(varInt);
    return bytesRead;
}

Y_FORCE_INLINE int ReadVarUint32(IInputStream* input, ui32* value)
{
    return ReadVarUint32Impl(value, input);
}

Y_FORCE_INLINE int ReadVarUint32(const char* input, ui32* value)
{
    return ReadVarUint32Impl(value, input);
}

Y_FORCE_INLINE int ReadVarUint32(const char* input, const char* end, ui32* value)
{
    return ReadVarUint32Impl(value, input, end);
}

////////////////////////////////////////////////////////////////////////////////

template <class... Args>
Y_FORCE_INLINE int ReadVarInt32Impl(i32* value, Args... args)
{
    ui64 varInt;
    int bytesRead = ReadVarUint64(args..., &varInt);
    if (varInt > std::numeric_limits<ui32>::max()) {
        throw TSimpleException("Value is too big for varint32");
    }
    *value = ZigZagDecode32(static_cast<ui32>(varInt));
    return bytesRead;
}

Y_FORCE_INLINE int ReadVarInt32(IInputStream* input, i32* value)
{
    return ReadVarInt32Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt32(const char* input, i32* value)
{
    return ReadVarInt32Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt32(const char* input, const char* end, i32* value)
{
    return ReadVarInt32Impl(value, input, end);
}

////////////////////////////////////////////////////////////////////////////////

template <class... Args>
Y_FORCE_INLINE int ReadVarInt64Impl(i64* value, Args... args)
{
    ui64 varInt;
    int bytesRead = ReadVarUint64(args..., &varInt);
    *value = ZigZagDecode64(varInt);
    return bytesRead;
}

Y_FORCE_INLINE int ReadVarInt64(IInputStream* input, i64* value)
{
    return ReadVarInt64Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt64(const char* input, i64* value)
{
    return ReadVarInt64Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt64(const char* input, const char* end, i64* value)
{
    return ReadVarInt64Impl(value, input, end);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
