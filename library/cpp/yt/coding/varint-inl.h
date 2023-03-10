#ifndef VARINT_INL_H_
#error "Direct inclusion of this file is not allowed, include varint.h"
// For the sake of sane code completion.
#include "varint.h"
#endif

#include "zig_zag.h"

#include <library/cpp/yt/exception/exception.h>

#include <array>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline int WriteVarUint64(char* output, ui64 value)
{
    output[0] = static_cast<ui8>(value);
    if (Y_LIKELY(value < 0x80)) {
        return 1;
    }
    output[0] |= 0x80;
    value >>= 7;
    output[1] = static_cast<ui8>(value);
    if (Y_LIKELY(value < 0x80)) {
        return 2;
    }
    int count = 2;
    do {
        output[count - 1] |= 0x80;
        value >>= 7;
        output[count] = static_cast<ui8>(value);
        ++count;
    } while (Y_UNLIKELY(value >= 0x80));
    return count;
}

inline int WriteVarUint64(IOutputStream* output, ui64 value)
{
    std::array<char, MaxVarUint64Size> buffer;
    auto size = WriteVarUint64(buffer.data(), value);
    output->Write(buffer.data(), size);
    return size;
}

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T, size_t N>
T ReadVarUintKnownSize(const char* buffer)
{
    auto result = static_cast<T>(static_cast<ui8>(buffer[N - 1])) << (7 * (N - 1));
    for (size_t i = 0, offset = 0; i < N - 1; i++, offset += 7) {
        result += static_cast<T>(static_cast<ui8>(buffer[i]) - 0x80) << offset;
    }
    return result;
}

} // namespace NDetail

#define XX(type, size) \
    if (static_cast<ui8>(input[size - 1]) < 0x80) { \
        *value = NYT::NDetail::ReadVarUintKnownSize<type, size>(input); \
        return size; \
    }

Y_FORCE_INLINE int ReadVarUint32(const char* input, ui32* value)
{
    XX(ui64, 1)
    XX(ui64, 2)
    XX(ui64, 3)
    XX(ui64, 4)
    XX(ui64, 5)
    throw TSimpleException("Value is too big for varuint32");
}

Y_FORCE_INLINE int ReadVarUint64(const char* input, ui64* value)
{
    XX(ui64, 1)
    XX(ui64, 2)
    XX(ui64, 3)
    XX(ui64, 4)
    XX(ui64, 5)
    XX(ui64, 6)
    XX(ui64, 7)
    XX(ui64, 8)
    XX(ui64, 9)
    XX(ui64, 10)
    throw TSimpleException("Value is too big for varuint64");
}

#undef XX

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class TOutput>
Y_FORCE_INLINE int WriteVarUint32Impl(TOutput output, ui32 value)
{
    return WriteVarUint64(output, static_cast<ui64>(value));
}

template <class TOutput>
Y_FORCE_INLINE int WriteVarInt32Impl(TOutput output, i32 value)
{
    return WriteVarUint64(output, ZigZagEncode32(value));
}

template <class TOutput>
Y_FORCE_INLINE int WriteVarInt64Impl(TOutput output, i64 value)
{
    return WriteVarUint64(output, ZigZagEncode64(value));
}

} // namespace NDetail

Y_FORCE_INLINE int WriteVarUint32(IOutputStream* output, ui32 value)
{
    return NYT::NDetail::WriteVarUint32Impl(output, value);
}

Y_FORCE_INLINE int WriteVarUint32(char* output, ui32 value)
{
    return NYT::NDetail::WriteVarUint32Impl(output, value);
}

Y_FORCE_INLINE int WriteVarInt32(IOutputStream* output, i32 value)
{
    return NYT::NDetail::WriteVarInt32Impl(output, value);
}

Y_FORCE_INLINE int WriteVarInt32(char* output, i32 value)
{
    return NYT::NDetail::WriteVarInt32Impl(output, value);
}

Y_FORCE_INLINE int WriteVarInt64(IOutputStream* output, i64 value)
{
    return NYT::NDetail::WriteVarInt64Impl(output, value);
}

Y_FORCE_INLINE int WriteVarInt64(char* output, i64 value)
{
    return NYT::NDetail::WriteVarInt64Impl(output, value);
}

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class TReadCallback>
int ReadVarUint64Impl(ui64* value, TReadCallback&& doRead)
{
    size_t count = 0;
    ui64 result = 0;

    ui8 byte;
    do {
        byte = doRead();
        result |= (static_cast<ui64> (byte & 0x7F)) << (7 * count);
        ++count;
        if (count > MaxVarUint64Size) {
            throw TSimpleException("Value is too big for varuint64");
        }
    } while (byte & 0x80);

    *value = result;
    return count;
}

inline int ReadVarUint64Fallback(const char* input, const char* end, ui64* value)
{
    return ReadVarUint64Impl(
        value,
        [&] {
            if (input == end) {
                throw TSimpleException("Premature end of data while reading varuint64");
            }
            return *input++;
        });
}

} // namespace NDetail

inline int ReadVarUint64(IInputStream* input, ui64* value)
{
    return NYT::NDetail::ReadVarUint64Impl(
        value,
        [&] {
            char byte;
            if (input->Read(&byte, 1) != 1) {
                throw TSimpleException("Premature end of stream while reading varuint64");
            }
            return byte;
        });
}

Y_FORCE_INLINE int ReadVarUint64(const char* input, const char* end, ui64* value)
{
    if (Y_LIKELY(static_cast<size_t>(end - input) >= MaxVarUint64Size)) {
        return ReadVarUint64(input, value);
    } else {
        return NYT::NDetail::ReadVarUint64Fallback(input, end, value);
    }
}

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class... TArgs>
Y_FORCE_INLINE int ReadVarUint32Impl(ui32* value, TArgs&&... args)
{
    ui64 varInt;
    int bytesRead = ReadVarUint64(std::forward<TArgs>(args)..., &varInt);
    if (varInt > std::numeric_limits<ui32>::max()) {
        throw TSimpleException("Value is too big for varuint32");
    }
    *value = static_cast<ui32>(varInt);
    return bytesRead;
}

} // namespace NDetail

Y_FORCE_INLINE int ReadVarUint32(IInputStream* input, ui32* value)
{
    return NYT::NDetail::ReadVarUint32Impl(value, input);
}

Y_FORCE_INLINE int ReadVarUint32(const char* input, const char* end, ui32* value)
{
    return NYT::NDetail::ReadVarUint32Impl(value, input, end);
}

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class... TArgs>
Y_FORCE_INLINE int ReadVarInt32Impl(i32* value, TArgs&&... args)
{
    ui64 varInt;
    int bytesRead = ReadVarUint64(std::forward<TArgs>(args)..., &varInt);
    if (varInt > std::numeric_limits<ui32>::max()) {
        throw TSimpleException("Value is too big for varint32");
    }
    *value = ZigZagDecode32(static_cast<ui32>(varInt));
    return bytesRead;
}

} // namespace NDetail

Y_FORCE_INLINE int ReadVarInt32(IInputStream* input, i32* value)
{
    return NYT::NDetail::ReadVarInt32Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt32(const char* input, i32* value)
{
    return NYT::NDetail::ReadVarInt32Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt32(const char* input, const char* end, i32* value)
{
    return NYT::NDetail::ReadVarInt32Impl(value, input, end);
}

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class... TArgs>
Y_FORCE_INLINE int ReadVarInt64Impl(i64* value, TArgs&&... args)
{
    ui64 varInt;
    int bytesRead = ReadVarUint64(std::forward<TArgs>(args)..., &varInt);
    *value = ZigZagDecode64(varInt);
    return bytesRead;
}

} // namespace NDetail

Y_FORCE_INLINE int ReadVarInt64(IInputStream* input, i64* value)
{
    return NYT::NDetail::ReadVarInt64Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt64(const char* input, i64* value)
{
    return NYT::NDetail::ReadVarInt64Impl(value, input);
}

Y_FORCE_INLINE int ReadVarInt64(const char* input, const char* end, i64* value)
{
    return NYT::NDetail::ReadVarInt64Impl(value, input, end);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
