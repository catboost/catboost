#pragma once

#include "public.h"
#include "zigzag.h"

#include <util/generic/vector.h>
#include <util/generic/maybe.h>
#include <util/generic/buffer.h>
#include <util/string/escape.h>
#include <util/string/cast.h>
#include <util/stream/input.h>

namespace NYson {
    namespace NDetail {
        ////////////////////////////////////////////////////////////////////////////////

        //! Indicates the beginning of a list.
        const char BeginListSymbol = '[';
        //! Indicates the end of a list.
        const char EndListSymbol = ']';

        //! Indicates the beginning of a map.
        const char BeginMapSymbol = '{';
        //! Indicates the end of a map.
        const char EndMapSymbol = '}';

        //! Indicates the beginning of an attribute map.
        const char BeginAttributesSymbol = '<';
        //! Indicates the end of an attribute map.
        const char EndAttributesSymbol = '>';

        //! Separates items in lists.
        const char ListItemSeparatorSymbol = ';';
        //! Separates items in maps, attributes.
        const char KeyedItemSeparatorSymbol = ';';
        //! Separates keys from values in maps.
        const char KeyValueSeparatorSymbol = '=';

        //! Indicates an entity.
        const char EntitySymbol = '#';

        //! Indicates end of stream.
        const char EndSymbol = '\0';

        //! Marks the beginning of a binary string literal.
        const char StringMarker = '\x01';
        //! Marks the beginning of a binary i64 literal.
        const char Int64Marker = '\x02';
        //! Marks the beginning of a binary double literal.
        const char DoubleMarker = '\x03';
        //! Marks true and false values of boolean.
        const char FalseMarker = '\x04';
        const char TrueMarker = '\x05';
        //! Marks the beginning of a binary ui64 literal.
        const char Uint64Marker = '\x06';

        ////////////////////////////////////////////////////////////////////////////////

        template <bool EnableLinePositionInfo>
        class TPositionInfo;

        template <>
        class TPositionInfo<true> {
        private:
            int Offset;
            int Line;
            int Column;

        public:
            TPositionInfo()
                : Offset(0)
                , Line(1)
                , Column(1)
            {
            }

            void OnRangeConsumed(const char* begin, const char* end) {
                Offset += end - begin;
                for (auto current = begin; current != end; ++current) {
                    ++Column;
                    if (*current == '\n') { //TODO: memchr
                        ++Line;
                        Column = 1;
                    }
                }
            }
        };

        template <>
        class TPositionInfo<false> {
        private:
            int Offset;

        public:
            TPositionInfo()
                : Offset(0)
            {
            }

            void OnRangeConsumed(const char* begin, const char* end) {
                Offset += end - begin;
            }
        };

        template <class TBlockStream, class TPositionBase>
        class TCharStream
           : public TBlockStream,
              public TPositionBase {
        public:
            TCharStream(const TBlockStream& blockStream)
                : TBlockStream(blockStream)
            {
            }

            bool IsEmpty() const {
                return TBlockStream::Begin() == TBlockStream::End();
            }

            template <bool AllowFinish>
            void Refresh() {
                while (IsEmpty() && !TBlockStream::IsFinished()) {
                    TBlockStream::RefreshBlock();
                }
                if (IsEmpty() && TBlockStream::IsFinished() && !AllowFinish) {
                    ythrow TYsonException() << "Premature end of yson stream";
                }
            }

            void Refresh() {
                return Refresh<false>();
            }

            template <bool AllowFinish>
            char GetChar() {
                Refresh<AllowFinish>();
                return !IsEmpty() ? *TBlockStream::Begin() : '\0';
            }

            char GetChar() {
                return GetChar<false>();
            }

            void Advance(size_t bytes) {
                TPositionBase::OnRangeConsumed(TBlockStream::Begin(), TBlockStream::Begin() + bytes);
                TBlockStream::Advance(bytes);
            }

            size_t Length() const {
                return TBlockStream::End() - TBlockStream::Begin();
            }
        };

        template <class TBaseStream>
        class TCodedStream
           : public TBaseStream {
        private:
            static const int MaxVarintBytes = 10;
            static const int MaxVarint32Bytes = 5;

            const ui8* BeginByte() const {
                return reinterpret_cast<const ui8*>(TBaseStream::Begin());
            }

            const ui8* EndByte() const {
                return reinterpret_cast<const ui8*>(TBaseStream::End());
            }

            // Following functions is an adaptation Protobuf code from coded_stream.cc
            bool ReadVarint32FromArray(ui32* value) {
                // Fast path:  We have enough bytes left in the buffer to guarantee that
                // this read won't cross the end, so we can skip the checks.
                const ui8* ptr = BeginByte();
                ui32 b;
                ui32 result;

                b = *(ptr++);
                result = (b & 0x7F);
                if (!(b & 0x80))
                    goto done;
                b = *(ptr++);
                result |= (b & 0x7F) << 7;
                if (!(b & 0x80))
                    goto done;
                b = *(ptr++);
                result |= (b & 0x7F) << 14;
                if (!(b & 0x80))
                    goto done;
                b = *(ptr++);
                result |= (b & 0x7F) << 21;
                if (!(b & 0x80))
                    goto done;
                b = *(ptr++);
                result |= b << 28;
                if (!(b & 0x80))
                    goto done;

                // If the input is larger than 32 bits, we still need to read it all
                // and discard the high-order bits.

                for (int i = 0; i < MaxVarintBytes - MaxVarint32Bytes; i++) {
                    b = *(ptr++);
                    if (!(b & 0x80))
                        goto done;
                }

                // We have overrun the maximum size of a Varint (10 bytes).  Assume
                // the data is corrupt.
                return false;

            done:
                TBaseStream::Advance(ptr - BeginByte());
                *value = result;
                return true;
            }

            bool ReadVarint32Fallback(ui32* value) {
                if (BeginByte() + MaxVarintBytes <= EndByte() ||
                    // Optimization:  If the Varint ends at exactly the end of the buffer,
                    // we can detect that and still use the fast path.
                    (BeginByte() < EndByte() && !(EndByte()[-1] & 0x80)))
                {
                    return ReadVarint32FromArray(value);
                } else {
                    // Really slow case: we will incur the cost of an extra function call here,
                    // but moving this out of line reduces the size of this function, which
                    // improves the common case. In micro benchmarks, this is worth about 10-15%
                    return ReadVarint32Slow(value);
                }
            }

            bool ReadVarint32Slow(ui32* value) {
                ui64 result;
                // Directly invoke ReadVarint64Fallback, since we already tried to optimize
                // for one-byte Varints.
                if (ReadVarint64Fallback(&result)) {
                    *value = static_cast<ui32>(result);
                    return true;
                } else {
                    return false;
                }
            }

            bool ReadVarint64Slow(ui64* value) {
                // Slow path:  This read might cross the end of the buffer, so we
                // need to check and refresh the buffer if and when it does.

                ui64 result = 0;
                int count = 0;
                ui32 b;

                do {
                    if (count == MaxVarintBytes) {
                        return false;
                    }
                    while (BeginByte() == EndByte()) {
                        TBaseStream::Refresh();
                    }
                    b = *BeginByte();
                    result |= static_cast<ui64>(b & 0x7F) << (7 * count);
                    TBaseStream::Advance(1);
                    ++count;
                } while (b & 0x80);

                *value = result;
                return true;
            }

            bool ReadVarint64Fallback(ui64* value) {
                if (BeginByte() + MaxVarintBytes <= EndByte() ||
                    // Optimization:  If the Varint ends at exactly the end of the buffer,
                    // we can detect that and still use the fast path.
                    (BeginByte() < EndByte() && !(EndByte()[-1] & 0x80)))
                {
                    // Fast path:  We have enough bytes left in the buffer to guarantee that
                    // this read won't cross the end, so we can skip the checks.

                    const ui8* ptr = BeginByte();
                    ui32 b;

                    // Splitting into 32-bit pieces gives better performance on 32-bit
                    // processors.
                    ui32 part0 = 0, part1 = 0, part2 = 0;

                    b = *(ptr++);
                    part0 = (b & 0x7F);
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part0 |= (b & 0x7F) << 7;
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part0 |= (b & 0x7F) << 14;
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part0 |= (b & 0x7F) << 21;
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part1 = (b & 0x7F);
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part1 |= (b & 0x7F) << 7;
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part1 |= (b & 0x7F) << 14;
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part1 |= (b & 0x7F) << 21;
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part2 = (b & 0x7F);
                    if (!(b & 0x80))
                        goto done;
                    b = *(ptr++);
                    part2 |= (b & 0x7F) << 7;
                    if (!(b & 0x80))
                        goto done;

                    // We have overrun the maximum size of a Varint (10 bytes).  The data
                    // must be corrupt.
                    return false;

                done:
                    TBaseStream::Advance(ptr - BeginByte());
                    *value = (static_cast<ui64>(part0)) |
                             (static_cast<ui64>(part1) << 28) |
                             (static_cast<ui64>(part2) << 56);
                    return true;
                } else {
                    return ReadVarint64Slow(value);
                }
            }

        public:
            TCodedStream(const TBaseStream& baseStream)
                : TBaseStream(baseStream)
            {
            }

            bool ReadVarint64(ui64* value) {
                if (BeginByte() < EndByte() && *BeginByte() < 0x80) {
                    *value = *BeginByte();
                    TBaseStream::Advance(1);
                    return true;
                } else {
                    return ReadVarint64Fallback(value);
                }
            }

            bool ReadVarint32(ui32* value) {
                if (BeginByte() < EndByte() && *BeginByte() < 0x80) {
                    *value = *BeginByte();
                    TBaseStream::Advance(1);
                    return true;
                } else {
                    return ReadVarint32Fallback(value);
                }
            }
        };

        enum ENumericResult {
            Int64 = 0,
            Uint64 = 1,
            Double = 2
        };

        template <class TBlockStream, bool EnableLinePositionInfo>
        class TLexerBase
           : public TCodedStream<TCharStream<TBlockStream, TPositionInfo<EnableLinePositionInfo>>> {
        private:
            using TBaseStream = TCodedStream<TCharStream<TBlockStream, TPositionInfo<EnableLinePositionInfo>>>;
            TVector<char> Buffer_;
            TMaybe<ui64> MemoryLimit_;

            void CheckMemoryLimit() {
                if (MemoryLimit_ && Buffer_.capacity() > *MemoryLimit_) {
                    ythrow TYsonException()
                        << "Memory limit exceeded while parsing YSON stream: allocated "
                        << Buffer_.capacity() << ", limit " << (*MemoryLimit_);
                }
            }

        public:
            TLexerBase(const TBlockStream& blockStream, TMaybe<ui64> memoryLimit)
                : TBaseStream(blockStream)
                , MemoryLimit_(memoryLimit)
            {
            }

        protected:
            /// Lexer routines

            template <bool AllowFinish>
            ENumericResult ReadNumeric(TStringBuf* value) {
                Buffer_.clear();
                ENumericResult result = ENumericResult::Int64;
                while (true) {
                    char ch = TBaseStream::template GetChar<AllowFinish>();
                    if (isdigit(ch) || ch == '+' || ch == '-') { // Seems like it can't be '+' or '-'
                        Buffer_.push_back(ch);
                    } else if (ch == '.' || ch == 'e' || ch == 'E') {
                        Buffer_.push_back(ch);
                        result = ENumericResult::Double;
                    } else if (ch == 'u') {
                        Buffer_.push_back(ch);
                        result = ENumericResult::Uint64;
                    } else if (isalpha(ch)) {
                        ythrow TYsonException() << "Unexpected '" << ch << "' in numeric literal";
                    } else {
                        break;
                    }
                    CheckMemoryLimit();
                    TBaseStream::Advance(1);
                }

                *value = TStringBuf(Buffer_.data(), Buffer_.size());
                return result;
            }

            template <bool AllowFinish>
            double ReadNanOrInf() {
                static const TStringBuf nanString = "nan";
                static const TStringBuf infString = "inf";
                static const TStringBuf plusInfString = "+inf";
                static const TStringBuf minusInfString = "-inf";

                TStringBuf expectedString;
                double expectedValue;
                char ch = TBaseStream::template GetChar<AllowFinish>();
                switch (ch) {
                    case '+':
                        expectedString = plusInfString;
                        expectedValue = std::numeric_limits<double>::infinity();
                        break;
                    case '-':
                        expectedString = minusInfString;
                        expectedValue = -std::numeric_limits<double>::infinity();
                        break;
                    case 'i':
                        expectedString = infString;
                        expectedValue = std::numeric_limits<double>::infinity();
                        break;
                    case 'n':
                        expectedString = nanString;
                        expectedValue = std::numeric_limits<double>::quiet_NaN();
                        break;
                    default:
                        ythrow TYsonException() << "Incorrect %-literal prefix: '" << ch << "'";
                }

                for (size_t i = 0; i < expectedString.size(); ++i) {
                    if (expectedString[i] != ch) {
                        ythrow TYsonException()
                            << "Incorrect %-literal prefix "
                            << "'" << expectedString.SubStr(0, i) << ch << "',"
                            << "expected " << expectedString;
                    }
                    TBaseStream::Advance(1);
                    ch = TBaseStream::template GetChar<AllowFinish>();
                }

                return expectedValue;
            }

            void ReadQuotedString(TStringBuf* value) {
                Buffer_.clear();
                while (true) {
                    if (TBaseStream::IsEmpty()) {
                        TBaseStream::Refresh();
                    }
                    char ch = *TBaseStream::Begin();
                    TBaseStream::Advance(1);
                    if (ch != '"') {
                        Buffer_.push_back(ch);
                    } else {
                        // We must count the number of '\' at the end of StringValue
                        // to check if it's not \"
                        int slashCount = 0;
                        int length = Buffer_.size();
                        while (slashCount < length && Buffer_[length - 1 - slashCount] == '\\') {
                            ++slashCount;
                        }
                        if (slashCount % 2 == 0) {
                            break;
                        } else {
                            Buffer_.push_back(ch);
                        }
                    }
                    CheckMemoryLimit();
                }

                auto unquotedValue = UnescapeC(Buffer_.data(), Buffer_.size());
                Buffer_.clear();
                Buffer_.insert(Buffer_.end(), unquotedValue.data(), unquotedValue.data() + unquotedValue.size());
                CheckMemoryLimit();
                *value = TStringBuf(Buffer_.data(), Buffer_.size());
            }

            template <bool AllowFinish>
            void ReadUnquotedString(TStringBuf* value) {
                Buffer_.clear();
                while (true) {
                    char ch = TBaseStream::template GetChar<AllowFinish>();
                    if (isalpha(ch) || isdigit(ch) ||
                        ch == '_' || ch == '-' || ch == '%' || ch == '.') {
                        Buffer_.push_back(ch);
                    } else {
                        break;
                    }
                    CheckMemoryLimit();
                    TBaseStream::Advance(1);
                }
                *value = TStringBuf(Buffer_.data(), Buffer_.size());
            }

            void ReadUnquotedString(TStringBuf* value) {
                return ReadUnquotedString<false>(value);
            }

            void ReadBinaryString(TStringBuf* value) {
                ui32 ulength = 0;
                if (!TBaseStream::ReadVarint32(&ulength)) {
                    ythrow TYsonException() << "Error parsing varint value";
                }

                i32 length = ZigZagDecode32(ulength);
                if (length < 0) {
                    ythrow TYsonException() << "Negative binary string literal length " << length;
                }

                if (TBaseStream::Begin() + length <= TBaseStream::End()) {
                    *value = TStringBuf(TBaseStream::Begin(), length);
                    TBaseStream::Advance(length);
                } else { // reading in Buffer
                    size_t needToRead = length;
                    Buffer_.clear();
                    while (needToRead) {
                        if (TBaseStream::IsEmpty()) {
                            TBaseStream::Refresh();
                            continue;
                        }
                        size_t readingBytes = Min(needToRead, TBaseStream::Length());

                        Buffer_.insert(Buffer_.end(), TBaseStream::Begin(), TBaseStream::Begin() + readingBytes);
                        CheckMemoryLimit();
                        needToRead -= readingBytes;
                        TBaseStream::Advance(readingBytes);
                    }
                    *value = TStringBuf(Buffer_.data(), Buffer_.size());
                }
            }

            template <bool AllowFinish>
            bool ReadBoolean() {
                Buffer_.clear();

                static TStringBuf trueString = "true";
                static TStringBuf falseString = "false";

                auto throwIncorrectBoolean = [&]() {
                    ythrow TYsonException() << "Incorrect boolean string " << TString(Buffer_.data(), Buffer_.size());
                };

                Buffer_.push_back(TBaseStream::template GetChar<AllowFinish>());
                TBaseStream::Advance(1);
                if (Buffer_[0] == trueString[0]) {
                    for (size_t i = 1; i < trueString.size(); ++i) {
                        Buffer_.push_back(TBaseStream::template GetChar<AllowFinish>());
                        TBaseStream::Advance(1);
                        if (Buffer_.back() != trueString[i]) {
                            throwIncorrectBoolean();
                        }
                    }
                    return true;
                } else if (Buffer_[0] == falseString[0]) {
                    for (size_t i = 1; i < falseString.size(); ++i) {
                        Buffer_.push_back(TBaseStream::template GetChar<AllowFinish>());
                        TBaseStream::Advance(1);
                        if (Buffer_.back() != falseString[i]) {
                            throwIncorrectBoolean();
                        }
                    }
                    return false;
                } else {
                    throwIncorrectBoolean();
                }

                Y_ABORT("unreachable");
                ;
            }

            void ReadBinaryInt64(i64* result) {
                ui64 uvalue;
                if (!TBaseStream::ReadVarint64(&uvalue)) {
                    ythrow TYsonException() << "Error parsing varint value";
                }
                *result = ZigZagDecode64(uvalue);
            }

            void ReadBinaryUint64(ui64* result) {
                ui64 uvalue;
                if (!TBaseStream::ReadVarint64(&uvalue)) {
                    ythrow TYsonException() << "Error parsing varint value";
                }
                *result = uvalue;
            }

            void ReadBinaryDouble(double* value) {
                size_t needToRead = sizeof(double);

                while (needToRead != 0) {
                    if (TBaseStream::IsEmpty()) {
                        TBaseStream::Refresh();
                        continue;
                    }

                    size_t chunkSize = Min(needToRead, TBaseStream::Length());
                    if (chunkSize == 0) {
                        ythrow TYsonException() << "Error parsing binary double literal";
                    }
                    std::copy(
                        TBaseStream::Begin(),
                        TBaseStream::Begin() + chunkSize,
                        reinterpret_cast<char*>(value) + (sizeof(double) - needToRead));
                    needToRead -= chunkSize;
                    TBaseStream::Advance(chunkSize);
                }
            }

            /// Helpers
            void SkipCharToken(char symbol) {
                char ch = SkipSpaceAndGetChar();
                if (ch != symbol) {
                    ythrow TYsonException() << "Expected '" << symbol << "' but found '" << ch << "'";
                }

                TBaseStream::Advance(1);
            }

            static bool IsSpaceFast(char ch) {
                static const ui8 lookupTable[] =
                    {
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                return lookupTable[static_cast<ui8>(ch)];
            }

            template <bool AllowFinish>
            char SkipSpaceAndGetChar() {
                if (!TBaseStream::IsEmpty()) {
                    char ch = *TBaseStream::Begin();
                    if (!IsSpaceFast(ch)) {
                        return ch;
                    }
                }
                return SkipSpaceAndGetCharFallback<AllowFinish>();
            }

            char SkipSpaceAndGetChar() {
                return SkipSpaceAndGetChar<false>();
            }

            template <bool AllowFinish>
            char SkipSpaceAndGetCharFallback() {
                while (true) {
                    if (TBaseStream::IsEmpty()) {
                        if (TBaseStream::IsFinished()) {
                            return '\0';
                        }
                        TBaseStream::template Refresh<AllowFinish>();
                        continue;
                    }
                    if (!IsSpaceFast(*TBaseStream::Begin())) {
                        break;
                    }
                    TBaseStream::Advance(1);
                }
                return TBaseStream::template GetChar<AllowFinish>();
            }
        };

        ////////////////////////////////////////////////////////////////////////////////

    }

    ////////////////////////////////////////////////////////////////////////////////

    class TStringReader {
    private:
        const char* BeginPtr;
        const char* EndPtr;

    public:
        TStringReader()
            : BeginPtr(nullptr)
            , EndPtr(nullptr)
        {
        }

        TStringReader(const char* begin, const char* end)
            : BeginPtr(begin)
            , EndPtr(end)
        {
        }

        const char* Begin() const {
            return BeginPtr;
        }

        const char* End() const {
            return EndPtr;
        }

        void RefreshBlock() {
            Y_ABORT("unreachable");
        }

        void Advance(size_t bytes) {
            BeginPtr += bytes;
        }

        bool IsFinished() const {
            return true;
        }

        void SetBuffer(const char* begin, const char* end) {
            BeginPtr = begin;
            EndPtr = end;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    class TStreamReader {
    public:
        TStreamReader(
            IInputStream* stream,
            char* buffer,
            size_t bufferSize)
            : Stream(stream)
            , Buffer(buffer)
            , BufferSize(bufferSize)
        {
            BeginPtr = EndPtr = Buffer;
            FinishFlag = false;
        }

        const char* Begin() const {
            return BeginPtr;
        }

        const char* End() const {
            return EndPtr;
        }

        void RefreshBlock() {
            size_t bytes = Stream->Read(Buffer, BufferSize);
            BeginPtr = Buffer;
            EndPtr = Buffer + bytes;
            FinishFlag = (bytes == 0);
        }

        void Advance(size_t bytes) {
            BeginPtr += bytes;
        }

        bool IsFinished() const {
            return FinishFlag;
        }

    private:
        IInputStream* Stream;
        char* Buffer;
        size_t BufferSize;

        const char* BeginPtr;
        const char* EndPtr;
        bool FinishFlag;
    };

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
