#include "int128.h"

#include <util/system/yassert.h>

#include <cmath>
#include <iterator>
#include <limits>

namespace {
    constexpr size_t DecimalBaseDigits = std::numeric_limits<ui64>::digits10;
    constexpr ui64 DecimalBase = 10'000'000'000'000'000'000ULL;
    constexpr size_t MaxDigits = std::numeric_limits<ui128>::digits10 + 1;
    constexpr size_t DecimalWordCount = (MaxDigits + DecimalBaseDigits - 1) / DecimalBaseDigits;
    constexpr size_t SignedBufferSize = std::numeric_limits<i128>::digits10 + 2;
    constexpr ui64 BinaryWordBase = ui64{1} << 32;
    constexpr ui64 BinaryWordMask = BinaryWordBase - 1;
    const ui64 DivisorHigh = DecimalBase >> 32;
    const ui64 DivisorLow = DecimalBase & BinaryWordMask;

    static_assert(DecimalWordCount * DecimalBaseDigits >= MaxDigits);
    static_assert((DecimalWordCount - 1) * DecimalBaseDigits < MaxDigits);
    static_assert(SignedBufferSize == MaxDigits + 1);

    bool QuotientEstimateNeedsCorrection(
        const ui64 quotientEstimate,
        const ui64 remainderEstimate,
        const ui64 nextDividendWord) noexcept
    {
        return quotientEstimate >= BinaryWordBase ||
               quotientEstimate * DivisorLow > BinaryWordBase * remainderEstimate + nextDividendWord;
    }

    void CorrectQuotientEstimate(
        ui64& quotientEstimate,
        ui64& remainderEstimate,
        const ui64 nextDividendWord) noexcept
    {
        if (!QuotientEstimateNeedsCorrection(quotientEstimate, remainderEstimate, nextDividendWord)) {
            return;
        }

        --quotientEstimate;
        remainderEstimate += DivisorHigh;
        if (remainderEstimate >= BinaryWordBase ||
            !QuotientEstimateNeedsCorrection(quotientEstimate, remainderEstimate, nextDividendWord))
        {
            return;
        }

        // Knuth D guarantees that the quotient estimate needs at most two corrections.
        --quotientEstimate;
        remainderEstimate += DivisorHigh;
        if (remainderEstimate < BinaryWordBase) {
            Y_ASSERT(!QuotientEstimateNeedsCorrection(quotientEstimate, remainderEstimate, nextDividendWord));
        }
    }

    ui64 Divide128ByNormalized64(const ui64 high, const ui64 low, ui64& remainder) noexcept {
        // Knuth's algorithm D specialized for a normalized 64-bit divisor.
        // The precondition high < DecimalBase guarantees that the quotient fits ui64.
        const ui64 dividendMiddle = low >> 32;
        const ui64 dividendLow = low & BinaryWordMask;

        ui64 quotientHigh = high / DivisorHigh;
        ui64 remainderEstimate = high - quotientHigh * DivisorHigh;
        CorrectQuotientEstimate(quotientHigh, remainderEstimate, dividendMiddle);

        const ui64 dividend = high * BinaryWordBase + dividendMiddle - quotientHigh * DecimalBase;

        ui64 quotientLow = dividend / DivisorHigh;
        remainderEstimate = dividend - quotientLow * DivisorHigh;
        CorrectQuotientEstimate(quotientLow, remainderEstimate, dividendLow);

        remainder = dividend * BinaryWordBase + dividendLow - quotientLow * DecimalBase;
        return quotientHigh * BinaryWordBase + quotientLow;
    }

    ui128 DivModBase1e19(const ui128 value, ui64& remainder) noexcept {
        const ui64 high = GetHigh(value);
        const ui64 quotientHigh = high / DecimalBase;
        const ui64 highRemainder = high % DecimalBase;
        const ui64 quotientLow = Divide128ByNormalized64(highRemainder, GetLow(value), remainder);
        return {quotientHigh, quotientLow};
    }

    char* FormatUi64(char* position, ui64 value) noexcept {
        do {
            *--position = static_cast<char>('0' + value % 10);
            value /= 10;
        } while (value != 0);
        return position;
    }

    char* FormatUi128(char* const bufferEnd, const ui64 high, const ui64 low) noexcept {
        if (high == 0) {
            return FormatUi64(bufferEnd, low);
        }

        ui64 decimalWords[DecimalWordCount];
        ui128 quotient{high, low};
        for (size_t i = 0; i + 1 < std::size(decimalWords); ++i) {
            quotient = DivModBase1e19(quotient, decimalWords[i]);
        }
        Y_ASSERT(GetHigh(quotient) == 0);
        decimalWords[std::size(decimalWords) - 1] = GetLow(quotient);

        size_t decimalWordCount = std::size(decimalWords);
        while (decimalWordCount > 1 && decimalWords[decimalWordCount - 1] == 0) {
            --decimalWordCount;
        }

        char* position = bufferEnd;
        for (size_t i = 0; i + 1 < decimalWordCount; ++i) {
            ui64 word = decimalWords[i];
            for (size_t digit = 0; digit < DecimalBaseDigits; ++digit) {
                *--position = static_cast<char>('0' + word % 10);
                word /= 10;
            }
        }

        return FormatUi64(position, decimalWords[decimalWordCount - 1]);
    }
} // namespace

IOutputStream& operator<<(IOutputStream& out, const ui128& other) {
    if (other.High_ == 0) {
        return out << other.Low_;
    }

    char buffer[MaxDigits];
    char* const position = FormatUi128(std::end(buffer), other.High_, other.Low_);
    out.Write(position, std::end(buffer) - position);
    return out;
}

void TSerializer<ui128>::Save(IOutputStream* out, const ui128& Number) {
    ::Save(out, GetHigh(Number));
    ::Save(out, GetLow(Number));
}

void TSerializer<ui128>::Load(IInputStream* in, ui128& Number) {
    ui64 High;
    ui64 Low;
    ::Load(in, High);
    ::Load(in, Low);
    Number = ui128(High, Low);
}

IOutputStream& operator<<(IOutputStream& out, const i128& other) {
    ui64 high = other.High_;
    ui64 low = other.Low_;
    const bool negative = signbit(other);
    if (negative) {
        low = ~low + 1;
        high = ~high + (low == 0);
    }

    char buffer[SignedBufferSize];
    char* position = FormatUi128(std::end(buffer), high, low);
    if (negative) {
        *--position = '-';
    }
    out.Write(position, std::end(buffer) - position);
    return out;
}
