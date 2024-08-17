#include <util/charset/wide.h>
#include <util/system/types.h>

#ifdef SSE41_STUB

namespace NDetail {
    void UTF8ToWideImplSSE41(const unsigned char*&, const unsigned char*, wchar16*&) noexcept {
    }
    void UTF8ToWideImplSSE41(const unsigned char*&, const unsigned char*, wchar32*&) noexcept {
    }
}

#else

    #include <util/system/compiler.h>

    #include <cstring>
    #include <emmintrin.h>
    #include <smmintrin.h>

// processes to the first error, or until less then 16 bytes left
// most code taken from https://woboq.com/blog/utf-8-processing-using-simd.html

// return dstAdvance 0 in case of problems
static Y_FORCE_INLINE ui32 Unpack16BytesIntoUtf16IfNoSurrogats(const unsigned char*& cur, __m128i& utf16Low, __m128i& utf16High) {
    unsigned char curAligned[16];

    memcpy(curAligned, cur, sizeof(__m128i));
    __m128i chunk = _mm_load_si128(reinterpret_cast<const __m128i*>(curAligned));

    // only ascii characters - simple copy
    if (!_mm_movemask_epi8(chunk)) {
        utf16Low = _mm_unpacklo_epi8(chunk, _mm_setzero_si128());
        utf16High = _mm_unpackhi_epi8(chunk, _mm_setzero_si128());
        cur += 16;
        return 16;
    }

    __m128i chunkSigned = _mm_add_epi8(chunk, _mm_set1_epi8(0x80));
    __m128i isAsciiMask = _mm_cmpgt_epi8(chunk, _mm_set1_epi8(0));

    __m128i cond2 = _mm_cmplt_epi8(_mm_set1_epi8(0xc2 - 1 - 0x80), chunkSigned);
    __m128i state = _mm_set1_epi8(0x0 | (char)0x80);

    __m128i cond3 = _mm_cmplt_epi8(_mm_set1_epi8(0xe0 - 1 - 0x80), chunkSigned);
    state = _mm_blendv_epi8(state, _mm_set1_epi8(0x2 | (char)0xc0), cond2);

    int sourceAdvance;
    __m128i shifts;
    __m128i chunkLow, chunkHigh;

    if (Y_LIKELY(!_mm_movemask_epi8(cond3))) {
        // main case: no bloks of size 3 or 4

        // rune len for start of multi-byte sequences (0 for b0... and b10..., 2 for b110..., etc.)
        __m128i count = _mm_and_si128(state, _mm_set1_epi8(0x7));

        __m128i countSub1 = _mm_subs_epu8(count, _mm_set1_epi8(0x1));

        shifts = countSub1;
        __m128i continuation1 = _mm_slli_si128(countSub1, 1);

        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 1));
        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 2));

        __m128i counts = _mm_or_si128(count, continuation1);

        __m128i isBeginMultibyteMask = _mm_cmpgt_epi8(count, _mm_set1_epi8(0));
        __m128i needNoContinuationMask = _mm_cmpeq_epi8(continuation1, _mm_set1_epi8(0));
        __m128i isBeginMask = _mm_add_epi8(isBeginMultibyteMask, isAsciiMask);
        // each symbol should be exactly one of ascii, continuation or begin
        __m128i okMask = _mm_cmpeq_epi8(isBeginMask, needNoContinuationMask);

        if (_mm_movemask_epi8(okMask) != 0xFFFF) {
            return 0;
        }

        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 4));

        __m128i mask = _mm_and_si128(state, _mm_set1_epi8(0xf8));
        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 8));

        chunk = _mm_andnot_si128(mask, chunk);                                    // from now on, we only have usefull bits
        shifts = _mm_and_si128(shifts, _mm_cmplt_epi8(counts, _mm_set1_epi8(2))); // <=1

        __m128i chunk_right = _mm_slli_si128(chunk, 1);
        shifts = _mm_blendv_epi8(shifts, _mm_srli_si128(shifts, 1),
                                 _mm_srli_si128(_mm_slli_epi16(shifts, 7), 1));

        chunkLow = _mm_blendv_epi8(chunk,
                                   _mm_or_si128(chunk, _mm_and_si128(_mm_slli_epi16(chunk_right, 6), _mm_set1_epi8(0xc0))),
                                   _mm_cmpeq_epi8(counts, _mm_set1_epi8(1)));

        chunkHigh = _mm_and_si128(chunk, _mm_cmpeq_epi8(counts, _mm_set1_epi8(2)));

        shifts = _mm_blendv_epi8(shifts, _mm_srli_si128(shifts, 2),
                                 _mm_srli_si128(_mm_slli_epi16(shifts, 6), 2));
        chunkHigh = _mm_srli_epi32(chunkHigh, 2);

        shifts = _mm_blendv_epi8(shifts, _mm_srli_si128(shifts, 4),
                                 _mm_srli_si128(_mm_slli_epi16(shifts, 5), 4));

        int c = _mm_extract_epi16(counts, 7);
        sourceAdvance = !(c & 0x0200) ? 16 : 15;

    } else {
        __m128i mask3 = _mm_slli_si128(cond3, 1);

        __m128i cond4 = _mm_cmplt_epi8(_mm_set1_epi8(0xf0 - 1 - 0x80), chunkSigned);
        state = _mm_blendv_epi8(state, _mm_set1_epi8(0x3 | (char)0xe0), cond3);

        // 4 bytes sequences are not vectorize. Fall back to the scalar processing
        if (Y_UNLIKELY(_mm_movemask_epi8(cond4))) {
            return 0;
        }

        // rune len for start of multi-byte sequences (0 for b0... and b10..., 2 for b110..., etc.)
        __m128i count = _mm_and_si128(state, _mm_set1_epi8(0x7));

        __m128i countSub1 = _mm_subs_epu8(count, _mm_set1_epi8(0x1));
        __m128i continuation2 = _mm_slli_si128(_mm_subs_epu8(count, _mm_set1_epi8(0x2)), 2);

        shifts = countSub1;
        __m128i continuation1 = _mm_slli_si128(countSub1, 1);

        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 1));
        __m128i continuationsRunelen = _mm_or_si128(continuation1, continuation2);

        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 2));
        __m128i counts = _mm_or_si128(count, continuationsRunelen);

        __m128i isBeginMultibyteMask = _mm_cmpgt_epi8(count, _mm_set1_epi8(0));
        __m128i needNoContinuationMask = _mm_cmpeq_epi8(continuationsRunelen, _mm_set1_epi8(0));
        __m128i isBeginMask = _mm_add_epi8(isBeginMultibyteMask, isAsciiMask);
        // each symbol should be exactly one of ascii, continuation or begin
        __m128i okMask = _mm_cmpeq_epi8(isBeginMask, needNoContinuationMask);

        if (_mm_movemask_epi8(okMask) != 0xFFFF) {
            return 0;
        }

        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 4));

        __m128i mask = _mm_and_si128(state, _mm_set1_epi8(0xf8));
        shifts = _mm_add_epi8(shifts, _mm_slli_si128(shifts, 8));

        chunk = _mm_andnot_si128(mask, chunk);                                    // from now on, we only have usefull bits
        shifts = _mm_and_si128(shifts, _mm_cmplt_epi8(counts, _mm_set1_epi8(2))); // <=1

        __m128i chunk_right = _mm_slli_si128(chunk, 1);
        shifts = _mm_blendv_epi8(shifts, _mm_srli_si128(shifts, 1),
                                 _mm_srli_si128(_mm_slli_epi16(shifts, 7), 1));

        chunkLow = _mm_blendv_epi8(chunk,
                                   _mm_or_si128(chunk, _mm_and_si128(_mm_slli_epi16(chunk_right, 6), _mm_set1_epi8(0xc0))),
                                   _mm_cmpeq_epi8(counts, _mm_set1_epi8(1)));

        chunkHigh = _mm_and_si128(chunk, _mm_cmpeq_epi8(counts, _mm_set1_epi8(2)));

        shifts = _mm_blendv_epi8(shifts, _mm_srli_si128(shifts, 2),
                                 _mm_srli_si128(_mm_slli_epi16(shifts, 6), 2));
        chunkHigh = _mm_srli_epi32(chunkHigh, 2);

        shifts = _mm_blendv_epi8(shifts, _mm_srli_si128(shifts, 4),
                                 _mm_srli_si128(_mm_slli_epi16(shifts, 5), 4));
        chunkHigh = _mm_or_si128(chunkHigh,
                                 _mm_and_si128(_mm_and_si128(_mm_slli_epi32(chunk_right, 4), _mm_set1_epi8(0xf0)),
                                               mask3));

        int c = _mm_extract_epi16(counts, 7);
        sourceAdvance = !(c & 0x0200) ? 16 : !(c & 0x02) ? 15
                                                         : 14;
    }

    shifts = _mm_blendv_epi8(shifts, _mm_srli_si128(shifts, 8),
                             _mm_srli_si128(_mm_slli_epi16(shifts, 4), 8));

    chunkHigh = _mm_slli_si128(chunkHigh, 1);

    __m128i shuf = _mm_add_epi8(shifts, _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

    chunkLow = _mm_shuffle_epi8(chunkLow, shuf);
    chunkHigh = _mm_shuffle_epi8(chunkHigh, shuf);

    utf16Low = _mm_unpacklo_epi8(chunkLow, chunkHigh);
    utf16High = _mm_unpackhi_epi8(chunkLow, chunkHigh);

    ui32 s = _mm_extract_epi32(shifts, 3);
    ui32 destAdvance = sourceAdvance - (0xff & (s >> (8 * (3 - 16 + sourceAdvance))));
    cur += sourceAdvance;
    return destAdvance;
}

namespace NDetail {
    void UTF8ToWideImplSSE41(const unsigned char*& cur, const unsigned char* last, wchar16*& dest) noexcept {
        alignas(16) wchar16 destAligned[16];

        while (cur + 16 <= last) {
            __m128i utf16Low;
            __m128i utf16High;
            ui32 dstAdvance = Unpack16BytesIntoUtf16IfNoSurrogats(cur, utf16Low, utf16High);

            if (dstAdvance == 0) {
                break;
            }

            _mm_store_si128(reinterpret_cast<__m128i*>(destAligned), utf16Low);
            _mm_store_si128(reinterpret_cast<__m128i*>(destAligned) + 1, utf16High);
            memcpy(dest, destAligned, sizeof(__m128i) * 2);
            dest += dstAdvance;
        }
        // The rest will be handled sequencially.
        // Possible improvement: go back to the vectorized processing after the error or the 4 byte sequence
    }

    void UTF8ToWideImplSSE41(const unsigned char*& cur, const unsigned char* last, wchar32*& dest) noexcept {
        alignas(16) wchar32 destAligned[16];

        while (cur + 16 <= last) {
            __m128i utf16Low;
            __m128i utf16High;
            ui32 dstAdvance = Unpack16BytesIntoUtf16IfNoSurrogats(cur, utf16Low, utf16High);

            if (dstAdvance == 0) {
                break;
            }

            // NOTE: we only work in case without surrogat pairs, so we can make simple copying with zeroes in 2 high bytes
            __m128i utf32_lowlow = _mm_unpacklo_epi16(utf16Low, _mm_set1_epi8(0));
            __m128i utf32_lowhigh = _mm_unpackhi_epi16(utf16Low, _mm_set1_epi8(0));
            __m128i utf32_highlow = _mm_unpacklo_epi16(utf16High, _mm_set1_epi8(0));
            __m128i utf32_highhigh = _mm_unpackhi_epi16(utf16High, _mm_set1_epi8(0));

            _mm_store_si128(reinterpret_cast<__m128i*>(destAligned), utf32_lowlow);
            _mm_store_si128(reinterpret_cast<__m128i*>(destAligned) + 1, utf32_lowhigh);
            _mm_store_si128(reinterpret_cast<__m128i*>(destAligned) + 2, utf32_highlow);
            _mm_store_si128(reinterpret_cast<__m128i*>(destAligned) + 3, utf32_highhigh);

            memcpy(dest, destAligned, sizeof(__m128i) * 4);
            dest += dstAdvance;
        }
        // The rest will be handled sequencially.
        // Possible improvement: go back to the vectorized processing after the error or the 4 byte sequence
    }
}

#endif
