#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include "libbase64.h"
#include "codecs.h"

#ifdef __SSSE3__
#include <tmmintrin.h>

#define CMPGT(s,n)	_mm_cmpgt_epi8((s), _mm_set1_epi8(n))
#define CMPEQ(s,n)	_mm_cmpeq_epi8((s), _mm_set1_epi8(n))
#define REPLACE(s,n)	_mm_and_si128((s), _mm_set1_epi8(n))
#define RANGE(s,a,b)	_mm_andnot_si128(CMPGT((s), (b)), CMPGT((s), (a) - 1))

static inline __m128i
_mm_bswap_epi32 (const __m128i in)
{
	return _mm_shuffle_epi8(in, _mm_setr_epi8(
		 3,  2,  1,  0,
		 7,  6,  5,  4,
		11, 10,  9,  8,
		15, 14, 13, 12));
}

static inline __m128i
enc_reshuffle (__m128i in)
{
	// Slice into 32-bit chunks and operate on all chunks in parallel.
	// All processing is done within the 32-bit chunk. First, shuffle:
	// before: [eeeeeeff|ccdddddd|bbbbcccc|aaaaaabb]
	// after:  [00000000|aaaaaabb|bbbbcccc|ccdddddd]
	in = _mm_shuffle_epi8(in, _mm_set_epi8(
		-1, 9, 10, 11,
		-1, 6,  7,  8,
		-1, 3,  4,  5,
		-1, 0,  1,  2));

	// cd      = [00000000|00000000|0000cccc|ccdddddd]
	const __m128i cd = _mm_and_si128(in, _mm_set1_epi32(0x00000FFF));

	// ab      = [0000aaaa|aabbbbbb|00000000|00000000]
	const __m128i ab = _mm_and_si128(_mm_slli_epi32(in, 4), _mm_set1_epi32(0x0FFF0000));

	// merged  = [0000aaaa|aabbbbbb|0000cccc|ccdddddd]
	const __m128i merged = _mm_or_si128(ab, cd);

	// bd      = [00000000|00bbbbbb|00000000|00dddddd]
	const __m128i bd = _mm_and_si128(merged, _mm_set1_epi32(0x003F003F));

	// ac      = [00aaaaaa|00000000|00cccccc|00000000]
	const __m128i ac = _mm_and_si128(_mm_slli_epi32(merged, 2), _mm_set1_epi32(0x3F003F00));

	// indices = [00aaaaaa|00bbbbbb|00cccccc|00dddddd]
	const __m128i indices = _mm_or_si128(ac, bd);

	// return  = [00dddddd|00cccccc|00bbbbbb|00aaaaaa]
	return _mm_bswap_epi32(indices);
}

static inline __m128i
enc_translate (const __m128i in)
{
	// Translate values 0..63 to the Base64 alphabet. There are five sets:
	// #  From      To         Abs  Delta  Characters
	// 0  [0..25]   [65..90]   +65  +65    ABCDEFGHIJKLMNOPQRSTUVWXYZ
	// 1  [26..51]  [97..122]  +71   +6    abcdefghijklmnopqrstuvwxyz
	// 2  [52..61]  [48..57]    -4  -75    0123456789
	// 3  [62]      [43]       -19  -15    +
	// 4  [63]      [47]       -16   +3    /

	// Create cumulative masks for characters in sets [1,2,3,4], [2,3,4],
	// [3,4], and [4]:
	const __m128i mask1 = CMPGT(in, 25);
	const __m128i mask2 = CMPGT(in, 51);
	const __m128i mask3 = CMPGT(in, 61);
	const __m128i mask4 = CMPEQ(in, 63);

	// All characters are at least in cumulative set 0, so add 'A':
	__m128i out = _mm_add_epi8(in, _mm_set1_epi8(65));

	// For inputs which are also in any of the other cumulative sets,
	// add delta values against the previous set(s) to correct the shift:
	out = _mm_add_epi8(out, REPLACE(mask1,  6));
	out = _mm_sub_epi8(out, REPLACE(mask2, 75));
	out = _mm_sub_epi8(out, REPLACE(mask3, 15));
	out = _mm_add_epi8(out, REPLACE(mask4,  3));

	return out;
}

static inline __m128i
dec_reshuffle (__m128i in)
{
	// Shuffle bytes to 32-bit bigendian:
	in = _mm_bswap_epi32(in);

	// Mask in a single byte per shift:
	__m128i mask = _mm_set1_epi32(0x3F000000);

	// Pack bytes together:
	__m128i out = _mm_slli_epi32(_mm_and_si128(in, mask), 2);
	mask = _mm_srli_epi32(mask, 8);

	out = _mm_or_si128(out, _mm_slli_epi32(_mm_and_si128(in, mask), 4));
	mask = _mm_srli_epi32(mask, 8);

	out = _mm_or_si128(out, _mm_slli_epi32(_mm_and_si128(in, mask), 6));
	mask = _mm_srli_epi32(mask, 8);

	out = _mm_or_si128(out, _mm_slli_epi32(_mm_and_si128(in, mask), 8));

	// Reshuffle and repack into 12-byte output format:
	return _mm_shuffle_epi8(out, _mm_setr_epi8(
		 3,  2,  1,
		 7,  6,  5,
		11, 10,  9,
		15, 14, 13,
		-1, -1, -1, -1));
}

#endif	// __SSSE3__

void
ssse3_base64_stream_encode
	( struct ssse3_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	)
{
#ifdef __SSSE3__
	#include "enc_head.c"
	#include "enc_ssse3.c"
	#include "enc_tail.c"
#else
    (void)state;
    (void)src;
    (void)srclen;
    (void)out;
    (void)outlen;
    abort();
#endif
}

int
ssse3_base64_stream_decode
	( struct ssse3_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	)
{
#ifdef __SSSE3__
	#include "dec_head.c"
	#include "dec_ssse3.c"
	#include "dec_tail.c"
#else
    (void)state;
    (void)src;
    (void)srclen;
    (void)out;
    (void)outlen;
    abort();
#endif
}
