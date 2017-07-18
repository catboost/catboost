#if (defined(__ARM_NEON) && !defined(__ARM_NEON__))
#define __ARM_NEON__
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#include "libbase64.h"
#include "codecs.h"

#if (defined(__arm__) && defined(__ARM_NEON__))

#define CMPGT(s,n)	vcgtq_u8((s), vdupq_n_u8(n))
#define CMPEQ(s,n)	vceqq_u8((s), vdupq_n_u8(n))
#define REPLACE(s,n)	vandq_u8((s), vdupq_n_u8(n))
#define RANGE(s,a,b)	vandq_u8(vcgeq_u8((s), vdupq_n_u8(a)), vcleq_u8((s), vdupq_n_u8(b)))

static inline uint8x16x4_t
enc_reshuffle (uint8x16x3_t in)
{
	uint8x16x4_t out;

	// Divide bits of three input bytes over four output bytes:
	out.val[0] = vshrq_n_u8(in.val[0], 2);
	out.val[1] = vorrq_u8(vshrq_n_u8(in.val[1], 4), vshlq_n_u8(in.val[0], 4));
	out.val[2] = vorrq_u8(vshrq_n_u8(in.val[2], 6), vshlq_n_u8(in.val[1], 2));
	out.val[3] = in.val[2];

	// Clear top two bits:
	out.val[0] = vandq_u8(out.val[0], vdupq_n_u8(0x3F));
	out.val[1] = vandq_u8(out.val[1], vdupq_n_u8(0x3F));
	out.val[2] = vandq_u8(out.val[2], vdupq_n_u8(0x3F));
	out.val[3] = vandq_u8(out.val[3], vdupq_n_u8(0x3F));

	return out;
}

static inline uint8x16x4_t
enc_translate (uint8x16x4_t in)
{
	uint8x16x4_t mask1, mask2, mask3, mask4, out;

	// Translate values 0..63 to the Base64 alphabet. There are five sets:
	// #  From      To         Abs  Delta  Characters
	// 0  [0..25]   [65..90]   +65  +65    ABCDEFGHIJKLMNOPQRSTUVWXYZ
	// 1  [26..51]  [97..122]  +71   +6    abcdefghijklmnopqrstuvwxyz
	// 2  [52..61]  [48..57]    -4  -75    0123456789
	// 3  [62]      [43]       -19  -15    +
	// 4  [63]      [47]       -16   +3    /

	// Create cumulative masks for characters in sets [1,2,3,4], [2,3,4],
	// [3,4], and [4]:
	mask1.val[0] = CMPGT(in.val[0], 25);
	mask1.val[1] = CMPGT(in.val[1], 25);
	mask1.val[2] = CMPGT(in.val[2], 25);
	mask1.val[3] = CMPGT(in.val[3], 25);

	mask2.val[0] = CMPGT(in.val[0], 51);
	mask2.val[1] = CMPGT(in.val[1], 51);
	mask2.val[2] = CMPGT(in.val[2], 51);
	mask2.val[3] = CMPGT(in.val[3], 51);

	mask3.val[0] = CMPGT(in.val[0], 61);
	mask3.val[1] = CMPGT(in.val[1], 61);
	mask3.val[2] = CMPGT(in.val[2], 61);
	mask3.val[3] = CMPGT(in.val[3], 61);

	mask4.val[0] = CMPEQ(in.val[0], 63);
	mask4.val[1] = CMPEQ(in.val[1], 63);
	mask4.val[2] = CMPEQ(in.val[2], 63);
	mask4.val[3] = CMPEQ(in.val[3], 63);

	// All characters are at least in cumulative set 0, so add 'A':
	out.val[0] = vaddq_u8(in.val[0], vdupq_n_u8(65));
	out.val[1] = vaddq_u8(in.val[1], vdupq_n_u8(65));
	out.val[2] = vaddq_u8(in.val[2], vdupq_n_u8(65));
	out.val[3] = vaddq_u8(in.val[3], vdupq_n_u8(65));

	// For inputs which are also in any of the other cumulative sets,
	// add delta values against the previous set(s) to correct the shift:
	out.val[0] = vaddq_u8(out.val[0], REPLACE(mask1.val[0], 6));
	out.val[1] = vaddq_u8(out.val[1], REPLACE(mask1.val[1], 6));
	out.val[2] = vaddq_u8(out.val[2], REPLACE(mask1.val[2], 6));
	out.val[3] = vaddq_u8(out.val[3], REPLACE(mask1.val[3], 6));

	out.val[0] = vsubq_u8(out.val[0], REPLACE(mask2.val[0], 75));
	out.val[1] = vsubq_u8(out.val[1], REPLACE(mask2.val[1], 75));
	out.val[2] = vsubq_u8(out.val[2], REPLACE(mask2.val[2], 75));
	out.val[3] = vsubq_u8(out.val[3], REPLACE(mask2.val[3], 75));

	out.val[0] = vsubq_u8(out.val[0], REPLACE(mask3.val[0], 15));
	out.val[1] = vsubq_u8(out.val[1], REPLACE(mask3.val[1], 15));
	out.val[2] = vsubq_u8(out.val[2], REPLACE(mask3.val[2], 15));
	out.val[3] = vsubq_u8(out.val[3], REPLACE(mask3.val[3], 15));

	out.val[0] = vaddq_u8(out.val[0], REPLACE(mask4.val[0], 3));
	out.val[1] = vaddq_u8(out.val[1], REPLACE(mask4.val[1], 3));
	out.val[2] = vaddq_u8(out.val[2], REPLACE(mask4.val[2], 3));
	out.val[3] = vaddq_u8(out.val[3], REPLACE(mask4.val[3], 3));

	return out;
}

#endif

// Stride size is so large on these NEON 32-bit functions
// (48 bytes encode, 32 bytes decode) that we inline the
// uint32 codec to stay performant on smaller inputs.

void
neon32_base64_stream_encode
	( struct neon32_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	)
{
#if (defined(__arm__) && defined(__ARM_NEON__))
	#include "enc_head.c"
	#include "enc_neon.c"
	#include "enc_uint32.c"
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
neon32_base64_stream_decode
	( struct neon32_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	)
{
#if (defined(__arm__) && defined(__ARM_NEON__))
	#include "dec_head.c"
	#include "dec_neon.c"
	#include "dec_uint32.c"
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
