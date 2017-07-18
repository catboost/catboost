// If we have NEON support, pick off 64 bytes at a time for as long as we can.
// Unlike the SSE codecs, we don't write trailing zero bytes to output, so we
// don't need to check if we have enough remaining input to cover them:
while (srclen >= 64)
{
	uint8x16x4_t set1, set2, set3, set4, set5, set6, set7, delta;
	uint8x16x3_t dec;

	// Load 64 bytes and deinterleave:
	uint8x16x4_t str = vld4q_u8((uint8_t *)c);

	// The input consists of six character sets in the Base64 alphabet,
	// which we need to map back to the 6-bit values they represent.
	// There are three ranges, two singles, and then there's the rest.
	//
	//  #  From       To        Add  Characters
	//  1  [43]       [62]      +19  +
	//  2  [47]       [63]      +16  /
	//  3  [48..57]   [52..61]   +4  0..9
	//  4  [65..90]   [0..25]   -65  A..Z
	//  5  [97..122]  [26..51]  -71  a..z
	// (6) Everything else => invalid input

	// Benchmarking on the Raspberry Pi 2B and Clang shows that looping
	// generates slightly faster code than explicit unrolling:
	for (int i = 0; i < 4; i++) {
		set1.val[i] = CMPEQ(str.val[i], '+');
		set2.val[i] = CMPEQ(str.val[i], '/');
		set3.val[i] = RANGE(str.val[i], '0', '9');
		set4.val[i] = RANGE(str.val[i], 'A', 'Z');
		set5.val[i] = RANGE(str.val[i], 'a', 'z');
		set6.val[i] = CMPEQ(str.val[i], '-');
		set7.val[i] = CMPEQ(str.val[i], '_');

		delta.val[i] = REPLACE(set1.val[i], 19);
		delta.val[i] = vorrq_u8(delta.val[i], REPLACE(set2.val[i],  16));
		delta.val[i] = vorrq_u8(delta.val[i], REPLACE(set3.val[i],   4));
		delta.val[i] = vorrq_u8(delta.val[i], REPLACE(set4.val[i], -65));
		delta.val[i] = vorrq_u8(delta.val[i], REPLACE(set5.val[i], -71));
		delta.val[i] = vorrq_u8(delta.val[i], REPLACE(set6.val[i], 17));
		delta.val[i] = vorrq_u8(delta.val[i], REPLACE(set7.val[i], -32));
	}

	// Check for invalid input: if any of the delta values are zero,
	// fall back on bytewise code to do error checking and reporting:
	uint8x16_t classified = CMPEQ(delta.val[0], 0);
	classified = vorrq_u8(classified, CMPEQ(delta.val[1], 0));
	classified = vorrq_u8(classified, CMPEQ(delta.val[2], 0));
	classified = vorrq_u8(classified, CMPEQ(delta.val[3], 0));

	// Extract both 32-bit halves; check that all bits are zero:
	if (vgetq_lane_u32((uint32x4_t)classified, 0) != 0
	 || vgetq_lane_u32((uint32x4_t)classified, 1) != 0
	 || vgetq_lane_u32((uint32x4_t)classified, 2) != 0
	 || vgetq_lane_u32((uint32x4_t)classified, 3) != 0) {
		break;
	}

	// Now simply add the delta values to the input:
	str.val[0] = vaddq_u8(str.val[0], delta.val[0]);
	str.val[1] = vaddq_u8(str.val[1], delta.val[1]);
	str.val[2] = vaddq_u8(str.val[2], delta.val[2]);
	str.val[3] = vaddq_u8(str.val[3], delta.val[3]);

	// Compress four bytes into three:
	dec.val[0] = vshlq_n_u8(str.val[0], 2) | vshrq_n_u8(str.val[1], 4);
	dec.val[1] = vshlq_n_u8(str.val[1], 4) | vshrq_n_u8(str.val[2], 2);
	dec.val[2] = vshlq_n_u8(str.val[2], 6) | str.val[3];

	// Interleave and store decoded result:
	vst3q_u8((uint8_t *)o, dec);

	c += 64;
	o += 48;
	outl += 48;
	srclen -= 64;
}
