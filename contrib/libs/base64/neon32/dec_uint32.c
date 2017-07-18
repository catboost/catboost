// If we have native uint32's, pick off 4 bytes at a time for as long as we
// can, but make sure that we quit before seeing any == markers at the end of
// the string. Also, because we write a zero at the end of the output, ensure
// that there are at least 2 valid bytes of input data remaining to close the
// gap. 4 + 2 + 2 = 8 bytes:
while (srclen >= 8)
{
	uint32_t str, res, dec;

	// Load string:
	str = *(uint32_t *)c;

	// Shuffle bytes to 32-bit bigendian:
	str = cpu_to_be32(str);

	// Lookup each byte in the decoding table; if we encounter any
	// "invalid" values, fall back on the bytewise code:
	if ((dec = neon32_base64_table_dec[str >> 24]) > 63) {
		break;
	}
	res = dec << 26;

	if ((dec = neon32_base64_table_dec[(str >> 16) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 20;

	if ((dec = neon32_base64_table_dec[(str >> 8) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 14;

	if ((dec = neon32_base64_table_dec[str & 0xFF]) > 63) {
		break;
	}
	res |= dec << 8;

	// Reshuffle and repack into 3-byte output format:
	res = be32_to_cpu(res);

	// Store back:
	*(uint32_t *)o = res;

	c += 4;
	o += 3;
	outl += 3;
	srclen -= 4;
}
