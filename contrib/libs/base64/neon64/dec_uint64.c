// If we have native uint64's, pick off 8 bytes at a time for as long as we
// can, but make sure that we quit before seeing any == markers at the end of
// the string. Also, because we write two zeroes at the end of the output,
// ensure that there are at least 3 valid bytes of input data remaining to
// close the gap. 8 + 2 + 3 = 13 bytes:
while (srclen >= 13)
{
	uint64_t str, res, dec;

	// Load string:
	str = *(uint64_t *)c;

	// Shuffle bytes to 64-bit bigendian:
	str = cpu_to_be64(str);

	// Lookup each byte in the decoding table; if we encounter any
	// "invalid" values, fall back on the bytewise code:
	if ((dec = neon64_base64_table_dec[str >> 56]) > 63) {
		break;
	}
	res = dec << 58;

	if ((dec = neon64_base64_table_dec[(str >> 48) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 52;

	if ((dec = neon64_base64_table_dec[(str >> 40) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 46;

	if ((dec = neon64_base64_table_dec[(str >> 32) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 40;

	if ((dec = neon64_base64_table_dec[(str >> 24) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 34;

	if ((dec = neon64_base64_table_dec[(str >> 16) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 28;

	if ((dec = neon64_base64_table_dec[(str >> 8) & 0xFF]) > 63) {
		break;
	}
	res |= dec << 22;

	if ((dec = neon64_base64_table_dec[str & 0xFF]) > 63) {
		break;
	}
	res |= dec << 16;

	// Reshuffle and repack into 6-byte output format:
	res = be64_to_cpu(res);

	// Store back:
	*(uint64_t *)o = res;

	c += 8;
	o += 6;
	outl += 6;
	srclen -= 8;
}
