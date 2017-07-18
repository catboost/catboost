// If we have AVX2 support, pick off 24 bytes at a time for as long as we can.
// But because we read 32 bytes at a time, ensure we have enough room to do a
// full 32-byte read without segfaulting:
while (srclen >= 32)
{
	// Load string:
	__m256i str = _mm256_loadu_si256((__m256i *)c);

	// Reshuffle:
	str = enc_reshuffle(str);

	// Translate reshuffled bytes to the Base64 alphabet:
	str = enc_translate(str);

	// Store:
	_mm256_storeu_si256((__m256i *)o, str);

	c += 24;	// 6 * 4 bytes of input
	o += 32;	// 8 * 4 bytes of output
	outl += 32;
	srclen -= 24;
}
