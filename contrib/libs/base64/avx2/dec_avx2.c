// If we have AVX2 support, pick off 32 bytes at a time for as long as we can,
// but make sure that we quit before seeing any == markers at the end of the
// string. Also, because we write 8 zeroes at the end of the output, ensure
// that there are at least 11 valid bytes of input data remaining to close the
// gap. 32 + 2 + 11 = 45 bytes:
while (srclen >= 45)
{
	// Load string:
	__m256i str = _mm256_loadu_si256((__m256i *)c);

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

	const __m256i set1 = CMPEQ(str, '+');
	const __m256i set2 = CMPEQ(str, '/');
	const __m256i set3 = RANGE(str, '0', '9');
	const __m256i set4 = RANGE(str, 'A', 'Z');
	const __m256i set5 = RANGE(str, 'a', 'z');
	const __m256i set6 = CMPEQ(str, '-');
	const __m256i set7 = CMPEQ(str, '_');

	__m256i delta = REPLACE(set1, 19);
	delta = _mm256_or_si256(delta, REPLACE(set2,  16));
	delta = _mm256_or_si256(delta, REPLACE(set3,   4));
	delta = _mm256_or_si256(delta, REPLACE(set4, -65));
	delta = _mm256_or_si256(delta, REPLACE(set5, -71));
	delta = _mm256_or_si256(delta, REPLACE(set6, 17));
	delta = _mm256_or_si256(delta, REPLACE(set7, -32));

	// Check for invalid input: if any of the delta values are zero,
	// fall back on bytewise code to do error checking and reporting:
#ifdef _MSC_VER
	// Hack for MSVC miscompilation - it inserts vzeroupper for the break
	// (we need to clear YMM registers before exiting the function)
	// while delta and str are still in the registers.
	// Save delta/str in memory manually.
	_mm256_zeroupper();
#endif    
	if (_mm256_movemask_epi8(CMPEQ(delta, 0))) {
		break;
	}

	// Now simply add the delta values to the input:
	str = _mm256_add_epi8(str, delta);

	// Reshuffle the input to packed 12-byte output format:
	str = dec_reshuffle(str);

	// Store back:
	_mm256_storeu_si256((__m256i *)o, str);

	c += 32;
	o += 24;
	outl += 24;
	srclen -= 32;
}
