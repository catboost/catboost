#pragma once

// Define machine endianness. This is for GCC:
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
	#define BASE64_SSSE3_LITTLE_ENDIAN 1
#else
	#define BASE64_SSSE3_LITTLE_ENDIAN 0
#endif

// This is for Clang:
#ifdef __LITTLE_ENDIAN__
	#define BASE64_SSSE3_LITTLE_ENDIAN 1
#endif

#ifdef __BIG_ENDIAN__
	#define BASE64_SSSE3_LITTLE_ENDIAN 0
#endif

// Endian conversion functions
#if BASE64_SSSE3_LITTLE_ENDIAN
	#define cpu_to_be32(x)	__builtin_bswap32(x)
	#define cpu_to_be64(x)	__builtin_bswap64(x)
	#define be32_to_cpu(x)	__builtin_bswap32(x)
	#define be64_to_cpu(x)	__builtin_bswap64(x)
#else
	#define cpu_to_be32(x)	(x)
	#define cpu_to_be64(x)	(x)
	#define be32_to_cpu(x)	(x)
	#define be64_to_cpu(x)	(x)
#endif

// These tables are used by all codecs
// for fallback plain encoding/decoding:
extern const uint8_t ssse3_base64_table_enc[];
extern const uint8_t ssse3_base64_table_dec[];
