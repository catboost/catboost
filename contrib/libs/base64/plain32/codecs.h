#pragma once

// Define machine endianness. This is for GCC:
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
	#define PLAIN32_LITTLE_ENDIAN 1
#else
	#define PLAIN32_LITTLE_ENDIAN 0
#endif

// This is for Clang:
#ifdef __LITTLE_ENDIAN__
	#define PLAIN32_LITTLE_ENDIAN 1
#endif

#ifdef __BIG_ENDIAN__
	#define PLAIN32_LITTLE_ENDIAN 0
#endif

// Endian conversion functions
#if PLAIN32_LITTLE_ENDIAN
#if defined(_WIN64) || defined(__WIN32__) || defined(_WIN32)
	#define cpu_to_be32(x)	_byteswap_ulong(x)
	#define cpu_to_be64(x)	_byteswap_uint64(x)
	#define be32_to_cpu(x)	_byteswap_ulong(x)
	#define be64_to_cpu(x)	_byteswap_uint64(x)
#else
	#define cpu_to_be32(x)	__builtin_bswap32(x)
	#define cpu_to_be64(x)	__builtin_bswap64(x)
	#define be32_to_cpu(x)	__builtin_bswap32(x)
	#define be64_to_cpu(x)	__builtin_bswap64(x)
#endif
#else
	#define cpu_to_be32(x)	(x)
	#define cpu_to_be64(x)	(x)
	#define be32_to_cpu(x)	(x)
	#define be64_to_cpu(x)	(x)
#endif

// These tables are used by all codecs
// for fallback plain encoding/decoding:
extern const uint8_t plain32_base64_table_enc[];
extern const uint8_t plain32_base64_table_dec[];
