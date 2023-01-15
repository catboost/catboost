//-----------------------------------------------------------------------------
// MurmurHash1 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#ifndef _MURMURHASH1_H_
#define _MURMURHASH1_H_

#include <stddef.h>

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER) && (_MSC_VER < 1600)

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned __int64 uint64_t;

// Other compilers

#else	// defined(_MSC_VER)

#include <stdint.h>

#endif // !defined(_MSC_VER)

//-----------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

uint32_t MurmurHash1        ( const void * key, size_t len, uint32_t seed );
uint32_t MurmurHash1Aligned ( const void * key, size_t len, uint32_t seed );

#ifdef __cplusplus
}
#endif

//-----------------------------------------------------------------------------

#endif // _MURMURHASH1_H_
