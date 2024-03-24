// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       crc_common.h
/// \brief      Some functions and macros for CRC32 and CRC64
//
//  Authors:    Lasse Collin
//              Ilya Kurdyukov
//              Hans Jansen
//              Jia Tan
//
///////////////////////////////////////////////////////////////////////////////

#ifndef LZMA_CRC_COMMON_H
#define LZMA_CRC_COMMON_H

#include "common.h"


#ifdef WORDS_BIGENDIAN
#	define A(x) ((x) >> 24)
#	define B(x) (((x) >> 16) & 0xFF)
#	define C(x) (((x) >> 8) & 0xFF)
#	define D(x) ((x) & 0xFF)

#	define S8(x) ((x) << 8)
#	define S32(x) ((x) << 32)

#else
#	define A(x) ((x) & 0xFF)
#	define B(x) (((x) >> 8) & 0xFF)
#	define C(x) (((x) >> 16) & 0xFF)
#	define D(x) ((x) >> 24)

#	define S8(x) ((x) >> 8)
#	define S32(x) ((x) >> 32)
#endif


// CRC CLMUL code needs this because accessing input buffers that aren't
// aligned to the vector size will inherently trip the address sanitizer.
#if lzma_has_attribute(__no_sanitize_address__)
#	define crc_attr_no_sanitize_address \
			__attribute__((__no_sanitize_address__))
#else
#	define crc_attr_no_sanitize_address
#endif

// Keep this in sync with changes to crc32_arm64.h
#if defined(_WIN32) || defined(HAVE_GETAUXVAL) \
		|| defined(HAVE_ELF_AUX_INFO) \
		|| (defined(__APPLE__) && defined(HAVE_SYSCTLBYNAME))
#	define ARM64_RUNTIME_DETECTION 1
#endif


#undef CRC32_GENERIC
#undef CRC64_GENERIC

#undef CRC32_ARCH_OPTIMIZED
#undef CRC64_ARCH_OPTIMIZED

// The x86 CLMUL is used for both CRC32 and CRC64.
#undef CRC_X86_CLMUL

#undef CRC32_ARM64
#undef CRC64_ARM64_CLMUL

#undef CRC_USE_IFUNC

#undef CRC_USE_GENERIC_FOR_SMALL_INPUTS

// ARM64 CRC32 instruction is only useful for CRC32. Currently, only
// little endian is supported since we were unable to test on a big
// endian machine.
//
// NOTE: Keep this and the next check in sync with the macro
//       ARM64_CRC32_NO_TABLE in crc32_table.c
#if defined(HAVE_ARM64_CRC32) && !defined(WORDS_BIGENDIAN)
// Allow ARM64 CRC32 instruction without a runtime check if
// __ARM_FEATURE_CRC32 is defined. GCC and Clang only define this if the
// proper compiler options are used.
#	if defined(__ARM_FEATURE_CRC32)
#		define CRC32_ARCH_OPTIMIZED 1
#		define CRC32_ARM64 1
#	elif defined(ARM64_RUNTIME_DETECTION)
#		define CRC32_ARCH_OPTIMIZED 1
#		define CRC32_ARM64 1
#		define CRC32_GENERIC 1
#	endif
#endif

#if defined(HAVE_USABLE_CLMUL)
// If CLMUL is allowed unconditionally in the compiler options then the
// generic version can be omitted. Note that this doesn't work with MSVC
// as I don't know how to detect the features here.
//
// NOTE: Keep this in sync with the CLMUL_NO_TABLE macro in crc32_table.c.
#	if (defined(__SSSE3__) && defined(__SSE4_1__) && defined(__PCLMUL__)) \
		|| (defined(__e2k__) && __iset__ >= 6)
#		define CRC32_ARCH_OPTIMIZED 1
#		define CRC64_ARCH_OPTIMIZED 1
#		define CRC_X86_CLMUL 1
#	else
#		define CRC32_GENERIC 1
#		define CRC64_GENERIC 1
#		define CRC32_ARCH_OPTIMIZED 1
#		define CRC64_ARCH_OPTIMIZED 1
#		define CRC_X86_CLMUL 1

#		ifdef HAVE_FUNC_ATTRIBUTE_IFUNC
#			define CRC_USE_IFUNC 1
#		endif
/*
		// The generic code is much faster with 1-8-byte inputs and
		// has similar performance up to 16 bytes  at least in
		// microbenchmarks (it depends on input buffer alignment
		// too). If both versions are built, this #define will use
		// the generic version for inputs up to 16 bytes and CLMUL
		// for bigger inputs. It saves a little in code size since
		// the special cases for 0-16-byte inputs will be omitted
		// from the CLMUL code.
#		ifndef CRC_USE_IFUNC
#			define CRC_USE_GENERIC_FOR_SMALL_INPUTS 1
#		endif
*/
#	endif
#endif

#ifdef CRC_USE_IFUNC
// Two function attributes are needed to make IFUNC safe with GCC.
//
// no-omit-frame-pointer prevents false Valgrind issues when combined with
// a few other compiler flags. The optimize attribute is supported on
// GCC >= 4.4 and is not supported with Clang.
#	if TUKLIB_GNUC_REQ(4,4) && !defined(__clang__)
#		define no_omit_frame_pointer \
			__attribute__((optimize("no-omit-frame-pointer")))
#	else
#		define no_omit_frame_pointer
#	endif

// The __no_profile_instrument_function__ attribute support is checked when
// determining if ifunc can be used, so it is safe to use unconditionally.
// This attribute is needed because GCC can add profiling to the IFUNC
// resolver, which calls functions that have not yet been relocated leading
// to a crash on liblzma start up.
#	define lzma_resolver_attributes \
		__attribute__((__no_profile_instrument_function__)) \
		no_omit_frame_pointer
#else
#	define lzma_resolver_attributes
#endif

// For CRC32 use the generic slice-by-eight implementation if no optimized
// version is available.
#if !defined(CRC32_ARCH_OPTIMIZED) && !defined(CRC32_GENERIC)
#	define CRC32_GENERIC 1
#endif

// For CRC64 use the generic slice-by-four implementation if no optimized
// version is available.
#if !defined(CRC64_ARCH_OPTIMIZED) && !defined(CRC64_GENERIC)
#	define CRC64_GENERIC 1
#endif

#endif
