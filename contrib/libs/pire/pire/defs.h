/*
 * defs.h -- common Pire definitions.
 *
 * Copyright (c) 2007-2010, Dmitry Prokoptsev <dprokoptsev@gmail.com>,
 *                          Alexander Gololobov <agololobov@gmail.com>
 *
 * This file is part of Pire, the Perl Incompatible
 * Regular Expressions library.
 *
 * Pire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Pire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * You should have received a copy of the GNU Lesser Public License
 * along with Pire.  If not, see <http://www.gnu.org/licenses>.
 */


#ifndef PIRE_DEFS_H
#define PIRE_DEFS_H

#ifndef PIRE_NO_CONFIG
#include <pire/config.h>
#endif
#include <stdlib.h>

#if defined(_MSC_VER)
#define PIRE_HAVE_DECLSPEC_ALIGN
#else
#define PIRE_HAVE_ALIGNAS
#endif

#define PIRE_HAVE_LAMBDAS

namespace Pire {

#ifdef PIRE_DEBUG
#	define PIRE_IFDEBUG(x) x
#else
#	define PIRE_IFDEBUG(x)
#endif

#ifdef PIRE_CHECKED
#	define PIRE_IF_CHECKED(e) e
#else
#	define PIRE_IF_CHECKED(e)
#endif


	typedef unsigned short Char;

	namespace SpecialChar {
	enum {
		Epsilon = 257,
		BeginMark = 258,
		EndMark = 259,

		// Actual size of input alphabet
		MaxCharUnaligned = 260,

		// Size of letter transition tables, must be a multiple of the machine word size
		MaxChar = (MaxCharUnaligned + (sizeof(void*)-1)) & ~(sizeof(void*)-1)
	};
	}

	using namespace SpecialChar;

	namespace Impl {
#ifndef PIRE_WORDS_BIGENDIAN
		inline size_t ToLittleEndian(size_t val) { return val; }
#else
		template<unsigned N>
		inline size_t SwapBytes(size_t val)
		{
			static const size_t Mask = (1 << (N/2)) - 1;
			return ((SwapBytes<N/2>(val) & Mask) << (N/2)) | SwapBytes<N/2>(val >> (N/2));
		}

		template<>
		inline size_t SwapBytes<8>(size_t val) { return val & 0xFF; }

		inline size_t ToLittleEndian(size_t val) { return SwapBytes<sizeof(val)*8>(val); }
#endif

        struct Struct { void* p; };
	}
}

#ifndef PIRE_ALIGNED_DECL
#	if defined(PIRE_HAVE_ALIGNAS)
#		define PIRE_ALIGNED_DECL(x) alignas(::Pire::Impl::Struct) static const char x[]
#	elif defined(PIRE_HAVE_ATTR_ALIGNED)
#		define PIRE_ALIGNED_DECL(x) static const char x[] __attribute__((aligned(sizeof(void*))))
#	elif defined(PIRE_HAVE_DECLSPEC_ALIGN)
#		define PIRE_ALIGNED_DECL(x) __declspec(align(8)) static const char x[]
#	endif
#endif

#ifndef PIRE_LITERAL
#	if defined(PIRE_HAVE_LAMBDAS)
#		define PIRE_LITERAL(data) ([]() -> const char* { PIRE_ALIGNED_DECL(__pire_regexp__) = data; return __pire_regexp__; })()
#	elif defined(PIRE_HAVE_SCOPED_EXPR)
#		define PIRE_LITERAL(data) ({ PIRE_ALIGNED_DECL(__pire_regexp__) = data; __pire_regexp__; })
#	endif
#endif

#endif
