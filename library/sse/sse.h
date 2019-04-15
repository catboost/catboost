#pragma once

/*
  The header chooses appropriate SSE support.
  On Intel: SSE intrinsics
  On ARM64: translation to NEON intrinsics or software emulation
  On PowerPc: translation to Altivec intrinsics or software emulation
*/
/* Author: Vitaliy Manushkin <agri@yandex-team.ru>, Danila Kutenin <danlark@yandex-team.ru> */

#include <util/system/platform.h>

#if (defined(_i386_) || defined(_x86_64_)) && defined(_sse_)
#include <xmmintrin.h>
#include <emmintrin.h>
#define ARCADIA_SSE
#elif defined(_arm64_)
#include "sse2neon.h"
#define ARCADIA_SSE
#elif defined(_ppc64_)
#include "powerpc.h"
#define ARCADIA_SSE
#endif
