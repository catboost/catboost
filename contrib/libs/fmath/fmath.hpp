#pragma once
/**
	@brief fast math library for float
	@author herumi
	@url https://github.com/herumi/fmath/
	@note modified new BSD license
	http://opensource.org/licenses/BSD-3-Clause

	cl /Ox /Ob2 /arch:SSE2 /fp:fast bench.cpp -I../xbyak /EHsc /DNOMINMAX
	g++ -O3 -fomit-frame-pointer -fno-operator-names -march=core2 -mssse3 -mfpmath=sse -ffast-math -fexcess-precision=fast
*/
/*
	function prototype list

	float fmath::exp(float);
	double fmath::expd(double);
	float fmath::log(float);

	__m128 fmath::exp_ps(__m128);
	__m256 fmath::exp_ps256(__m256);
	__m128 fmath::log_ps(__m128);

	double fmath::expd_v(double *, size_t n);

	if FMATH_USE_XBYAK is defined then Xbyak version are used
*/
//#define FMATH_USE_XBYAK

#include <math.h>
#include <stddef.h>
#include <assert.h>
#include <limits>
#include <stdlib.h>
#include <float.h>
#include <string.h> // for memcpy
#if defined(_WIN32) && !defined(__GNUC__)
	#include <intrin.h>
	#ifndef MIE_ALIGN
		#define MIE_ALIGN(x) __declspec(align(x))
	#endif
#else
	#ifndef __GNUC_PREREQ
	#define __GNUC_PREREQ(major, minor) ((((__GNUC__) << 16) + (__GNUC_MINOR__)) >= (((major) << 16) + (minor)))
	#endif
	#if __GNUC_PREREQ(4, 4) || (__clang__ > 0 && __clang_major__ >= 3) || !defined(__GNUC__)
		/* GCC >= 4.4 or clang or non-GCC compilers */
		#include <x86intrin.h>
	#elif __GNUC_PREREQ(4, 1)
		/* GCC 4.1, 4.2, and 4.3 do not have x86intrin.h, directly include SSE2 header */
		#include <emmintrin.h>
	#endif
	#ifndef MIE_ALIGN
		#define MIE_ALIGN(x) __attribute__((aligned(x)))
	#endif
#endif
#ifndef MIE_PACK
	#define MIE_PACK(x, y, z, w) ((x) * 64 + (y) * 16 + (z) * 4 + (w))
#endif
//#ifdef FMATH_USE_XBYAK
//	#define XBYAK_NO_OP_NAMES
//	#include "xbyak/xbyak.h"
//	#include "xbyak/xbyak_util.h"
//#endif

namespace {

namespace fmath {

namespace local {

const size_t EXP_TABLE_SIZE = 10;
const size_t EXPD_TABLE_SIZE = 11;
const size_t LOG_TABLE_SIZE = 12;

typedef unsigned long long uint64_t;

union fi {
	float f;
	unsigned int i;
};

union di {
	double d;
	uint64_t i;
};

inline unsigned int mask(int x)
{
	return (1U << x) - 1;
}

inline uint64_t mask64(int x)
{
	return (1ULL << x) - 1;
}

template<class T>
inline const T* cast_to(const void *p)
{
	return reinterpret_cast<const T*>(p);
}

template<class T, size_t N>
size_t NumOfArray(const T (&)[N]) { return N; }

/*
	exp(88.722839f) = inf ; 0x42b17218
	exp(-87.33655f) = 1.175491e-038f(007fffe6) denormal ; 0xc2aeac50
	exp(-103.972081f) = 0 ; 0xc2cff1b5
*/
template<size_t N = EXP_TABLE_SIZE>
struct ExpVar {
	enum {
		s = N,
		n = 1 << s,
		f88 = 0x42b00000 /* 88.0 */
	};
	float minX[8];
	float maxX[8];
	float a[8];
	float b[8];
	float f1[8];
	unsigned int i127s[8];
	unsigned int mask_s[8];
	unsigned int i7fffffff[8];
	unsigned int tbl[n];
	ExpVar()
	{
		float log_2 = ::logf(2.0f);
		for (int i = 0; i < 8; i++) {
			maxX[i] = 88;
			minX[i] = -88;
			a[i] = n / log_2;
			b[i] = log_2 / n;
			f1[i] = 1.0f;
			i127s[i] = 127 << s;
			i7fffffff[i] = 0x7fffffff;
			mask_s[i] = mask(s);
		}

		for (int i = 0; i < n; i++) {
			float y = pow(2.0f, (float)i / n);
			fi fi;
			fi.f = y;
			tbl[i] = fi.i & mask(23);
		}
	}
};

template<size_t sbit_ = EXPD_TABLE_SIZE>
struct ExpdVar {
	enum {
		sbit = sbit_,
		s = 1UL << sbit,
		adj = (1UL << (sbit + 10)) - (1UL << sbit)
	};
	// A = 1, B = 1, C = 1/2, D = 1/6
	double C1[2]; // A
	double C2[2]; // D
	double C3[2]; // C/D
	uint64_t tbl[s];
	double a;
	double ra;
	ExpdVar()
		: a(s / ::log(2.0))
		, ra(1 / a)
	{
		for (int i = 0; i < 2; i++) {
#if 0
			C1[i] = 1.0;
			C2[i] = 0.16667794882310216;
			C3[i] = 2.9997969303278795;
#else
			C1[i] = 1.0;
			C2[i] = 0.16666666685227835064;
			C3[i] = 3.0000000027955394;
#endif
		}
		for (int i = 0; i < s; i++) {
			di di;
			di.d = ::pow(2.0, i * (1.0 / s));
			tbl[i] = di.i & mask64(52);
		}
	}
};

template<size_t N = LOG_TABLE_SIZE>
struct LogVar {
	enum {
		LEN = N - 1
	};
	unsigned int m1[4]; // 0
	unsigned int m2[4]; // 16
	unsigned int m3[4]; // 32
	float m4[4];		// 48
	unsigned int m5[4]; // 64
	struct {
		float app;
		float rev;
	} tbl[1 << LEN];
	float c_log2;
	LogVar()
		: c_log2(::logf(2.0f) / (1 << 23))
	{
		const double e = 1 / double(1 << 24);
		const double h = 1 / double(1 << LEN);
		const size_t n = 1U << LEN;
		for (size_t i = 0; i < n; i++) {
			double x = 1 + double(i) / n;
			double a = ::log(x);
			tbl[i].app = (float)a;
			if (i < n - 1) {
				double b = ::log(x + h - e);
				tbl[i].rev = (float)((b - a) / ((h - e) * (1 << 23)));
			} else {
				tbl[i].rev = (float)(1 / (x * (1 << 23)));
			}
		}
		for (int i = 0; i < 4; i++) {
			m1[i] = mask(8) << 23;
			m2[i] = mask(LEN) << (23 - LEN);
			m3[i] = mask(23 - LEN);
			m4[i] = c_log2;
			m5[i] = 127U << 23;
		}
	}
};

#ifdef FMATH_USE_XBYAK
struct ExpCode : public Xbyak::CodeGenerator {
	float (*exp_)(float);
	__m128 (*exp_ps_)(__m128);
	template<size_t N>
	ExpCode(const ExpVar<N> *self)
	{
		Xbyak::util::Cpu cpu;
		try {
			makeExp(self, cpu);
			exp_ = getCode<float (*)(float)>();
			align(16);
			exp_ps_ = getCurr<__m128(*)(__m128)>();
			makeExpPs(self, cpu);
			return;
		} catch (std::exception& e) {
			fprintf(stderr, "ExpCode ERR:%s\n", e.what());
		} catch (...) {
			fprintf(stderr, "ExpCode ERR:unknown error\n");
		}
		::exit(1);
	}
	template<size_t N>
	void makeExp(const ExpVar<N> *self, const Xbyak::util::Cpu& /*cpu*/)
	{
		typedef ExpVar<N> Self;
		using namespace local;
		using namespace Xbyak;

		inLocalLabel();
#ifdef XBYAK64
		const Reg64& base = rcx;
		const Reg64& a = rax;
#else
		const Reg32& base = ecx;
		const Reg32& a = eax;
#endif

		mov(base, (size_t)self);

#ifdef XBYAK32
		movss(xm0, ptr [esp + 4]);
#endif
	L(".retry");
		movaps(xm1, xm0);
		movd(edx, xm0);
		mulss(xm1, ptr [base + offsetof(Self, a)]); // t
		and_(edx, 0x7fffffff);
		cvtss2si(eax, xm1);
		cmp(edx, ExpVar<N>::f88);
		jg(".overflow");
		lea(edx, ptr [eax + (127 << self->s)]);
		cvtsi2ss(xm1, eax);
		and_(eax, mask(self->s)); // v
		mov(eax, ptr [base + a * 4 + offsetof(Self, tbl)]); // expVar.tbl[v]
		shr(edx, self->s);
		mulss(xm1, ptr [base + offsetof(Self, b)]);
		shl(edx, 23); // u
		subss(xm0, xm1); // t
		or_(eax, edx); // fi.f
		addss(xm0, ptr [base + offsetof(Self, f1)]);
		movd(xm1, eax);
		mulss(xm0, xm1);
#ifdef XBYAK32
		movss(ptr[esp + 4], xm0);
		fld(dword[esp + 4]);
#endif
		ret();
	L(".overflow");
		minss(xm0, ptr [base + offsetof(Self, maxX)]);
		maxss(xm0, ptr [base + offsetof(Self, minX)]);
		jmp(".retry");
		outLocalLabel();
	}
	template<size_t N>
	void makeExpPs(const ExpVar<N> *self, const Xbyak::util::Cpu& cpu)
	{
		typedef ExpVar<N> Self;
		using namespace local;
		using namespace Xbyak;

		inLocalLabel();
#ifdef XBYAK64
		const Reg64& base = rcx;
		const Reg64& a = rax;
		const Reg64& d = rdx;
#else
		const Reg32& base = ecx;
		const Reg32& a = eax;
		const Reg32& d = edx;
#endif

/*
	if abs(x) >= maxX then x = max(min(x, maxX), -maxX) and try
	minps, maxps are very slow then avoid them
*/
		const bool useSSE41 = cpu.has(Xbyak::util::Cpu::tSSE41);
#if defined(XBYAK64_WIN) && !defined(__INTEL_COMPILER)
		movaps(xm0, ptr [rcx]);
#endif
		mov(base, (size_t)self);
	L(".retry");
		movaps(xm5, xm0);
		andps(xm5, ptr [base + offsetof(Self, i7fffffff)]);
		movaps(xm3, ptr [base + offsetof(Self, a)]);
		movaps(xm4, ptr [base + offsetof(Self, b)]);
		pcmpgtd(xm5, ptr [base + offsetof(Self, maxX)]);
		mulps(xm3, xm0);
		movaps(xm1, ptr [base + offsetof(Self, i127s)]);
		pmovmskb(eax, xm5);
		movaps(xm5, ptr [base + offsetof(Self, mask_s)]);
		cvtps2dq(xm2, xm3);
		pand(xm5, xm2);
		cvtdq2ps(xm3, xm2);
		test(eax, eax);
		jnz(".overflow");
		paddd(xm1, xm2);
		movd(eax, xm5);
		mulps(xm4, xm3);
		pextrw(edx, xm5, 2);
		subps(xm0, xm4);
		movd(xm4, ptr [base + a * 4 + offsetof(Self, tbl)]);
		addps(xm0, ptr [base + offsetof(Self, f1)]);
		pextrw(eax, xm5, 4);
		if (useSSE41) {
			pinsrd(xm4, ptr [base + d * 4 + offsetof(Self, tbl)], 1);
		} else {
			movd(xm3, ptr [base + d * 4 + offsetof(Self, tbl)]);
			movlhps(xm4, xm3);
		}
		pextrw(edx, xm5, 6);
		psrld(xm1, self->s);
		pslld(xm1, 23);
		if (useSSE41) {
			pinsrd(xm4, ptr [base + a * 4 + offsetof(Self, tbl)], 2);
			pinsrd(xm4, ptr [base + d * 4 + offsetof(Self, tbl)], 3);
		} else {
			movd(xm2, ptr [base + a * 4 + offsetof(Self, tbl)]);
			movd(xm3, ptr [base + d * 4 + offsetof(Self, tbl)]);
			movlhps(xm2, xm3);
			shufps(xm4, xm2, MIE_PACK(2, 0, 2, 0));
		}
		por(xm1, xm4);
		mulps(xm0, xm1);
		ret();
	L(".overflow");
		minps(xm0, ptr [base + offsetof(Self, maxX)]);
		maxps(xm0, ptr [base + offsetof(Self, minX)]);
		jmp(".retry");
		outLocalLabel();
	}
};
#endif

/* to define static variables in fmath.hpp */
template<size_t EXP_N = EXP_TABLE_SIZE, size_t LOG_N = LOG_TABLE_SIZE, size_t EXPD_N = EXPD_TABLE_SIZE>
struct C {
	static const ExpVar<EXP_N>& expVar() {
		static MIE_ALIGN(32) const ExpVar<EXP_N> var;
		return var;
	}
	static const LogVar<LOG_N>& logVar() {
		static MIE_ALIGN(32) const LogVar<LOG_N> var;
		return var;
	}
	static const ExpdVar<EXPD_N>& expdVar() {
		static MIE_ALIGN(32) const ExpdVar<EXPD_N> var;
		return var;
	}
#ifdef FMATH_USE_XBYAK
	static const ExpCode& getInstance() {
		static const MIE_ALIGN(32) ExpCode expCode(&expVar);
		return expCode;
	}
#endif
};
#if 0
/* to define static variables in fmath.hpp */
template<size_t EXP_N = EXP_TABLE_SIZE, size_t LOG_N = LOG_TABLE_SIZE, size_t EXPD_N = EXPD_TABLE_SIZE>
struct C {
	static const ExpVar<EXP_N> expVar;
	static const LogVar<LOG_N> logVar;
	static const ExpdVar<EXPD_N> expdVar;
#ifdef FMATH_USE_XBYAK
	static const ExpCode& getInstance() {
		static const ExpCode expCode(&expVar);
		return expCode;
	}
#endif
};

template<size_t EXP_N, size_t LOG_N, size_t EXPD_N>
MIE_ALIGN(32) const ExpVar<EXP_N> C<EXP_N, LOG_N, EXPD_N>::expVar;

template<size_t EXP_N, size_t LOG_N, size_t EXPD_N>
MIE_ALIGN(32) const LogVar<LOG_N> C<EXP_N, LOG_N, EXPD_N>::logVar;

template<size_t EXP_N, size_t LOG_N, size_t EXPD_N>
MIE_ALIGN(32) const ExpdVar<EXPD_N> C<EXP_N, LOG_N, EXPD_N>::expdVar;
#endif

} // fmath::local

#ifdef FMATH_USE_XBYAK
inline float expC(float x)
#else
inline float exp(float x)
#endif
{
	using namespace local;
	const ExpVar<>& expVar = C<>::expVar();

#if 1
	__m128 x1 = _mm_set_ss(x);

	int limit = _mm_cvtss_si32(x1) & 0x7fffffff;
	if (limit > ExpVar<>::f88) {
		x1 = _mm_min_ss(x1, _mm_load_ss(expVar.maxX));
		x1 = _mm_max_ss(x1, _mm_load_ss(expVar.minX));
	}

	int r = _mm_cvtss_si32(_mm_mul_ss(x1, _mm_load_ss(expVar.a)));
	unsigned int v = r & mask(expVar.s);
	float t = _mm_cvtss_f32(x1) - r * expVar.b[0];
	int u = r >> expVar.s;
	fi fi;
	fi.i = ((u + 127) << 23) | expVar.tbl[v];
	return (1 + t) * fi.f;
#else
	x = std::min(x, expVar.maxX[0]);
	x = std::max(x, expVar.minX[0]);
	float t = x * expVar.a[0];
	const float magic = (1 << 23) + (1 << 22); // to round
	t += magic;
	fi fi;
	fi.f = t;
	t = x - (t - magic) * expVar.b[0];
	int u = ((fi.i + (127 << expVar.s)) >> expVar.s) << 23;
	unsigned int v = fi.i & mask(expVar.s);
	fi.i = u | expVar.tbl[v];
	return (1 + t) * fi.f;
//	return (1 + t) * pow(2, (float)u) * pow(2, (float)v / n);
#endif
}

inline double expd(double x)
{
	if (x <= -708.39641853226408) return 0;
	if (x >= 709.78271289338397) return std::numeric_limits<double>::infinity();
	using namespace local;
	const ExpdVar<>& c = C<>::expdVar();
#if 1
	const double _b = double(uint64_t(3) << 51);
	__m128d b = _mm_load_sd(&_b);
	__m128d xx = _mm_load_sd(&x);
	__m128d d = _mm_add_sd(_mm_mul_sd(xx, _mm_load_sd(&c.a)), b);
	uint64_t di = _mm_cvtsi128_si32(_mm_castpd_si128(d));
	uint64_t iax = c.tbl[di & mask(c.sbit)];
	__m128d _t = _mm_sub_sd(_mm_mul_sd(_mm_sub_sd(d, b), _mm_load_sd(&c.ra)), xx);
	uint64_t u = ((di + c.adj) >> c.sbit) << 52;
	double t;
	_mm_store_sd(&t, _t);
	double y = (c.C3[0] - t) * (t * t) * c.C2[0] - t + c.C1[0];
	double did;
	u |= iax;
	memcpy(&did, &u, sizeof(did));
	return y * did;
#else
/*
	remark : -ffast-math option of gcc may generate bad code for fmath::expd
*/
	const uint64_t b = 3ULL << 51;
	di di;
	di.d = x * c.a + b;
	uint64_t iax = c.tbl[di.i & mask(c.sbit)];

	double t = (di.d - b) * c.ra - x;
	uint64_t u = ((di.i + c.adj) >> c.sbit) << 52;
	double y = (c.C3[0] - t) * (t * t) * c.C2[0] - t + c.C1[0];

	di.i = u | iax;
	return y * di.d;
#endif
}

inline __m128d exp_pd(__m128d x)
{
#if 0 // faster on Haswell
	MIE_ALIGN(16) double buf[2];
	memcpy(buf, &x, sizeof(buf));
	buf[0] = expd(buf[0]);
	buf[1] = expd(buf[1]);
	__m128d y;
	memcpy(&y, buf, sizeof(buf));
	return y;
#else // faster on Skeylake
	using namespace local;
	const ExpdVar<>& c = C<>::expdVar();
	const double b = double(3ULL << 51);
	const __m128d mC1 = *cast_to<__m128d>(c.C1);
	const __m128d mC2 = *cast_to<__m128d>(c.C2);
	const __m128d mC3 = *cast_to<__m128d>(c.C3);
	const __m128d ma = _mm_set1_pd(c.a);
	const __m128d mra = _mm_set1_pd(c.ra);
	const __m128i madj = _mm_set1_epi32(c.adj);
	MIE_ALIGN(16) const double expMax[2] = { 709.78271289338397, 709.78271289338397 };
	MIE_ALIGN(16) const double expMin[2] = { -708.39641853226408, -708.39641853226408 };
	x = _mm_min_pd(x, *(const __m128d*)expMax);
	x = _mm_max_pd(x, *(const __m128d*)expMin);

	__m128d d = _mm_mul_pd(x, ma);
	d = _mm_add_pd(d, _mm_set1_pd(b));
	int adr0 = _mm_cvtsi128_si32(_mm_castpd_si128(d)) & mask(c.sbit);
	int adr1 = _mm_cvtsi128_si32(_mm_srli_si128(_mm_castpd_si128(d), 8)) & mask(c.sbit);
__m128i iaxL = _mm_castpd_si128(_mm_load_sd((const double*)&c.tbl[adr0]));
	__m128i iax = _mm_castpd_si128(_mm_load_sd((const double*)&c.tbl[adr1]));
	iax = _mm_unpacklo_epi64(iaxL, iax);

	__m128d t = _mm_sub_pd(_mm_mul_pd(_mm_sub_pd(d, _mm_set1_pd(b)), mra), x);
	__m128i u = _mm_castpd_si128(d);
	u = _mm_add_epi64(u, madj);
	u = _mm_srli_epi64(u, c.sbit);
	u = _mm_slli_epi64(u, 52);
	u = _mm_or_si128(u, iax);
	__m128d y = _mm_mul_pd(_mm_sub_pd(mC3, t), _mm_mul_pd(t, t));
	y = _mm_mul_pd(y, mC2);
	y = _mm_add_pd(_mm_sub_pd(y, t), mC1);
	y = _mm_mul_pd(y, _mm_castsi128_pd(u));
	return y;
#endif
}

/*
	px : pointer to array of double
	n : size of array(assume multiple of 2 or 4)
*/
inline void expd_v(double *px, size_t n)
{
	using namespace local;
	const ExpdVar<>& c = C<>::expdVar();
	const double b = double(3ULL << 51);
#ifdef __AVX2__
	size_t r = n & 3;
	n &= ~3;
	const __m256d mC1 = _mm256_set1_pd(c.C1[0]);
	const __m256d mC2 = _mm256_set1_pd(c.C2[0]);
	const __m256d mC3 = _mm256_set1_pd(c.C3[0]);
	const __m256d ma = _mm256_set1_pd(c.a);
	const __m256d mra = _mm256_set1_pd(c.ra);
	const __m256i madj = _mm256_set1_epi64x(c.adj);
	const __m256i maskSbit = _mm256_set1_epi64x(mask(c.sbit));
	const __m256d expMax = _mm256_set1_pd(709.78272569338397);
	const __m256d expMin = _mm256_set1_pd(-708.39641853226408);
	for (size_t i = 0; i < n; i += 4) {
		__m256d x = _mm256_loadu_pd(px);
		x = _mm256_min_pd(x, expMax);
		x = _mm256_max_pd(x, expMin);

		__m256d d = _mm256_mul_pd(x, ma);
		d = _mm256_add_pd(d, _mm256_set1_pd(b));
		__m256i adr = _mm256_and_si256(_mm256_castpd_si256(d), maskSbit);
		__m256i iax = _mm256_i64gather_epi64((const long long*)c.tbl, adr, 8);
		__m256d t = _mm256_sub_pd(_mm256_mul_pd(_mm256_sub_pd(d, _mm256_set1_pd(b)), mra), x);
		__m256i u = _mm256_castpd_si256(d);
		u = _mm256_add_epi64(u, madj);
		u = _mm256_srli_epi64(u, c.sbit);
		u = _mm256_slli_epi64(u, 52);
		u = _mm256_or_si256(u, iax);
		__m256d y = _mm256_mul_pd(_mm256_sub_pd(mC3, t), _mm256_mul_pd(t, t));
		y = _mm256_mul_pd(y, mC2);
		y = _mm256_add_pd(_mm256_sub_pd(y, t), mC1);
		_mm256_storeu_pd(px, _mm256_mul_pd(y, _mm256_castsi256_pd(u)));
		px += 4;
	}
#else
	size_t r = n & 1;
	n &= ~1;
	const __m128d mC1 = _mm_set1_pd(c.C1[0]);
	const __m128d mC2 = _mm_set1_pd(c.C2[0]);
	const __m128d mC3 = _mm_set1_pd(c.C3[0]);
	const __m128d ma = _mm_set1_pd(c.a);
	const __m128d mra = _mm_set1_pd(c.ra);
#if defined(__x86_64__) || defined(_WIN64)
	const __m128i madj = _mm_set1_epi64x(c.adj);
#else
	const __m128i madj = _mm_set_epi32(0, c.adj, 0, c.adj);
#endif
	const __m128d expMax = _mm_set1_pd(709.78272569338397);
	const __m128d expMin = _mm_set1_pd(-708.39641853226408);
	for (size_t i = 0; i < n; i += 2) {
		__m128d x = _mm_loadu_pd(px);
		x = _mm_min_pd(x, expMax);
		x = _mm_max_pd(x, expMin);

		__m128d d = _mm_mul_pd(x, ma);
		d = _mm_add_pd(d, _mm_set1_pd(b));
		int adr0 = _mm_cvtsi128_si32(_mm_castpd_si128(d)) & mask(c.sbit);
		int adr1 = _mm_cvtsi128_si32(_mm_srli_si128(_mm_castpd_si128(d), 8)) & mask(c.sbit);

		__m128i iaxL = _mm_castpd_si128(_mm_load_sd((const double*)&c.tbl[adr0]));
		__m128i iax = _mm_castpd_si128(_mm_load_sd((const double*)&c.tbl[adr1]));
		iax = _mm_unpacklo_epi64(iaxL, iax);

		__m128d t = _mm_sub_pd(_mm_mul_pd(_mm_sub_pd(d, _mm_set1_pd(b)), mra), x);
		__m128i u = _mm_castpd_si128(d);
		u = _mm_add_epi64(u, madj);
		u = _mm_srli_epi64(u, c.sbit);
		u = _mm_slli_epi64(u, 52);
		u = _mm_or_si128(u, iax);
		__m128d y = _mm_mul_pd(_mm_sub_pd(mC3, t), _mm_mul_pd(t, t));
		y = _mm_mul_pd(y, mC2);
		y = _mm_add_pd(_mm_sub_pd(y, t), mC1);
		_mm_storeu_pd(px, _mm_mul_pd(y, _mm_castsi128_pd(u)));
		px += 2;
	}
#endif
	for (size_t i = 0; i < r; i++) {
		px[i] = expd(px[i]);
	}
}

#ifdef FMATH_USE_XBYAK
inline __m128 exp_psC(__m128 x)
#else
inline __m128 exp_ps(__m128 x)
#endif
{
	using namespace local;
	const ExpVar<>& expVar = C<>::expVar();

	__m128i limit = _mm_castps_si128(_mm_and_ps(x, *cast_to<__m128>(expVar.i7fffffff)));
	int over = _mm_movemask_epi8(_mm_cmpgt_epi32(limit, *cast_to<__m128i>(expVar.maxX)));
	if (over) {
		x = _mm_min_ps(x, _mm_load_ps(expVar.maxX));
		x = _mm_max_ps(x, _mm_load_ps(expVar.minX));
	}

	__m128i r = _mm_cvtps_epi32(_mm_mul_ps(x, *cast_to<__m128>(expVar.a)));
	__m128 t = _mm_sub_ps(x, _mm_mul_ps(_mm_cvtepi32_ps(r), *cast_to<__m128>(expVar.b)));
	t = _mm_add_ps(t, *cast_to<__m128>(expVar.f1));

	__m128i v4 = _mm_and_si128(r, *cast_to<__m128i>(expVar.mask_s));
	__m128i u4 = _mm_add_epi32(r, *cast_to<__m128i>(expVar.i127s));
	u4 = _mm_srli_epi32(u4, expVar.s);
	u4 = _mm_slli_epi32(u4, 23);

#ifdef __AVX2__ // fast?
	__m128i ti = _mm_i32gather_epi32((const int*)expVar.tbl, v4, 4);
	__m128 t0 = _mm_castsi128_ps(ti);
#else
	unsigned int v0, v1, v2, v3;
	v0 = _mm_cvtsi128_si32(v4);
	v1 = _mm_extract_epi16(v4, 2);
	v2 = _mm_extract_epi16(v4, 4);
	v3 = _mm_extract_epi16(v4, 6);
#if 1
	__m128 t0, t1, t2, t3;

	t0 = _mm_castsi128_ps(_mm_set1_epi32(expVar.tbl[v0]));
	t1 = _mm_castsi128_ps(_mm_set1_epi32(expVar.tbl[v1]));
	t2 = _mm_castsi128_ps(_mm_set1_epi32(expVar.tbl[v2]));
	t3 = _mm_castsi128_ps(_mm_set1_epi32(expVar.tbl[v3]));

	t1 = _mm_movelh_ps(t1, t3);
	t1 = _mm_castsi128_ps(_mm_slli_epi64(_mm_castps_si128(t1), 32));
	t0 = _mm_movelh_ps(t0, t2);
	t0 = _mm_castsi128_ps(_mm_srli_epi64(_mm_castps_si128(t0), 32));
	t0 = _mm_or_ps(t0, t1);
#else
	__m128i ti = _mm_castps_si128(_mm_load_ss((const float*)&expVar.tbl[v0]));
	ti = _mm_insert_epi32(ti, expVar.tbl[v1], 1);
	ti = _mm_insert_epi32(ti, expVar.tbl[v2], 2);
	ti = _mm_insert_epi32(ti, expVar.tbl[v3], 3);
	__m128 t0 = _mm_castsi128_ps(ti);
#endif
#endif
	t0 = _mm_or_ps(t0, _mm_castsi128_ps(u4));

	t = _mm_mul_ps(t, t0);

	return t;
}
#ifdef __AVX2__
inline __m256 exp_ps256(__m256 x)
{
	using namespace local;
	const ExpVar<>& expVar = C<>::expVar();

	__m256i limit = _mm256_castps_si256(_mm256_and_ps(x, *reinterpret_cast<const __m256*>(expVar.i7fffffff)));
	int over = _mm256_movemask_epi8(_mm256_cmpgt_epi32(limit, *reinterpret_cast<const __m256i*>(expVar.maxX)));
	if (over) {
		x = _mm256_min_ps(x, _mm256_load_ps(expVar.maxX));
		x = _mm256_max_ps(x, _mm256_load_ps(expVar.minX));
	}
	__m256i r = _mm256_cvtps_epi32(_mm256_mul_ps(x, *reinterpret_cast<const __m256*>(expVar.a)));
	__m256 t = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_cvtepi32_ps(r), *reinterpret_cast<const __m256*>(expVar.b)));
	t = _mm256_add_ps(t, *reinterpret_cast<const __m256*>(expVar.f1));
	__m256i v8 = _mm256_and_si256(r, *reinterpret_cast<const __m256i*>(expVar.mask_s));
	__m256i u8 = _mm256_add_epi32(r, *reinterpret_cast<const __m256i*>(expVar.i127s));
	u8 = _mm256_srli_epi32(u8, expVar.s);
	u8 = _mm256_slli_epi32(u8, 23);
#if 1
	__m256i ti = _mm256_i32gather_epi32((const int*)expVar.tbl, v8, 4);
#else
	unsigned int v0, v1, v2, v3, v4, v5, v6, v7;
	v0 = _mm256_extract_epi16(v8, 0);
	v1 = _mm256_extract_epi16(v8, 2);
	v2 = _mm256_extract_epi16(v8, 4);
	v3 = _mm256_extract_epi16(v8, 6);
	v4 = _mm256_extract_epi16(v8, 8);
	v5 = _mm256_extract_epi16(v8, 10);
	v6 = _mm256_extract_epi16(v8, 12);
	v7 = _mm256_extract_epi16(v8, 14);
	__m256i ti = _mm256_setzero_si256();
	ti = _mm256_insert_epi32(ti, expVar.tbl[v0], 0);
	ti = _mm256_insert_epi32(ti, expVar.tbl[v1], 1);
	ti = _mm256_insert_epi32(ti, expVar.tbl[v2], 2);
	ti = _mm256_insert_epi32(ti, expVar.tbl[v3], 3);
	ti = _mm256_insert_epi32(ti, expVar.tbl[v4], 4);
	ti = _mm256_insert_epi32(ti, expVar.tbl[v5], 5);
	ti = _mm256_insert_epi32(ti, expVar.tbl[v6], 6);
	ti = _mm256_insert_epi32(ti, expVar.tbl[v7], 7);
#endif
	__m256 t0 = _mm256_castsi256_ps(ti);
	t0 = _mm256_or_ps(t0, _mm256_castsi256_ps(u8));
	t = _mm256_mul_ps(t, t0);
	return t;
}
#endif

inline float log(float x)
{
	using namespace local;
	const LogVar<>& logVar = C<>::logVar();
	const size_t logLen = logVar.LEN;

	fi fi;
	fi.f = x;
	int a = fi.i & (mask(8) << 23);
	unsigned int b1 = fi.i & (mask(logLen) << (23 - logLen));
	unsigned int b2 = fi.i & mask(23 - logLen);
	int idx = b1 >> (23 - logLen);
	float f = float(a - (127 << 23)) * logVar.c_log2 + logVar.tbl[idx].app + float(b2) * logVar.tbl[idx].rev;
	return f;
}

inline __m128 log_ps(__m128 x)
{
	using namespace local;
	const LogVar<>& logVar = C<>::logVar();

	__m128i xi = _mm_castps_si128(x);
	__m128i idx = _mm_srli_epi32(_mm_and_si128(xi, *cast_to<__m128i>(logVar.m2)), (23 - logVar.LEN));
	__m128 a  = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_and_si128(xi, *cast_to<__m128i>(logVar.m1)), *cast_to<__m128i>(logVar.m5)));
	__m128 b2 = _mm_cvtepi32_ps(_mm_and_si128(xi, *cast_to<__m128i>(logVar.m3)));

	a = _mm_mul_ps(a, *cast_to<__m128>(logVar.m4)); // c_log2

	unsigned int i0 = _mm_cvtsi128_si32(idx);

#if 1
	unsigned int i1 = _mm_extract_epi16(idx, 2);
	unsigned int i2 = _mm_extract_epi16(idx, 4);
	unsigned int i3 = _mm_extract_epi16(idx, 6);
#else
	idx = _mm_srli_si128(idx, 4);
	unsigned int i1 = _mm_cvtsi128_si32(idx);

	idx = _mm_srli_si128(idx, 4);
	unsigned int i2 = _mm_cvtsi128_si32(idx);

	idx = _mm_srli_si128(idx, 4);
	unsigned int i3 = _mm_cvtsi128_si32(idx);
#endif

	__m128 app, rev;
	__m128i L = _mm_loadl_epi64(cast_to<__m128i>(&logVar.tbl[i0].app));
	__m128i H = _mm_loadl_epi64(cast_to<__m128i>(&logVar.tbl[i1].app));
	__m128 t = _mm_castsi128_ps(_mm_unpacklo_epi64(L, H));
	L = _mm_loadl_epi64(cast_to<__m128i>(&logVar.tbl[i2].app));
	H = _mm_loadl_epi64(cast_to<__m128i>(&logVar.tbl[i3].app));
	rev = _mm_castsi128_ps(_mm_unpacklo_epi64(L, H));
	app = _mm_shuffle_ps(t, rev, MIE_PACK(2, 0, 2, 0));
	rev = _mm_shuffle_ps(t, rev, MIE_PACK(3, 1, 3, 1));

	a = _mm_add_ps(a, app);
	rev = _mm_mul_ps(b2, rev);
	return _mm_add_ps(a, rev);
}

#ifndef __CYGWIN__
// cygwin defines log2() in global namespace!
// log2(x) = log(x) / log(2)
inline float log2(float x) { return fmath::log(x) * 1.442695f; }
#endif

/*
	for given y > 0
	get f_y(x) := pow(x, y) for x >= 0
*/
class PowGenerator {
	enum {
		N = 11
	};
	float tbl0_[256];
	struct {
		float app;
		float rev;
	} tbl1_[1 << N];
public:
	PowGenerator(float y)
	{
		for (int i = 0; i < 256; i++) {
			tbl0_[i] = ::powf(2, (i - 127) * y);
		}
		const double e = 1 / double(1 << 24);
		const double h = 1 / double(1 << N);
		const size_t n = 1U << N;
		for (size_t i = 0; i < n; i++) {
			double x = 1 + double(i) / n;
			double a = ::pow(x, (double)y);
			tbl1_[i].app = (float)a;
			double b = ::pow(x + h - e, (double)y);
			tbl1_[i].rev = (float)((b - a) / (h - e) / (1 << 23));
		}
	}
	float get(float x) const
	{
		using namespace local;
		fi fi;
		fi.f = x;
		int a = (fi.i >> 23) & mask(8);
		unsigned int b = fi.i & mask(23);
		unsigned int b1 = b & (mask(N) << (23 - N));
		unsigned int b2 = b & mask(23 - N);
		float f;
		int idx = b1 >> (23 - N);
		f = tbl0_[a] * (tbl1_[idx].app + float(b2) * tbl1_[idx].rev);
		return f;
	}
};

// for Xbyak version
#ifdef FMATH_USE_XBYAK
float exp(float x) {
	static float (*const jitExp)(float) = local::C<>::getInstance().exp_;
	return jitExp(x);
}
__m128 exp(__m128 x) {
	static __m128 (*const jitExp)(__m128) = local::C<>::getInstance().exp_ps_;
	return jitExp(x);
}
#if 0
float (*const exp)(float) = local::C<>::getInstance().exp_;
__m128 (*const exp_ps)(__m128) = local::C<>::getInstance().exp_ps_;
#endif
#endif

// exp2(x) = pow(2, x)
inline float exp2(float x) { return fmath::exp(x * 0.6931472f); }

/*
	this function may be optimized in the future
*/
inline __m128d log_pd(__m128d x)
{
	double d[2];
	memcpy(d, &x, sizeof(d));
	d[0] = ::log(d[0]);
	d[1] = ::log(d[1]);
	__m128d m;
	memcpy(&m, d, sizeof(m));
	return m;
}
inline __m128 pow_ps(__m128 x, __m128 y)
{
	return exp_ps(_mm_mul_ps(y, log_ps(x)));
}
inline __m128d pow_pd(__m128d x, __m128d y)
{
	return exp_pd(_mm_mul_pd(y, log_pd(x)));
}

} // fmath

} // anonymous
