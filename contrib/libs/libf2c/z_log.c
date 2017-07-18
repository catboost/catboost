#include "f2c.h"

#ifdef KR_headers
double log(), f__cabs(), atan2();
#define ANSI(x) ()
#else
#define ANSI(x) x
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
extern double f__cabs(double, double);
#endif

#ifndef NO_DOUBLE_EXTENDED
#ifndef GCC_COMPARE_BUG_FIXED
#ifndef Pre20000310
#ifdef Comment
Some versions of gcc, such as 2.95.3 and 3.0.4, are buggy under -O2 or -O3:
on IA32 (Intel 80x87) systems, they may do comparisons on values computed
in extended-precision registers.  This can lead to the test "s > s0" that
was used below being carried out incorrectly.  The fix below cannot be
spoiled by overzealous optimization, since the compiler cannot know
whether gcc_bug_bypass_diff_F2C will be nonzero.  (We expect it always
to be zero.  The weird name is unlikely to collide with anything.)

An example (provided by Ulrich Jakobus) where the bug fix matters is

	double complex a, b
	a = (.1099557428756427618354862829619, .9857360542953131909982289471372)
	b = log(a)

An alternative to the fix below would be to use 53-bit rounding precision,
but the means of specifying this 80x87 feature are highly unportable.
#endif /*Comment*/
#define BYPASS_GCC_COMPARE_BUG
double (*gcc_bug_bypass_diff_F2C) ANSI((double*,double*));
 static double
#ifdef KR_headers
diff1(a,b) double *a, *b;
#else
diff1(double *a, double *b)
#endif
{ return *a - *b; }
#endif /*Pre20000310*/
#endif /*GCC_COMPARE_BUG_FIXED*/
#endif /*NO_DOUBLE_EXTENDED*/

#ifdef KR_headers
VOID z_log(r, z) doublecomplex *r, *z;
#else
void z_log(doublecomplex *r, doublecomplex *z)
#endif
{
	double s, s0, t, t2, u, v;
	double zi = z->i, zr = z->r;
#ifdef BYPASS_GCC_COMPARE_BUG
	double (*diff) ANSI((double*,double*));
#endif

	r->i = atan2(zi, zr);
#ifdef Pre20000310
	r->r = log( f__cabs( zr, zi ) );
#else
	if (zi < 0)
		zi = -zi;
	if (zr < 0)
		zr = -zr;
	if (zr < zi) {
		t = zi;
		zi = zr;
		zr = t;
		}
	t = zi/zr;
	s = zr * sqrt(1 + t*t);
	/* now s = f__cabs(zi,zr), and zr = |zr| >= |zi| = zi */
	if ((t = s - 1) < 0)
		t = -t;
	if (t > .01)
		r->r = log(s);
	else {

#ifdef Comment

	log(1+x) = x - x^2/2 + x^3/3 - x^4/4 + - ...

		 = x(1 - x/2 + x^2/3 -+...)

	[sqrt(y^2 + z^2) - 1] * [sqrt(y^2 + z^2) + 1] = y^2 + z^2 - 1, so

	sqrt(y^2 + z^2) - 1 = (y^2 + z^2 - 1) / [sqrt(y^2 + z^2) + 1]

#endif /*Comment*/

#ifdef BYPASS_GCC_COMPARE_BUG
		if (!(diff = gcc_bug_bypass_diff_F2C))
			diff = diff1;
#endif
		t = ((zr*zr - 1.) + zi*zi) / (s + 1);
		t2 = t*t;
		s = 1. - 0.5*t;
		u = v = 1;
		do {
			s0 = s;
			u *= t2;
			v += 2;
			s += u/v - t*u/(v+1);
			}
#ifdef BYPASS_GCC_COMPARE_BUG
			while(s - s0 > 1e-18 || (*diff)(&s,&s0) > 0.);
#else
			while(s > s0);
#endif
		r->r = s*t;
		}
#endif
	}
#ifdef __cplusplus
}
#endif
