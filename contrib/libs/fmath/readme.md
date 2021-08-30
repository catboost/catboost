
# fast approximate function of exponential function exp and log

## How to use

include fmath.hpp and use fmath::log, fmath::exp, fmath::expd.

fmath::PowGenerator is a class to generate a function to compute pow(x, y)
of x >= 0 for a given fixed y > 0.

eg.
fmath::PowGenerator f(1.234);
f.get(x) returns pow(x, 1.234);

## Prototype of function

* float fmath::exp(float);
* float fmath::log(float);
* double fmath::logd(double);
*
* __m128 fmath::exp_ps(__m128);
* __m128 fmath::log_ps(__m128);
* void fmath::expv_d(double *p, size_t n); // for double p[n];


### for AVX-512

`fmath2.hpp` provides the following functions:

- void expf_v(float *dst, const float *src, size_t n);
- void logf_v(float *dst, const float *src, size_t n);

## Experimental

If you install [xbyak](https://github.com/herumi/xbyak/)
and define FMATH_USE_XBYAK before including fmath.hpp,
then fmath::exp() and fmath::exp_ps() will be about 10~20 % faster.
Xbyak version uses SSE4.1 if available.

### AVX version of fmath::exp is experimental

## Remark

gcc puts warnings such as "dereferencing type-punned pointer will break strict-aliasing rules."
It is no problem.
Please change #if 1 in fmath.hpp:423 if you worry about it. But it causes a little slower.

-ffast-math option of gcc may generate bad code for fmath::expd.

# License

- modified new BSD License
- http://opensource.org/licenses/BSD-3-Clause

# History

* 2020/Jul/10 add expf_v and logf_v for AVX-512
* 2012/Oct/30 fix fmath::expd for small value
* 2011/Aug/26 add fmath::expd_v
* 2011/Mar/25 exp supports AVX
* 2011/Mar/25 exp, exp_ps support avx
* 2010/Feb/16 add fmath::exp_ps, log_ps and optimize functions
* 2010/Jan/10 add fmath::PowGenerator
* 2009/Dec/28 add fmath::log()
* 2009/Dec/09 support cygwin
* 2009/Dec/08 first version

Author
-----------

MITSUNARI Shigeo(herumi@nifty.com)
http://herumi.in.coocan.jp/


Benchmark
-----------
### compiler
* Visual Studio 2010RC
* icc 11.1
* gcc 4.3.2 on cygwin
* gcc 4.4.1 on 64bit Linux

### option

* cl(icl):
>    /Ox /Ob2 /GS- /Zi /D_SECURE_SCL=0 /MD /Oy /arch:SSE2 /fp:fast /DNOMINMAX

* gcc:
> -O3 -fomit-frame-pointer -DNDEBUG -fno-operator-names -msse2 -mfpmath=sse -march=native

see fastexp.cpp
