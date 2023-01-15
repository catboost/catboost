
fast approximate function of float exp(float) and float log(float)

-----------------------------------------------------------------------------

<How to use>
Include fmath.hpp and use fmath::log() and fmath::exp().

fmath::PowGenerator is a class to generate a function to compute pow(x, y)
of x >= 0 for a given fixed y > 0.

eg.
fmath::PowGenerator f(1.234);
f.get(x) returns pow(x, 1.234);

<Prototype>
-----------------------------------------------------------------------------
float fmath::exp(float);
float fmath::log(float);

__m128 fmath::exp_ps(__m128);
__m128 fmath::log_ps(__m128);

-----------------------------------------------------------------------------
<Experimental>

If you install xbyak(https://github.com/herumi/xbyak/)
and define FMATH_USE_XBYAK before including fmath.hpp,
then fmath::exp() and fmath::exp_ps() will be about 10~20 % faster.
Xbyak version uses SSE4.1 if available.

# AVX version of fmath::exp is experimental

<Benchmark>
-----------------------------------------------------------------------------

compiler Visual Studio 2010RC / icc 11.1 / gcc 4.3.2 on cygwin / gcc 4.4.1 on 64bit Linux

option

cl(icl):
    /Ox /Ob2 /GS- /Zi /D_SECURE_SCL=0 /MD /Oy /arch:SSE2 /fp:fast /DNOMINMAX

gcc:
    -O3 -fomit-frame-pointer -DNDEBUG -fno-operator-names -msse2 -mfpmath=sse -ffast-math -march=core2

unit ; <clocks without dummy loop> / <clocks with loop>


                  Core i7-2600 3.4GHz

                Windows7 SP1               ubuntu 10.10
             VC10(32bit)  VC10(64bit)      gcc4.5(64bit)

    std::exp  91.8/101.0   14.1/ 23.5       639.8/650.8
  fmath::exp   7.6/ 16.9    1.3/ 10.7         3.1/ 14.1

  std::expx4 340.5/351.6   86.9/ 95.5      2608.2/2617.2
fmath::expx4  61.0/ 72.1   42.6/ 51.3        46.5/ 55.5
fmath::exp_ps 22.9/ 34.1   19.9/ 28.6        58.8/ 67.9

     std log  57.2/ 66.3   15.9/ 24.1        41.4/ 49.6
   fmath log   9.5/ 18.6    4.4/ 12.6         4.8/ 13.0

   std logx4 230.8/241.9   94.7/103.5       204.5/213.5
 fmath logx4  34.1/ 45.2   28.6/ 37.4        27.8/ 36.8
fmath log_ps  21.1/ 32.2   15.3/ 24.1        56.9/ 65.9


           Xeon X5650 2.67GHz

   ubuntu 10.04
   gcc4.4.3-4(64bit)

    std::exp  559.9/568.7
  fmath::exp    5.1/ 13.8

  std::expx4 2257.1/2266.0
fmath::expx4   44.5/  53.3
fmath::exp_ps  59.8/  68.7

     std log   45.4/ 54.1
   fmath log    4.2/ 12.9

   std logx4  212.1/220.8
 fmath logx4   27.2/ 35.9
fmath log_ps   58.0/ 66.7

                            Core i7 2.8GHz

                        VC2010                       icc11.1
                 Xp(32bit)    Xp(64bit)     Xp(32bit)     Xp(64bit)
std::exp        88.2/ 97.2   14.6/ 22.4    19.1/ 28.8    13.7/ 21.7
fmath::exp      10.2/ 19.2    3.6/ 11.4     8.2/ 17.9     3.5/ 11.5

std::exp x 4   342.0/357.5   94.8/102.8    79.1/ 91.9    94.3/102.6
fmath::exp_ps   25.7/ 41.2   25.2/ 33.3    31.8/ 44.6    25.6/ 34.0
icl::exp_ps                                34.8/ 47.6    31.6/ 39.9

std::log        57.9/ 67.1   18.8/ 27.0    41.4/ 51.8    19.6/ 28.2
fmath::log       8.5/ 17.7    3.4/ 11.6    13.3/ 23.7     1.2/  9.8

std::log x 4   241.2/255.8  114.2/122.5   113.0/127.0   102.0/110.6
fmath::log_ps   23.9/ 38.4   20.5/ 28.8    26.2/ 40.2    23.9/ 32.5
icl::log_ps                                34.2/ 48.2    34.8/ 43.4


                             Core2Duo 2.6GHz

                        VC2010                       icc11.1          gcc 4.4.3    gcc 4.3.2 on cygwin
                 Xp(32bit)    Xp(64bit)     Xp(32bit)    Xp(64bit)  Linux(64bit)       Xp(32bit)
std::exp       139.9/150.1   24.5/ 33.0    27.4/ 38.4   18.0/ 27.1    586.0/591.5     157.8/167.9
fmath::exp      10.1/ 20.3    5.6/ 14.1    10.1/ 21.1    5.8/ 14.9      9.1/ 14.6      10.8/ 20.9

std::exp x 4   572.9/585.5  122.5/133.3   107.7/124.2  100.7/111.4   2583.6/2608.5    658.7/694.6
fmath::exp_ps   41.7/ 54.3   35.9/ 46.7    48.4/ 64.9   38.8/ 49.4     49.0/ 73.9      52.8/ 88.7
icl::exp_ps                                45.1/ 61.6   47.8/ 58.5

std::log        66.3/ 77.0   22.8/ 31.8    42.8/ 53.9   18.8/ 27.8     82.8/ 91.7     114.7/124.8
fmath::log      10.3/ 21.1    4.2/ 13.3    12.3/ 23.4    2.6/ 11.7      5.0/ 13.9      13.1/ 23.2

std::log x 4   273.3/286.1  125.2/136.1   123.4/139.1  104.7/115.1    329.2/356.3     473.2/509.2
fmath::log_ps   38.4/ 51.2   29.3/ 40.1    36.3/ 52.0   31.8/ 42.3     28.8/ 55.8      50.5/ 86.4
icl::log_ps                                58.7/ 74.4   56.3/ 66.7


           Quad-Core AMD Opteron 2376

                 gcc 4.4.1      gcc.4.41
               Linux(32bit)   Linux(64bit)
std::exp        112.9/128.9    528.7/542.2
fmath::exp       23.7/ 39.6     17.1/ 30.7

std::exp x 4    540.7/562.4   2127.3/2159.7
fmath::exp_ps   108.6/130.3     71.3/103.7

std::log        182.4/198.7    110.7/124.0
fmath::log       23.0/ 39.3     11.6/ 24.9

std::log x 4    827.1/848.6    464.8/497.2
fmath::log_ps   102.0/123.5     76.5/108.9


-----------------------------------------------------------------------------
<Remark>
gcc puts warnings such as "dereferencing type-punned pointer will break strict-aliasing rules."
It is no problem.
Please change #if 1 in fmath.hpp:423 if you worry about it. But it causes a little slower.

-----------------------------------------------------------------------------
<License>

modified new BSD License
http://www.opensource.org/licenses/bsd-license.php

-----------------------------------------------------------------------------
<History>
2011/Mar/25 exp supports AVX
2011/Mar/25 exp, exp_ps support avx
2010/Feb/16 add fmath::exp_ps, log_ps and optimize functions
2010/Jan/10 add fmath::PowGenerator
2009/Dec/28 add fmath::log()
2009/Dec/09 support cygwin
2009/Dec/08 first version

-----------------------------------------------------------------------------
<Author>

http://herumi.in.coocan.jp/
MITSUNARI Shigeo(herumi@nifty.com)
