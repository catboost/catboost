FarmHash, a family of hash functions.
Version 1.1

Introduction
============

A general overview of hash functions and their use is available in the file
Understanding_Hash_Functions in this directory.  It may be helpful to read it
before using FarmHash.

FarmHash provides hash functions for strings and other data.  The functions
mix the input bits thoroughly but are not suitable for cryptography.  See
"Hash Quality," below, for details on how FarmHash was tested and so on.

We provide reference implementations in C++, with a friendly MIT license.

All members of the FarmHash family were designed with heavy reliance on
previous work by Jyrki Alakuijala, Austin Appleby, Bob Jenkins, and others.


Recommended Usage
=================

Our belief is that the typical hash function is mostly used for in-memory hash
tables and similar.  That use case allows hash functions that differ on
different platforms, and that change from time to time.  For this, I recommend
using wrapper functions in a .h file with comments such as, "may change from
time to time, may differ on different platforms, and may change depending on
NDEBUG."

Some projects may also require a forever-fixed, portable hash function.  Again
we recommend using wrapper functions in a .h, but in this case the comments on
them would be very different.

We have provided a sample of these wrapper functions in src/farmhash.h.  Our
hope is that most people will need nothing more than src/farmhash.h and
src/farmhash.cc.  Those two files are a usable and relatively portable library.
(One portability snag: if your compiler doesn't have __builtin_expect then
you may need to define FARMHASH_NO_BUILTIN_EXPECT.)  For those that prefer
using a configure script (perhaps because they want to "make install" later),
FarmHash has one, but for many people it's best to ignore it.

Note that the wrapper functions such as Hash() in src/farmhash.h can select
one of several hash functions.  The selection is done at compile time, based
on your machine architecture (e.g., sizeof(size_t)) and the availability of
vector instructions (e.g., SSE4.1).

To get the best performance from FarmHash, one will need to think a bit about
when to use compiler flags that allow vector instructions and such: -maes,
-msse4.2, -mavx, etc., or their equivalents for other compilers.  Those are
the g++ flags that make g++ emit more types of machine instructions than it
otherwise would.  For example, if you are confident that you will only be
using FarmHash on systems with SSE4.2 and/or AES, you may communicate that to
the compiler as explained in src/farmhash.cc.  If not, use -maes, -mavx, etc.,
when you can, and the appropriate choices will be made by via conditional
compilation in src/farmhash.cc.

It may be beneficial to try -O3 or other compiler flags as well.  I also have
found feedback-directed optimization (FDO) to improve the speed of FarmHash.

The "configure" script: creating config.h
=========================================

We provide reference implementations of several FarmHash functions, written in
C++.  The build system is based on autoconf.  It defaults the C++ compiler
flags to "-g -O2", which may or may not be best.

If you are planning to use the configure script, I generally recommend
trying this first, unless you know that your system lacks AVX and/or AESNI:

  ./configure CXXFLAGS="-g -mavx -maes -O3"
  make all check

If that fails, you can retry with -mavx and/or -maes removed, or with -mavx replaced by
-msse4.1 or -msse4.2.

Please see below for thoughts on cross-platform testing, if that is a concern.
Finally, if you want to install a library, you may use

  make install

Some useful flags for configure include:

  --enable-optional-builtin-expect: This causes __builtin_expect to be optional.
    If you don't use this flag, the assumption is that FarmHash will be compiled
    with compilers that provide __builtin_expect.  In practice, some FarmHash
    variants may be slightly faster if __builtin_expect is available, but it
    isn't very important and affects speed only.

Further Details
===============

The above instructions will produce a single source-level library that
includes multiple hash functions.  It will use conditional compilation, and
perhaps GCC's multiversioning, to select among the functions.  In addition,
"make all check" will create an object file using your chosen compiler, and
test it.  The object file won't necessarily contain all the code that would be
used if you were to compile the code on other platforms.  The downside of this
is obvious: the paths not tested may not actually work if and when you try
them.  The FarmHash developers try hard to prevent such problems; please let
us know if you find bugs.

To aid your cross-platform testing, for each relevant platform you may
compile your program that uses farmhash.cc with the preprocessor flag
FARMHASHSELFTEST equal to 1.  This causes a FarmHash self test to run
at program startup; the self test writes output to stdout and then
calls std::exit().  You can see this in action by running "make check":
see src/farm-test.cc for details.

There's also a trivial workaround to force particular functions to be used:
modify the wrapper functions in hash.h.  You can prevent choices being made via
conditional compilation or multiversioning by choosing FarmHash variants with
names like farmhashaa::Hash32, farmhashab::Hash64, etc.: those compute the same
hash function regardless of conditional compilation, multiversioning, or
endianness.  Consult their comments and ifdefs to learn their requirements: for
example, they are not all guaranteed to work on all platforms.

Known Issues
============

1) FarmHash was developed with little-endian architectures in mind.  It should
work on big-endian too, but less work has gone into optimizing for those
platforms.  To make FarmHash work properly on big-endian platforms you may
need to modify the wrapper .h file and/or your compiler flags to arrange for
FARMHASH_BIG_ENDIAN to be defined, though there is logic that tries to figure
it out automatically.

2) FarmHash's implementation is fairly complex.

3) The techniques described in dev/INSTRUCTIONS to let hash function
developers regenerate src/*.cc from dev/* are hacky and not so portable.

Hash Quality
============

We like to test hash functions with SMHasher, among other things.
SMHasher isn't perfect, but it seems to find almost any significant flaw.
SMHasher is available at http://code.google.com/p/smhasher/

SMHasher is designed to pass a 32-bit seed to the hash functions it tests.
For our functions that accept a seed, we use the given seed directly (padded
with zeroes as needed); for our functions that don't accept a seed, we hash
the concatenation of the given seed and the input string.

Some minor flaws in 32-bit and 64-bit functions are harmless, as we
expect the primary use of these functions will be in hash tables.  We
may have gone slightly overboard in trying to please SMHasher and other
similar tests, but we don't want anyone to choose a different hash function
because of some minor issue reported by a quality test.

If your setup is similar enough to mine, it's easy to use SMHasher and other
tools yourself via the "builder" in the dev directory.  See dev/INSTRUCTIONS.
(Improvements to that directory are a relatively low priority, and code
there is never going to be as portable as the other parts of FarmHash.)

For more information
====================

http://code.google.com/p/farmhash/

farmhash-discuss@googlegroups.com

Please feel free to send us comments, questions, bug reports, or patches.
