//
// This header includes all C++ headers required for generated Octave wrapper code.
// Using a single header file allows pre-compilation of Octave headers, as follows:
// * Check out this header file:
//     swig -octave -co octheaders.hpp
// * Pre-compile header file into octheaders.hpp.gch:
//     g++ -c ... octheaders.hpp
// * Use pre-compiled header file:
//     g++ -c -include octheaders.hpp ...
//

#if !defined(_SWIG_OCTAVE_OCTHEADERS_HPP)
#define _SWIG_OCTAVE_OCTHEADERS_HPP

// Required C++ headers
#include <cstdlib>
#include <climits>
#include <iostream>
#include <exception>
#include <functional>
#include <complex>
#include <string>
#include <vector>
#include <map>

// Minimal headers to define Octave version
#include <octave/oct.h>
#include <octave/version.h>

// Macro for enabling features which require Octave version >= major.minor.patch
// - Use (OCTAVE_PATCH_VERSION + 0) to handle both '<digit>' (released) and '<digit>+' (in development) patch numbers
#define SWIG_OCTAVE_PREREQ(major, minor, patch) \
  ( (OCTAVE_MAJOR_VERSION<<16) + (OCTAVE_MINOR_VERSION<<8) + (OCTAVE_PATCH_VERSION + 0) >= ((major)<<16) + ((minor)<<8) + (patch) )

// Reconstruct Octave major, minor, and patch versions for releases prior to 3.8.1
#if !defined(OCTAVE_MAJOR_VERSION)

# if !defined(OCTAVE_API_VERSION_NUMBER)

// Hack to distinguish between Octave 3.8.0, which removed OCTAVE_API_VERSION_NUMBER but did not yet
// introduce OCTAVE_MAJOR_VERSION, and Octave <= 3.2, which did not define OCTAVE_API_VERSION_NUMBER
#  include <octave/ov.h>
#  if defined(octave_ov_h)
#   define OCTAVE_MAJOR_VERSION 3
#   define OCTAVE_MINOR_VERSION 8
#   define OCTAVE_PATCH_VERSION 0
#  else

// Hack to distinguish between Octave 3.2 and earlier versions, before OCTAVE_API_VERSION_NUMBER existed
#   define ComplexLU __ignore
#   include <octave/CmplxLU.h>
#   undef ComplexLU
#   if defined(octave_Complex_LU_h)

// We know only that this version is prior to Octave 3.2, i.e. OCTAVE_API_VERSION_NUMBER < 37
#    define OCTAVE_MAJOR_VERSION 3
#    define OCTAVE_MINOR_VERSION 1
#    define OCTAVE_PATCH_VERSION 99

#   else

// OCTAVE_API_VERSION_NUMBER == 37
#    define OCTAVE_MAJOR_VERSION 3
#    define OCTAVE_MINOR_VERSION 2
#    define OCTAVE_PATCH_VERSION 0

#   endif // defined(octave_Complex_LU_h)

#  endif // defined(octave_ov_h)

// Correlation between Octave API and version numbers extracted from Octave's
// ChangeLogs; version is the *earliest* released Octave with that API number
# elif OCTAVE_API_VERSION_NUMBER >= 48
#  define OCTAVE_MAJOR_VERSION 3
#  define OCTAVE_MINOR_VERSION 6
#  define OCTAVE_PATCH_VERSION 0

# elif OCTAVE_API_VERSION_NUMBER >= 45
#  define OCTAVE_MAJOR_VERSION 3
#  define OCTAVE_MINOR_VERSION 4
#  define OCTAVE_PATCH_VERSION 1

# elif OCTAVE_API_VERSION_NUMBER >= 42
#  define OCTAVE_MAJOR_VERSION 3
#  define OCTAVE_MINOR_VERSION 3
#  define OCTAVE_PATCH_VERSION 54

# elif OCTAVE_API_VERSION_NUMBER >= 41
#  define OCTAVE_MAJOR_VERSION 3
#  define OCTAVE_MINOR_VERSION 3
#  define OCTAVE_PATCH_VERSION 53

# elif OCTAVE_API_VERSION_NUMBER >= 40
#  define OCTAVE_MAJOR_VERSION 3
#  define OCTAVE_MINOR_VERSION 3
#  define OCTAVE_PATCH_VERSION 52

# elif OCTAVE_API_VERSION_NUMBER >= 39
#  define OCTAVE_MAJOR_VERSION 3
#  define OCTAVE_MINOR_VERSION 3
#  define OCTAVE_PATCH_VERSION 51

# else // OCTAVE_API_VERSION_NUMBER == 38
#  define OCTAVE_MAJOR_VERSION 3
#  define OCTAVE_MINOR_VERSION 3
#  define OCTAVE_PATCH_VERSION 50

# endif // !defined(OCTAVE_API_VERSION_NUMBER)

#endif // !defined(OCTAVE_MAJOR_VERSION)

// Required Octave headers
#include <octave/Cell.h>
#include <octave/dynamic-ld.h>
#include <octave/oct-env.h>
#include <octave/oct-map.h>
#include <octave/ov-scalar.h>
#include <octave/ov-fcn-handle.h>
#include <octave/parse.h>
#if SWIG_OCTAVE_PREREQ(4,2,0)
#include <octave/interpreter.h>
#else
#include <octave/toplev.h>
#endif
#include <octave/unwind-prot.h>
#if SWIG_OCTAVE_PREREQ(4,2,0)
#include <octave/call-stack.h>
#endif

#endif // !defined(_SWIG_OCTAVE_OCTHEADERS_HPP)
