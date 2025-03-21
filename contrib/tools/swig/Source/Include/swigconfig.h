/* Source/Include/swigconfig.h.  Generated from swigconfig.h.in by configure.  */
/* Source/Include/swigconfig.h.in.  Generated from configure.ac by autoheader.  */

/* define if the Boost library is available */
/* #undef HAVE_BOOST */

/* define if the compiler supports basic C++11 syntax */
/* #undef HAVE_CXX11 */

/* define if the compiler supports basic C++14 syntax */
/* #undef HAVE_CXX14 */

/* define if the compiler supports basic C++17 syntax */
/* #undef HAVE_CXX17 */

/* define if the compiler supports basic C++20 syntax */
#define HAVE_CXX20 1

/* Define to 1 if you have the <inttypes.h> header file. */
/* #undef HAVE_INTTYPES_H */

/* Define to 1 if you have the 'dl' library (-ldl). */
#define HAVE_LIBDL 1

/* Define to 1 if you have the 'dld' library (-ldld). */
/* #undef HAVE_LIBDLD */

/* Define if you have PCRE2 library */
#define HAVE_PCRE 1

/* Define to 1 if you have the <stdint.h> header file. */
/* #undef HAVE_STDINT_H */

/* Define to 1 if you have the <stdio.h> header file. */
/* #undef HAVE_STDIO_H */

/* Define to 1 if you have the <stdlib.h> header file. */
/* #undef HAVE_STDLIB_H */

/* Define to 1 if you have the <strings.h> header file. */
/* #undef HAVE_STRINGS_H */

/* Define to 1 if you have the <string.h> header file. */
/* #undef HAVE_STRING_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
/* #undef HAVE_SYS_STAT_H */

/* Define to 1 if you have the <sys/types.h> header file. */
/* #undef HAVE_SYS_TYPES_H */

/* Define to 1 if you have the <unistd.h> header file. */
/* #undef HAVE_UNISTD_H */

/* Name of package */
#define PACKAGE "swig"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://www.swig.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "swig"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "swig 4.3.0"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "swig"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "4.3.0"

/* Define to 1 if all of the C89 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
/* #undef STDC_HEADERS */

/* Compiler that built SWIG */
#define SWIG_CXX "g++"

/* Directory for SWIG system-independent libraries */
#define SWIG_LIB "/var/empty/swig-4.3.0/share/swig/4.3.0"

/* Directory for SWIG system-independent libraries (Unix install on native
   Windows) */
#define SWIG_LIB_WIN_UNIX ""

/* Platform that SWIG is built for */
#define SWIG_PLATFORM "x86_64-pc-linux-gnu"

/* Version number of package */
#define VERSION "4.3.0"


/* Deal with attempt by Microsoft to deprecate C standard runtime functions */
#if defined(_MSC_VER)
# define _CRT_SECURE_NO_DEPRECATE
#endif

