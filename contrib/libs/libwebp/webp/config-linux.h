/* src/webp/config.h.  Generated from config.h.in by configure.  */
/* src/webp/config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Set to 1 if __builtin_bswap16 is available */
#define HAVE_BUILTIN_BSWAP16 1

/* Set to 1 if __builtin_bswap32 is available */
#define HAVE_BUILTIN_BSWAP32 1

/* Set to 1 if __builtin_bswap64 is available */
#define HAVE_BUILTIN_BSWAP64 1

/* Define to 1 if you have the <cpu-features.h> header file. */
/* #undef HAVE_CPU_FEATURES_H */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <GLUT/glut.h> header file. */
/* #undef HAVE_GLUT_GLUT_H */

/* Define to 1 if you have the <GL/glut.h> header file. */
/* #undef HAVE_GL_GLUT_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <OpenGL/glut.h> header file. */
/* #undef HAVE_OPENGL_GLUT_H */

/* Have PTHREAD_PRIO_INHERIT. */
#define HAVE_PTHREAD_PRIO_INHERIT 1

/* Define to 1 if you have the <shlwapi.h> header file. */
/* #undef HAVE_SHLWAPI_H */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the <wincodec.h> header file. */
/* #undef HAVE_WINCODEC_H */

/* Define to 1 if you have the <windows.h> header file. */
/* #undef HAVE_WINDOWS_H */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "libwebp"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://bugs.chromium.org/p/webp"

/* Define to the full name of this package. */
#define PACKAGE_NAME "libwebp"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "libwebp 1.2.2"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "libwebp"

/* Define to the home page for this package. */
#define PACKAGE_URL "https://developers.google.com/speed/webp"

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.2.2"

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "1.2.2"

/* Set to 1 if GIF library is installed */
#define WEBP_HAVE_GIF 1

/* Set to 1 if OpenGL is supported */
/* #undef WEBP_HAVE_GL */

/* Set to 1 if JPEG library is installed */
#define WEBP_HAVE_JPEG 1

/* Set to 1 if NEON is supported */
/* #undef WEBP_HAVE_NEON */

/* Set to 1 if runtime detection of NEON is enabled */
/* #undef WEBP_HAVE_NEON_RTCD */

/* Set to 1 if PNG library is installed */
#define WEBP_HAVE_PNG 1

/* Set to 1 if SDL library is installed */
/* #undef WEBP_HAVE_SDL */

/* Set to 1 if SSE2 is supported */
#define WEBP_HAVE_SSE2 1

/* Set to 1 if SSE4.1 is supported */
#define WEBP_HAVE_SSE41 1

/* Set to 1 if TIFF library is installed */
#define WEBP_HAVE_TIFF 1

/* Enable near lossless encoding */
#define WEBP_NEAR_LOSSLESS 1

/* Undefine this to disable thread support. */
#define WEBP_USE_THREAD 1

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif
