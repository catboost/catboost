/* define if your compiler has __attribute__ */
#define HAVE___ATTRIBUTE__ /**/

/* most gcc compilers know a function __attribute__((__warn_unused_result__)) */
#define GCC_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))


/* Define to 1 if you have the declaration of `ffs', and to 0 if you don't. */
#define HAVE_DECL_FFS 1

/* Define to 1 if you have the declaration of `__builtin_ffs', and to 0 if you
   don't. */
#define HAVE_DECL___BUILTIN_FFS 1

/* Define to 1 if you have the declaration of `_BitScanForward', and to 0 if
   you don't. */
#define HAVE_DECL__BITSCANFORWARD 0


/* Define to 1 if you have the declaration of `strcasecmp', and to 0 if you
   don't. */
#define HAVE_DECL_STRCASECMP 1

/* Define to 1 if you have the declaration of `_stricmp', and to 0 if you
   don't. */
#define HAVE_DECL__STRICMP 0


/* Define to 1 if you have the declaration of `strncasecmp', and to 0 if you
   don't. */
#define HAVE_DECL_STRNCASECMP 1

/* Define to 1 if you have the declaration of `_strnicmp', and to 0 if you
   don't. */
#define HAVE_DECL__STRNICMP 0


/* Define to 1 if you have the declaration of `snprintf', and to 0 if you
   don't. */
#define HAVE_DECL_SNPRINTF 1

/* Define to 1 if you have the declaration of `_snprintf', and to 0 if you
   don't. */
#define HAVE_DECL__SNPRINTF 0


/* use gmp to implement isl_int */
/* #undef USE_GMP_FOR_MP */

/* use imath to implement isl_int */
#define USE_IMATH_FOR_MP

/* Use small integer optimization */
#define USE_SMALL_INT_OPT

#include <isl_config_post.h>
