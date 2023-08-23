/* clang-format off */

/* clang-format disabled because FindTIFF.cmake is very sensitive to the
 * formatting of below line being a single line.
 */
#define TIFFLIB_VERSION_STR "LIBTIFF, Version 4.5.1\nCopyright (c) 1988-1996 Sam Leffler\nCopyright (c) 1991-1996 Silicon Graphics, Inc."
/*
 * This define can be used in code that requires
 * compilation-related definitions specific to a
 * version or versions of the library.  Runtime
 * version checking should be done based on the
 * string returned by TIFFGetVersion.
 */
#define TIFFLIB_VERSION 20230609

/* The following defines have been added in 4.5.0 */
#define TIFFLIB_MAJOR_VERSION 4
#define TIFFLIB_MINOR_VERSION 5
#define TIFFLIB_MICRO_VERSION 1

/* Macro added in 4.5.0. Returns TRUE if the current libtiff version is
 * greater or equal to major.minor.micro
 */
#define TIFFLIB_AT_LEAST(major, minor, micro) \
    (TIFFLIB_MAJOR_VERSION > (major) || \
     (TIFFLIB_MAJOR_VERSION == (major) && TIFFLIB_MINOR_VERSION > (minor)) || \
     (TIFFLIB_MAJOR_VERSION == (major) && TIFFLIB_MINOR_VERSION == (minor) && \
      TIFFLIB_MICRO_VERSION >= (micro)))

/* clang-format on */
