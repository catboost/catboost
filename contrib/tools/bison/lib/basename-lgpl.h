/*  Extract the last component (base name) of a file name.

    Copyright (C) 1998, 2001, 2003-2006, 2009-2020 Free Software Foundation,
    Inc.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifndef _BASENAME_LGPL_H
#define _BASENAME_LGPL_H

#include <stddef.h>

#ifndef DOUBLE_SLASH_IS_DISTINCT_ROOT
# define DOUBLE_SLASH_IS_DISTINCT_ROOT 0
#endif

#ifdef __cplusplus
extern "C" {
#endif


/* Return the address of the last file name component of FILENAME.
   If FILENAME has some trailing slash(es), they are considered to be
   part of the last component.
   If FILENAME has no relative file name components because it is a file
   system root, return the empty string.
   Examples:
              FILENAME      RESULT
              "foo.c"       "foo.c"
              "foo/bar.c"   "bar.c"
              "/foo/bar.c"  "bar.c"
              "foo/bar/"    "bar/"
              "foo/bar//"   "bar//"
              "/"           ""
              "//"          ""
              ""            ""
   The return value is a tail of the given FILENAME; do NOT free() it!  */

/* This function was traditionally called 'basename', but we avoid this
   function name because
     * Various platforms have different functions in their libc.
       In particular, the glibc basename(), defined in <string.h>, does
       not consider trailing slashes to be part of the component:
              FILENAME      RESULT
              "foo/bar/"    ""
              "foo/bar//"   ""
     * The 'basename' command eliminates trailing slashes and for a root
       produces a non-empty result:
              FILENAME      RESULT
              "foo/bar/"    "bar"
              "foo/bar//"   "bar"
              "/"           "/"
              "//"          "/"
 */
extern char *last_component (char const *filename) _GL_ATTRIBUTE_PURE;

/* Return the length of the basename FILENAME.
   Typically FILENAME is the value returned by base_name or last_component.
   Act like strlen (FILENAME), except omit all trailing slashes.  */
extern size_t base_len (char const *filename) _GL_ATTRIBUTE_PURE;


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _BASENAME_LGPL_H */
