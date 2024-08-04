/* Concatenate path components.
   Copyright (C) 2018-2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 3 of the License, or any
   later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Akim Demaille <akim@lrde.epita.fr>.  */

#ifndef _PATH_JOIN_H
# define _PATH_JOIN_H

# ifdef __cplusplus
extern "C" {
# endif


/* Concatenate two paths together.  PATH2 may be null, or empty, or
   absolute: do what is right.  Return a freshly allocated
   filename.  */
char *
xpath_join (const char *path1, const char *path2);


# ifdef __cplusplus
}
# endif

#endif /* _PATH_JOIN_H */
