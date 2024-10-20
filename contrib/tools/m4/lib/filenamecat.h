/* Concatenate two arbitrary file names.

   Copyright (C) 1996-1997, 2003, 2005, 2007, 2009-2016 Free Software
   Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* Written by Jim Meyering.  */

#if GNULIB_FILENAMECAT
char *file_name_concat (char const *dir, char const *base,
                        char **base_in_result);
#endif

char *mfile_name_concat (char const *dir, char const *base,
                         char **base_in_result);
