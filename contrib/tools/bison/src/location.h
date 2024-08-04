/* Locations for Bison

   Copyright (C) 2002, 2004-2015, 2018-2020 Free Software Foundation,
   Inc.

   This file is part of Bison, the GNU Compiler Compiler.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef LOCATION_H_
# define LOCATION_H_

# include <stdbool.h>
# include <stdio.h>
# include <string.h> /* strcmp */

# include "uniqstr.h"

/* A boundary between two characters.  */
typedef struct
{
  /* The name of the file that contains the boundary.  */
  uniqstr file;

  /* If positive, the line (starting at 1) that contains the boundary.
     If this is INT_MAX, the line number has overflowed.

     Meaningless and not displayed if nonpositive.
  */
  int line;

  /* If positive, the column (starting at 1) just after the boundary.
     This is neither a byte count, nor a character count; it is a
     (visual) column count.  If this is INT_MAX, the column number has
     overflowed.

     Meaningless and not displayed if nonpositive.  */
  int column;

  /* If nonnegative, the byte number (starting at 0) in the current
     line.  Not displayed (unless --trace=location).  */
  int byte;

} boundary;

/* Set the position of \a p. */
static inline void
boundary_set (boundary *p, const char *f, int l, int c, int b)
{
  p->file = f;
  p->line = l;
  p->column = c;
  p->byte = b;
}

/* Return -1, 0, 1, depending whether a is before, equal, or
   after b.  */
static inline int
boundary_cmp (boundary a, boundary b)
{
  /* Locations with no file first.  */
  int res =
    a.file && b.file ? strcmp (a.file, b.file)
    : a.file ? 1
    : b.file ? -1
    : 0;
  if (!res)
    res = a.line - b.line;
  if (!res)
    res = a.column - b.column;
  return res;
}

/* Return nonzero if A and B are equal boundaries.  */
static inline bool
equal_boundaries (boundary a, boundary b)
{
  return (a.column == b.column
          && a.line == b.line
          && UNIQSTR_EQ (a.file, b.file));
}

/* A location, that is, a region of source code.  */
typedef struct
{
  /* Boundary just before the location starts.  */
  boundary start;

  /* Boundary just after the location ends.  */
  boundary end;

} location;

# define GRAM_LTYPE location

# define EMPTY_LOCATION_INIT {{NULL, 0, 0, 0}, {NULL, 0, 0, 0}}
extern location const empty_loc;

/* Set *LOC and adjust scanner cursor to account for token TOKEN of
   size SIZE.  */
void location_compute (location *loc,
                       boundary *cur, char const *token, size_t size);

/* Print location to file.
   Return number of actually printed characters.
   Warning: uses quotearg's slot 3. */
int location_print (location loc, FILE *out);

/* Prepare the use of location_caret.  */
void caret_init (void);

/* Free any allocated resources and close any open file handles that are
   left-over by the usage of location_caret.  */
void caret_free (void);

/* Quote the line containing LOC onto OUT.  Highlight the part of LOC
   with the color STYLE.  */
void location_caret (location loc, const char* style, FILE *out);

/* Display a suggestion of replacement for LOC with S.  To call after
   location_caret.  */
void location_caret_suggestion (location loc, const char *s, FILE *out);

/* Return -1, 0, 1, depending whether a is before, equal, or
   after b.  */
static inline int
location_cmp (location a, location b)
{
  int res = boundary_cmp (a.start, b.start);
  if (!res)
    res = boundary_cmp (a.end, b.end);
  return res;
}

/* Whether this is the empty location.  */
bool location_empty (location loc);

/* STR must be formatted as 'file:line.column@byte' or 'file:line.column',
   it will be modified.  */
void boundary_set_from_string (boundary *bound, char *str);

#endif /* ! defined LOCATION_H_ */
