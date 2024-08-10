/* readline.h --- Simple implementation of readline.
   Copyright (C) 2005, 2009-2020 Free Software Foundation, Inc.
   Written by Simon Josefsson

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

#ifndef GL_READLINE_H
#define GL_READLINE_H

#if HAVE_READLINE_READLINE_H
/* <readline/readline.h> makes use of the FILE type without including
   <stdio.h> itself. */
# include <stdio.h>
# include <readline/readline.h>
#else
/* Prints a prompt PROMPT and then reads and returns a single line of
   text from the user.  If PROMPT is NULL or the empty string, no
   prompt is displayed.  The returned line is allocated with malloc;
   the caller should free the line when it has finished with it. */
extern char *readline (const char *prompt);
#endif

#endif /* GL_READLINE_H */
