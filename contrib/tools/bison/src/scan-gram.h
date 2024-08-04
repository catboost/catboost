/* Bison Grammar Scanner

   Copyright (C) 2006-2007, 2009-2015, 2018-2020 Free Software
   Foundation, Inc.

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

#ifndef SCAN_GRAM_H_
# define SCAN_GRAM_H_

/* Initialize the scanner to read file GRAM. */
void gram_scanner_open (const char *gram);
/* Close the open files.  */
void gram_scanner_close (void);

/* Free all the memory allocated to the scanner. */
void gram_scanner_free (void);
void gram_scanner_last_string_free (void);

# define GRAM_LEX_DECL int gram_lex (GRAM_STYPE *val, location *loc)
GRAM_LEX_DECL;

#endif /* !SCAN_GRAM_H_ */
