/* Grammar reduction for Bison.

   Copyright (C) 2000-2002, 2007, 2009-2015, 2018 Free Software
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

#ifndef REDUCE_H_
# define REDUCE_H_

void reduce_grammar (void);
void reduce_output (FILE *out);
bool reduce_token_unused_in_grammar (symbol_number i);

/** Whether symbol \a i is useless in the grammar.
 * \pre  reduce_grammar was called before.
 */
bool reduce_nonterminal_useless_in_grammar (const sym_content *sym);

void reduce_free (void);

extern unsigned nuseless_nonterminals;
extern unsigned nuseless_productions;
#endif /* !REDUCE_H_ */
