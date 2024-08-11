/* Counterexample derivation trees

   Copyright (C) 2020-2021 Free Software Foundation, Inc.

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
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifndef DERIVATION_H
# define DERIVATION_H

# include <gl_linked_list.h>
# include <gl_xlist.h>

# include "gram.h"

/* Derivations are trees of symbols such that each nonterminal's
   children are symbols that produce that nonterminal if they are
   relevant to the counterexample.  The leaves of a derivation form a
   counterexample when printed.  */

typedef gl_list_t derivation_list;
typedef struct derivation derivation;

static inline derivation_list derivation_list_new (void)
{
  return gl_list_create_empty (GL_LINKED_LIST, NULL, NULL, NULL, true);
}

static inline bool
derivation_list_next (gl_list_iterator_t *it, derivation **d)
{
  const void *p = NULL;
  bool res = gl_list_iterator_next (it, &p, NULL);
  if (res)
    *d = (derivation *) p;
  else
    gl_list_iterator_free (it);
  return res;
}

void derivation_list_append (derivation_list dl, derivation *d);
void derivation_list_prepend (derivation_list dl, derivation *d);
void derivation_list_free (derivation_list dl);

derivation *derivation_new (symbol_number sym, derivation_list children);

static inline derivation *derivation_new_leaf (symbol_number sym)
{
  return derivation_new (sym, NULL);
}

// Number of symbols.
size_t derivation_size (const derivation *deriv);
void derivation_print (const derivation *deriv, FILE *out, const char *prefix);
void derivation_print_leaves (const derivation *deriv, FILE *out);
void derivation_free (derivation *deriv);
void derivation_retain (derivation *deriv);

// A derivation denoting the position of the dot.
derivation *derivation_dot (void);

#endif /* DERIVATION_H */
