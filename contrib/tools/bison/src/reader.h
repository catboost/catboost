/* Input parser for Bison

   Copyright (C) 2000-2003, 2005-2007, 2009-2015, 2018-2021 Free
   Software Foundation, Inc.

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

#ifndef READER_H_
# define READER_H_

# include "location.h"
# include "symlist.h"
# include "named-ref.h"

# include "parse-gram.h"

typedef struct merger_list
{
  struct merger_list* next;
  uniqstr name;
  /* One symbol whose type is the one used by all the symbols on which
     this merging function is used.  */
  symbol *sym;
  /* Where SYM was bound to this merging function.  */
  location type_declaration_loc;
} merger_list;

void grammar_start_symbol_set (symbol *sym, location loc);
void grammar_current_rule_begin (symbol *lhs, location loc,
                                 named_ref *lhs_named_ref);
void grammar_current_rule_end (location loc);
void grammar_midrule_action (void);
/* Apply %empty to the current rule.  */
void grammar_current_rule_empty_set (location loc);
void grammar_current_rule_prec_set (symbol *precsym, location loc);
void grammar_current_rule_dprec_set (int dprec, location loc);
void grammar_current_rule_merge_set (uniqstr name, location loc);
void grammar_current_rule_expect_sr (int count, location loc);
void grammar_current_rule_expect_rr (int count, location loc);
void grammar_current_rule_symbol_append (symbol *sym, location loc,
                                         named_ref *nref);
/* Attach an ACTION to the current rule.  */
void grammar_current_rule_action_append (const char *action, location loc,
                                         named_ref *nref, uniqstr tag);
/* Attach a PREDICATE to the current rule.  */
void grammar_current_rule_predicate_append (const char *predicate, location loc);
void reader (const char *gram);
void free_merger_functions (void);

extern merger_list *merge_functions;

/* Was %union seen?  */
extern bool union_seen;

/* Should rules have a default precedence?  */
extern bool default_prec;

#endif /* !READER_H_ */
