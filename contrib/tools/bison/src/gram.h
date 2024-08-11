/* Data definitions for internal representation of Bison's input.

   Copyright (C) 1984, 1986, 1989, 1992, 2001-2007, 2009-2015, 2018-2021
   Free Software Foundation, Inc.

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

#ifndef GRAM_H_
# define GRAM_H_

/* Representation of the grammar rules:

   NTOKENS is the number of tokens, and NNTERMS is the number of
   variables (nonterminals).  NSYMS is the total number, ntokens +
   nnterms.

   Each symbol (either token or variable) receives a symbol number.
   Numbers 0 to NTOKENS - 1 are for tokens, and NTOKENS to NSYMS - 1
   are for variables.  Symbol number zero is the end-of-input token.
   This token is counted in ntokens.  The true number of token values
   assigned is NTOKENS reduced by one for each alias declaration.

   The rules receive rule numbers 1 to NRULES in the order they are
   written.  More precisely Bison augments the grammar with the
   initial rule, '$accept: START-SYMBOL $end', which is numbered 1,
   all the user rules are 2, 3 etc.  Each time a rule number is
   presented to the user, we subtract 1, so *displayed* rule numbers
   are 0, 1, 2...

   Internally, we cannot use the number 0 for a rule because for
   instance RITEM stores both symbol (the RHS) and rule numbers: the
   symbols are integers >= 0, and rule numbers are stored negative.
   Therefore 0 cannot be used, since it would be both the rule number
   0, and the token $end.

   Actions are accessed via the rule number.

   The rules themselves are described by several arrays: amongst which
   RITEM, and RULES.

   RULES is an array of rules, whose members are:

   RULES[R].lhs -- the symbol of the left hand side of rule R.

   RULES[R].rhs -- the beginning of the portion of RITEM for rule R.

   RULES[R].prec -- the symbol providing the precedence level of R.

   RULES[R].precsym -- the symbol attached (via %prec) to give its
   precedence to R.  Of course, if set, it is equal to 'prec', but we
   need to distinguish one from the other when reducing: a symbol used
   in a %prec is not useless.

   RULES[R].assoc -- the associativity of R.

   RULES[R].dprec -- the dynamic precedence level of R (for GLR
   parsing).

   RULES[R].merger -- index of merging function for R (for GLR
   parsing).

   RULES[R].line -- the line where R was defined.

   RULES[R].useful -- whether the rule is used.  False if thrown away
   by reduce().

   The right hand side is stored as symbol numbers in a portion of
   RITEM.

   The length of the portion is one greater than the number of symbols
   in the rule's right hand side.  The last element in the portion
   contains -R, which identifies it as the end of a portion and says
   which rule it is for.

   The portions of RITEM come in order of increasing rule number.
   NRITEMS is the total length of RITEM.  Each element of RITEM is
   called an "item" and its index in RITEM is an item number.

   Item numbers are used in the finite state machine to represent
   places that parsing can get to.

   SYMBOLS[I]->prec records the precedence level of each symbol.

   Precedence levels are assigned in increasing order starting with 1
   so that numerically higher precedence values mean tighter binding
   as they ought to.  Zero as a symbol or rule's precedence means none
   is assigned.

   Associativities are recorded similarly in SYMBOLS[I]->assoc.  */

# include "system.h"

# include "location.h"
# include "symtab.h"

# define ISTOKEN(i)     ((i) < ntokens)
# define ISVAR(i)       ((i) >= ntokens)

extern int nsyms;
extern int ntokens;
extern int nnterms;

/* Elements of ritem. */
typedef int item_number;
# define ITEM_NUMBER_MAX INT_MAX
extern item_number *ritem;
extern int nritems;

/* Indices into ritem. */
typedef unsigned int item_index;

/* There is weird relationship between OT1H item_number and OTOH
   symbol_number and rule_number: we store the latter in
   item_number.  symbol_number values are stored as-is, while
   the negation of (rule_number + 1) is stored.

   Therefore, a symbol_number must be a valid item_number, and we
   sometimes have to perform the converse transformation.  */

static inline item_number
symbol_number_as_item_number (symbol_number sym)
{
  return sym;
}

static inline symbol_number
item_number_as_symbol_number (item_number i)
{
  return i;
}

static inline bool
item_number_is_symbol_number (item_number i)
{
  return i >= 0;
}

/* Rule numbers.  */
typedef int rule_number;
# define RULE_NUMBER_MAX INT_MAX

static inline item_number
rule_number_as_item_number (rule_number r)
{
  return -1 - r;
}

static inline rule_number
item_number_as_rule_number (item_number i)
{
  return -1 - i;
}

static inline bool
item_number_is_rule_number (item_number i)
{
  return i < 0;
}


/*--------.
| Rules.  |
`--------*/

typedef struct
{
  /* The number of the rule in the source.  It is usually the index in
     RULES too, except if there are useless rules.  */
  rule_number code;

  /* The index in RULES.  Usually the rule number in the source,
     except if some rules are useless.  */
  rule_number number;

  sym_content *lhs;
  item_number *rhs;

  /* This symbol provides both the associativity, and the precedence. */
  sym_content *prec;

  int dprec;
  int merger;

  /* This symbol was attached to the rule via %prec. */
  sym_content *precsym;

  /* Location of the rhs.  */
  location location;
  bool useful;
  bool is_predicate;

  /* Counts of the numbers of expected conflicts for this rule, or -1 if none
     given. */
  int expected_sr_conflicts;
  int expected_rr_conflicts;

  const char *action;
  location action_loc;
} rule;

/* The used rules (size NRULES).  */
extern rule *rules;
extern rule_number nrules;

/* Get the rule associated to this item.  ITEM points inside RITEM.  */
static inline rule const *
item_rule (item_number const *item)
{
  item_number const *sp = item;
  while (!item_number_is_rule_number (*sp))
    ++sp;
  rule_number r = item_number_as_rule_number (*sp);
  return &rules[r];
}

/* Pretty-print this ITEM (as in the report).  ITEM points inside
   RITEM.  PREVIOUS_RULE is used to see if the lhs is common, in which
   case LHS is factored.  Passing NULL is fine.  */
void item_print (item_number *item, rule const *previous_rule,
                 FILE *out);

/* A function that selects a rule.  */
typedef bool (*rule_filter) (rule const *);

/* Whether the rule has a 'number' smaller than NRULES.  That is, it
   is useful in the grammar.  */
bool rule_useful_in_grammar_p (rule const *r);

/* Whether the rule has a 'number' higher than NRULES.  That is, it is
   useless in the grammar.  */
bool rule_useless_in_grammar_p (rule const *r);

/* Whether the rule is not flagged as useful but is useful in the
   grammar.  In other words, it was discarded because of conflicts.  */
bool rule_useless_in_parser_p (rule const *r);

/* Whether the rule has a single RHS, and no user action. */
bool rule_useless_chain_p (rule const *r);

/* Print this rule's number and lhs on OUT.  If a PREVIOUS_LHS was
   already displayed (by a previous call for another rule), avoid
   useless repetitions.  */
void rule_lhs_print (rule const *r, sym_content const *previous_lhs,
                     FILE *out);
void rule_lhs_print_xml (rule const *r, FILE *out, int level);

/* The length of the RHS.  */
size_t rule_rhs_length (rule const *r);

/* Print this rule's RHS on OUT.  */
void rule_rhs_print (rule const *r, FILE *out);

/* Print this rule on OUT.  If a PREVIOUS_RULE was already displayed,
   avoid useless repetitions of their LHS. */
void rule_print (rule const *r, rule const *prev_rule, FILE *out);



/* Table of the symbols, indexed by the symbol number. */
extern symbol **symbols;

/* TOKEN_TRANSLATION -- a table indexed by a token number as returned
   by the user's yylex routine, it yields the internal token number
   used by the parser and throughout bison.  */
extern symbol_number *token_translations;
extern int max_code;



/* Dump RITEM for traces. */
void ritem_print (FILE *out);

/* The size of the longest rule RHS.  */
size_t ritem_longest_rhs (void);

/* Print the grammar's rules that match FILTER on OUT under TITLE.  */
void grammar_rules_partial_print (FILE *out, const char *title,
                                  rule_filter filter);

/* Print the grammar's useful rules on OUT.  */
void grammar_rules_print (FILE *out);
/* Print all of the grammar's rules with a "usefulness" attribute.  */
void grammar_rules_print_xml (FILE *out, int level);

/* Dump the grammar. */
void grammar_dump (FILE *out, const char *title);

/* Report on STDERR the rules that are not flagged USEFUL, using the
   MESSAGE (which can be 'rule useless in grammar' when invoked after grammar
   reduction, or 'rule useless in parser due to conflicts' after conflicts
   were taken into account).  */
void grammar_rules_useless_report (const char *message);

/* Free the packed grammar. */
void grammar_free (void);

/* The version %required by the grammar file, as an int (100 * major +
   minor).  0 if unspecified.  */
extern int required_version;

#endif /* !GRAM_H_ */
