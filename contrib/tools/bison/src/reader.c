/* Input parser for Bison

   Copyright (C) 1984, 1986, 1989, 1992, 1998, 2000-2003, 2005-2007,
   2009-2015, 2018-2020 Free Software Foundation, Inc.

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

#include <config.h>
#include "system.h"

#include <quote.h>

#include "complain.h"
#include "conflicts.h"
#include "files.h"
#include "fixits.h"
#include "getargs.h"
#include "gram.h"
#include "muscle-tab.h"
#include "reader.h"
#include "symlist.h"
#include "symtab.h"
#include "scan-gram.h"
#include "scan-code.h"

static void prepare_percent_define_front_end_variables (void);
static void check_and_convert_grammar (void);

static symbol_list *grammar = NULL;
static bool start_flag = false;
merger_list *merge_functions;

/* Was %union seen?  */
bool union_seen = false;

/* Should rules have a default precedence?  */
bool default_prec = true;

/*-----------------------.
| Set the start symbol.  |
`-----------------------*/

void
grammar_start_symbol_set (symbol *sym, location loc)
{
  if (start_flag)
    complain (&loc, complaint, _("multiple %s declarations"), "%start");
  else
    {
      start_flag = true;
      startsymbol = sym;
      startsymbol_loc = loc;
    }
}



/*------------------------------------------------------------------------.
| Return the merger index for a merging function named NAME.  Records the |
| function, if new, in MERGER_LIST.                                       |
`------------------------------------------------------------------------*/

static int
get_merge_function (uniqstr name)
{
  if (! glr_parser)
    return 0;

  merger_list *syms;
  merger_list head;
  int n;

  head.next = merge_functions;
  for (syms = &head, n = 1; syms->next; syms = syms->next, n += 1)
    if (UNIQSTR_EQ (name, syms->next->name))
      break;
  if (syms->next == NULL)
    {
      syms->next = xmalloc (sizeof syms->next[0]);
      syms->next->name = uniqstr_new (name);
      /* After all symbol type declarations have been parsed, packgram invokes
         record_merge_function_type to set the type.  */
      syms->next->type = NULL;
      syms->next->next = NULL;
      merge_functions = head.next;
    }
  return n;
}

/*-------------------------------------------------------------------------.
| For the existing merging function with index MERGER, record the result   |
| type as TYPE as required by the lhs of the rule whose %merge declaration |
| is at DECLARATION_LOC.                                                   |
`-------------------------------------------------------------------------*/

static void
record_merge_function_type (int merger, uniqstr type, location declaration_loc)
{
  if (merger <= 0)
    return;

  if (type == NULL)
    type = uniqstr_new ("");

  merger_list *merge_function;
  int merger_find = 1;
  for (merge_function = merge_functions;
       merge_function != NULL && merger_find != merger;
       merge_function = merge_function->next)
    merger_find += 1;
  aver (merge_function != NULL && merger_find == merger);
  if (merge_function->type != NULL && !UNIQSTR_EQ (merge_function->type, type))
    {
      complain (&declaration_loc, complaint,
                _("result type clash on merge function %s: "
                "<%s> != <%s>"),
                quote (merge_function->name), type,
                merge_function->type);
      subcomplain (&merge_function->type_declaration_loc, complaint,
                   _("previous declaration"));
    }
  merge_function->type = uniqstr_new (type);
  merge_function->type_declaration_loc = declaration_loc;
}

/*--------------------------------------.
| Free all merge-function definitions.  |
`--------------------------------------*/

void
free_merger_functions (void)
{
  merger_list *L0 = merge_functions;
  while (L0)
    {
      merger_list *L1 = L0->next;
      free (L0);
      L0 = L1;
    }
}


/*-------------------------------------------------------------------.
| Parse the input grammar into a one symbol_list structure.  Each    |
| rule is represented by a sequence of symbols: the left hand side   |
| followed by the contents of the right hand side, followed by a     |
| null pointer instead of a symbol to terminate the rule.  The next  |
| symbol is the lhs of the following rule.                           |
|                                                                    |
| All actions are copied out, labelled by the rule number they apply |
| to.                                                                |
`-------------------------------------------------------------------*/

/* The (currently) last symbol of GRAMMAR. */
static symbol_list *grammar_end = NULL;

/* Append SYM to the grammar.  */
static symbol_list *
grammar_symbol_append (symbol *sym, location loc)
{
  symbol_list *p = symbol_list_sym_new (sym, loc);

  if (grammar_end)
    grammar_end->next = p;
  else
    grammar = p;

  grammar_end = p;

  /* A null SYM stands for an end of rule; it is not an actual
     part of it.  */
  if (sym)
    ++nritems;

  return p;
}

static void
assign_named_ref (symbol_list *p, named_ref *name)
{
  symbol *sym = p->content.sym;

  if (name->id == sym->tag)
    {
      complain (&name->loc, Wother,
                _("duplicated symbol name for %s ignored"),
                quote (sym->tag));
      named_ref_free (name);
    }
  else
    p->named_ref = name;
}


/* The rule currently being defined, and the previous rule.
   CURRENT_RULE points to the first LHS of the current rule, while
   PREVIOUS_RULE_END points to the *end* of the previous rule (NULL).  */
static symbol_list *current_rule = NULL;
static symbol_list *previous_rule_end = NULL;


/*----------------------------------------------.
| Create a new rule for LHS in to the GRAMMAR.  |
`----------------------------------------------*/

void
grammar_current_rule_begin (symbol *lhs, location loc,
                            named_ref *lhs_name)
{
  /* Start a new rule and record its lhs.  */
  ++nrules;
  previous_rule_end = grammar_end;

  current_rule = grammar_symbol_append (lhs, loc);
  if (lhs_name)
    assign_named_ref (current_rule, named_ref_copy (lhs_name));

  /* Mark the rule's lhs as a nonterminal if not already so.  */
  if (lhs->content->class == unknown_sym || lhs->content->class == pct_type_sym)
    symbol_class_set (lhs, nterm_sym, empty_loc, false);
  else if (lhs->content->class == token_sym)
    complain (&loc, complaint, _("rule given for %s, which is a token"),
              lhs->tag);
}


/*----------------------------------------------------------------------.
| A symbol should be used if either:                                    |
|   1. It has a destructor.                                             |
|   2. The symbol is a midrule symbol (i.e., the generated LHS          |
|      replacing a midrule action) that was assigned to or used, as in  |
|      "exp: { $$ = 1; } { $$ = $1; }".                                 |
`----------------------------------------------------------------------*/

static bool
symbol_should_be_used (symbol_list const *s, bool *midrule_warning)
{
  if (symbol_code_props_get (s->content.sym, destructor)->code)
    return true;
  if ((s->midrule && s->midrule->action_props.is_value_used)
      || (s->midrule_parent_rule
          && (symbol_list_n_get (s->midrule_parent_rule,
                                 s->midrule_parent_rhs_index)
              ->action_props.is_value_used)))
    {
      *midrule_warning = true;
      return true;
    }
  return false;
}

/*-----------------------------------------------------------------.
| Check that the rule R is properly defined.  For instance, there  |
| should be no type clash on the default action.  Possibly install |
| the default action.                                              |
`-----------------------------------------------------------------*/

static void
grammar_rule_check_and_complete (symbol_list *r)
{
  /* Type check.

     If there is an action, then there is nothing we can do: the user
     is allowed to shoot herself in the foot.

     Don't worry about the default action if $$ is untyped, since $$'s
     value can't be used.  */
  if (!r->action_props.code && r->content.sym->content->type_name)
    {
      symbol *first_rhs = r->next->content.sym;
      /* If $$ is being set in default way, report if any type mismatch.  */
      if (first_rhs)
        {
          char const *lhs_type = r->content.sym->content->type_name;
          char const *rhs_type =
            first_rhs->content->type_name ? first_rhs->content->type_name : "";
          if (!UNIQSTR_EQ (lhs_type, rhs_type))
            complain (&r->rhs_loc, Wother,
                      _("type clash on default action: <%s> != <%s>"),
                      lhs_type, rhs_type);
          else
            {
              /* Install the default action only for C++.  */
              const bool is_cxx =
                STREQ (language->language, "c++")
                || (skeleton && (STREQ (skeleton, "glr.cc")
                                 || STREQ (skeleton, "lalr1.cc")));
              if (is_cxx)
                {
                  code_props_rule_action_init (&r->action_props, "{ $$ = $1; }",
                                               r->rhs_loc, r,
                                               /* name */ NULL,
                                               /* type */ NULL,
                                               /* is_predicate */ false);
                  code_props_translate_code (&r->action_props);
                }
            }
        }
      /* Warn if there is no default for $$ but we need one.  */
      else
        complain (&r->rhs_loc, Wother,
                  _("empty rule for typed nonterminal, and no action"));
    }

  /* Check that symbol values that should be used are in fact used.  */
  {
    int n = 0;
    for (symbol_list const *l = r; l && l->content.sym; l = l->next, ++n)
      {
        bool midrule_warning = false;
        if (!l->action_props.is_value_used
            && symbol_should_be_used (l, &midrule_warning)
            /* The default action, $$ = $1, 'uses' both.  */
            && (r->action_props.code || (n != 0 && n != 1)))
          {
            warnings warn_flag = midrule_warning ? Wmidrule_values : Wother;
            if (n)
              complain (&l->sym_loc, warn_flag, _("unused value: $%d"), n);
            else
              complain (&l->rhs_loc, warn_flag, _("unset value: $$"));
          }
      }
  }

  /* Check that %empty => empty rule.  */
  if (r->percent_empty_loc.start.file
      && r->next && r->next->content.sym)
    {
      complain (&r->percent_empty_loc, complaint,
                _("%%empty on non-empty rule"));
      fixits_register (&r->percent_empty_loc, "");
    }

  /* Check that empty rule => %empty.  */
  if (!(r->next && r->next->content.sym)
      && !r->midrule_parent_rule
      && !r->percent_empty_loc.start.file
      && warning_is_enabled (Wempty_rule))
    {
      complain (&r->rhs_loc, Wempty_rule, _("empty rule without %%empty"));
      if (feature_flag & feature_caret)
        location_caret_suggestion (r->rhs_loc, "%empty", stderr);
      location loc = r->rhs_loc;
      loc.end = loc.start;
      fixits_register (&loc, " %empty ");
    }

  /* See comments in grammar_current_rule_prec_set for how POSIX
     mandates this complaint.  It's only for identifiers, so skip
     it for char literals and strings, which are always tokens.  */
  if (r->ruleprec
      && r->ruleprec->tag[0] != '\'' && r->ruleprec->tag[0] != '"'
      && r->ruleprec->content->status != declared
      && !r->ruleprec->content->prec)
    complain (&r->rhs_loc, Wother,
              _("token for %%prec is not defined: %s"), r->ruleprec->tag);

  /* Check that the (main) action was not typed.  */
  if (r->action_props.type)
    complain (&r->rhs_loc, Wother,
              _("only midrule actions can be typed: %s"), r->action_props.type);
}


/*-------------------------------------.
| End the currently being grown rule.  |
`-------------------------------------*/

void
grammar_current_rule_end (location loc)
{
  /* Put an empty link in the list to mark the end of this rule  */
  grammar_symbol_append (NULL, grammar_end->rhs_loc);
  current_rule->rhs_loc = loc;
}


/*-------------------------------------------------------------------.
| The previous action turns out to be a midrule action.  Attach it   |
| to the current rule, i.e., create a dummy symbol, attach it this   |
| midrule action, and append this dummy nonterminal to the current   |
| rule.                                                              |
`-------------------------------------------------------------------*/

void
grammar_midrule_action (void)
{
  /* Since the action was written out with this rule's number, we must
     give the new rule this number by inserting the new rule before
     it.  */

  /* Make a DUMMY nonterminal, whose location is that of the midrule
     action.  Create the MIDRULE.  */
  location dummy_loc = current_rule->action_props.location;
  symbol *dummy = dummy_symbol_get (dummy_loc);
  symbol_type_set(dummy,
                  current_rule->action_props.type, current_rule->action_props.location);
  symbol_list *midrule = symbol_list_sym_new (dummy, dummy_loc);

  /* Remember named_ref of previous action. */
  named_ref *action_name = current_rule->action_props.named_ref;

  /* Make a new rule, whose body is empty, before the current one, so
     that the action just read can belong to it.  */
  ++nrules;
  ++nritems;
  /* Attach its location and actions to that of the DUMMY.  */
  midrule->rhs_loc = dummy_loc;
  code_props_rule_action_init (&midrule->action_props,
                               current_rule->action_props.code,
                               current_rule->action_props.location,
                               midrule,
                               /* name_ref */ NULL,
                               /* type */ NULL,
                               current_rule->action_props.is_predicate);
  code_props_none_init (&current_rule->action_props);

  midrule->expected_sr_conflicts = current_rule->expected_sr_conflicts;
  midrule->expected_rr_conflicts = current_rule->expected_rr_conflicts;
  current_rule->expected_sr_conflicts = -1;
  current_rule->expected_rr_conflicts = -1;

  if (previous_rule_end)
    previous_rule_end->next = midrule;
  else
    grammar = midrule;

  /* End the dummy's rule.  */
  midrule->next = symbol_list_sym_new (NULL, dummy_loc);
  midrule->next->next = current_rule;

  previous_rule_end = midrule->next;

  /* Insert the dummy nonterminal replacing the midrule action into
     the current rule.  Bind it to its dedicated rule.  */
  grammar_current_rule_symbol_append (dummy, dummy_loc,
                                      action_name);
  grammar_end->midrule = midrule;
  midrule->midrule_parent_rule = current_rule;
  midrule->midrule_parent_rhs_index = symbol_list_length (current_rule->next);
}

/* Set the precedence symbol of the current rule to PRECSYM. */

void
grammar_current_rule_prec_set (symbol *precsym, location loc)
{
  /* POSIX says that any identifier is a nonterminal if it does not
     appear on the LHS of a grammar rule and is not defined by %token
     or by one of the directives that assigns precedence to a token.
     We ignore this here because the only kind of identifier that
     POSIX allows to follow a %prec is a token and because assuming
     it's a token now can produce more logical error messages.
     Nevertheless, grammar_rule_check_and_complete does obey what we
     believe is the real intent of POSIX here: that an error be
     reported for any identifier that appears after %prec but that is
     not defined separately as a token.  */
  symbol_class_set (precsym, token_sym, loc, false);
  if (current_rule->ruleprec)
    duplicate_rule_directive ("%prec",
                              current_rule->ruleprec->location, loc);
  else
    current_rule->ruleprec = precsym;
}

/* Set %empty for the current rule. */

void
grammar_current_rule_empty_set (location loc)
{
  /* If %empty is used and -Wno-empty-rule is not, then enable
     -Wempty-rule.  */
  if (warning_is_unset (Wempty_rule))
    warning_argmatch ("empty-rule", 0, 0);
  if (current_rule->percent_empty_loc.start.file)
    duplicate_rule_directive ("%empty",
                              current_rule->percent_empty_loc, loc);
  else
    current_rule->percent_empty_loc = loc;
}

/* Attach dynamic precedence DPREC to the current rule. */

void
grammar_current_rule_dprec_set (int dprec, location loc)
{
  if (! glr_parser)
    complain (&loc, Wother, _("%s affects only GLR parsers"),
              "%dprec");
  if (dprec <= 0)
    complain (&loc, complaint, _("%s must be followed by positive number"),
              "%dprec");
  else if (current_rule->dprec != 0)
    duplicate_rule_directive ("%dprec",
                              current_rule->dprec_loc, loc);
  else
    {
      current_rule->dprec = dprec;
      current_rule->dprec_loc = loc;
    }
}

/* Attach a merge function NAME with argument type TYPE to current
   rule. */

void
grammar_current_rule_merge_set (uniqstr name, location loc)
{
  if (! glr_parser)
    complain (&loc, Wother, _("%s affects only GLR parsers"),
              "%merge");
  if (current_rule->merger != 0)
    duplicate_rule_directive ("%merge",
                              current_rule->merger_declaration_loc, loc);
  else
    {
      current_rule->merger = get_merge_function (name);
      current_rule->merger_declaration_loc = loc;
    }
}

/* Attach SYM to the current rule.  If needed, move the previous
   action as a midrule action.  */

void
grammar_current_rule_symbol_append (symbol *sym, location loc,
                                    named_ref *name)
{
  if (current_rule->action_props.code)
    grammar_midrule_action ();
  symbol_list *p = grammar_symbol_append (sym, loc);
  if (name)
    assign_named_ref (p, name);
  if (sym->content->status == undeclared || sym->content->status == used)
    sym->content->status = needed;
}

void
grammar_current_rule_action_append (const char *action, location loc,
                                    named_ref *name, uniqstr type)
{
  if (current_rule->action_props.code)
    grammar_midrule_action ();
  if (type)
    complain (&loc, Wyacc,
              _("POSIX Yacc does not support typed midrule actions"));
  /* After all symbol declarations have been parsed, packgram invokes
     code_props_translate_code.  */
  code_props_rule_action_init (&current_rule->action_props, action, loc,
                               current_rule,
                               name, type,
                               /* is_predicate */ false);
}

void
grammar_current_rule_predicate_append (const char *pred, location loc)
{
  if (current_rule->action_props.code)
    grammar_midrule_action ();
  code_props_rule_action_init (&current_rule->action_props, pred, loc,
                               current_rule,
                               NULL, NULL,
                               /* is_predicate */ true);
}

/* Set the expected number of shift-reduce (reduce-reduce) conflicts for
 * the current rule.  If a midrule is encountered later, the count
 * is transferred to it and reset in the current rule to -1. */

void
grammar_current_rule_expect_sr (int count, location loc)
{
  (void) loc;
  current_rule->expected_sr_conflicts = count;
}

void
grammar_current_rule_expect_rr (int count, location loc)
{
  if (! glr_parser)
    complain (&loc, Wother, _("%s affects only GLR parsers"),
              "%expect-rr");
  else
    current_rule->expected_rr_conflicts = count;
}


/*---------------------------------------------.
| Build RULES and RITEM from what was parsed.  |
`---------------------------------------------*/

static void
packgram (void)
{
  int itemno = 0;
  ritem = xnmalloc (nritems + 1, sizeof *ritem);
  /* This sentinel is used by build_relations() in lalr.c.  */
  *ritem++ = 0;

  rule_number ruleno = 0;
  rules = xnmalloc (nrules, sizeof *rules);

  for (symbol_list *p = grammar; p; p = p->next)
    {
      symbol_list *lhs = p;
      record_merge_function_type (lhs->merger, lhs->content.sym->content->type_name,
                                  lhs->merger_declaration_loc);
      /* If the midrule's $$ is set or its $n is used, remove the '$' from the
         symbol name so that it's a user-defined symbol so that the default
         %destructor and %printer apply.  */
      if (lhs->midrule_parent_rule /* i.e., symbol_is_dummy (lhs->content.sym).  */
          && (lhs->action_props.is_value_used
              || (symbol_list_n_get (lhs->midrule_parent_rule,
                                     lhs->midrule_parent_rhs_index)
                  ->action_props.is_value_used)))
        lhs->content.sym->tag += 1;

      /* Don't check the generated rule 0.  It has no action, so some rhs
         symbols may appear unused, but the parsing algorithm ensures that
         %destructor's are invoked appropriately.  */
      if (lhs != grammar)
        grammar_rule_check_and_complete (lhs);

      rules[ruleno].user_number = ruleno;
      rules[ruleno].number = ruleno;
      rules[ruleno].lhs = lhs->content.sym->content;
      rules[ruleno].rhs = ritem + itemno;
      rules[ruleno].prec = NULL;
      rules[ruleno].dprec = lhs->dprec;
      rules[ruleno].merger = lhs->merger;
      rules[ruleno].precsym = NULL;
      rules[ruleno].location = lhs->rhs_loc;
      rules[ruleno].useful = true;
      rules[ruleno].action = lhs->action_props.code;
      rules[ruleno].action_loc = lhs->action_props.location;
      rules[ruleno].is_predicate = lhs->action_props.is_predicate;
      rules[ruleno].expected_sr_conflicts = lhs->expected_sr_conflicts;
      rules[ruleno].expected_rr_conflicts = lhs->expected_rr_conflicts;

      /* Traverse the rhs.  */
      {
        size_t rule_length = 0;
        for (p = lhs->next; p->content.sym; p = p->next)
          {
            ++rule_length;

            /* Don't allow rule_length == INT_MAX, since that might
               cause confusion with strtol if INT_MAX == LONG_MAX.  */
            if (rule_length == INT_MAX)
              complain (&rules[ruleno].location, fatal, _("rule is too long"));

            /* item_number = symbol_number.
               But the former needs to contain more: negative rule numbers. */
            ritem[itemno++] =
              symbol_number_as_item_number (p->content.sym->content->number);
            /* A rule gets by default the precedence and associativity
               of its last token.  */
            if (p->content.sym->content->class == token_sym && default_prec)
              rules[ruleno].prec = p->content.sym->content;
          }
      }

      /* If this rule has a %prec,
         the specified symbol's precedence replaces the default.  */
      if (lhs->ruleprec)
        {
          rules[ruleno].precsym = lhs->ruleprec->content;
          rules[ruleno].prec = lhs->ruleprec->content;
        }

      /* An item ends by the rule number (negated).  */
      ritem[itemno++] = rule_number_as_item_number (ruleno);
      aver (itemno < ITEM_NUMBER_MAX);
      ++ruleno;
      aver (ruleno < RULE_NUMBER_MAX);
    }

  aver (itemno == nritems);

  if (trace_flag & trace_sets)
    ritem_print (stderr);
}


/*------------------------------------------------------------------.
| Read in the grammar specification and record it in the format     |
| described in gram.h.  All actions are copied into ACTION_OBSTACK, |
| in each case forming the body of a C function (YYACTION) which    |
| contains a switch statement to decide which action to execute.    |
`------------------------------------------------------------------*/

void
reader (const char *gram)
{
  /* Set up symbol_table, semantic_type_table, and the built-in
     symbols.  */
  symbols_new ();

  gram_scanner_open (gram);
  gram_parse ();
  gram_scanner_close ();

  prepare_percent_define_front_end_variables ();

  if (complaint_status  < status_complaint)
    check_and_convert_grammar ();
}

static void
prepare_percent_define_front_end_variables (void)
{
  /* Set %define front-end variable defaults.  */
  muscle_percent_define_default ("lr.keep-unreachable-state", "false");
  {
    /* IELR would be a better default, but LALR is historically the
       default.  */
    muscle_percent_define_default ("lr.type", "lalr");
    char *lr_type = muscle_percent_define_get ("lr.type");
    if (STRNEQ (lr_type, "canonical-lr"))
      muscle_percent_define_default ("lr.default-reduction", "most");
    else
      muscle_percent_define_default ("lr.default-reduction", "accepting");
    free (lr_type);
  }

  /* Check %define front-end variables.  */
  {
    static char const * const values[] =
      {
       "lr.type", "lr""(0)", "lalr", "ielr", "canonical-lr", NULL,
       "lr.default-reduction", "most", "consistent", "accepting", NULL,
       NULL
      };
    muscle_percent_define_check_values (values);
  }
}

/* Find the first LHS which is not a dummy.  */

static symbol *
find_start_symbol (void)
{
  symbol_list *res = grammar;
  /* Skip all the possible dummy rules of the first rule.  */
  for (; symbol_is_dummy (res->content.sym); res = res->next)
    /* Skip the LHS, and then all the RHS of the dummy rule.  */
    for (res = res->next; res->content.sym; res = res->next)
      continue;
  return res->content.sym;
}


/*-------------------------------------------------------------.
| Check the grammar that has just been read, and convert it to |
| internal form.                                               |
`-------------------------------------------------------------*/

static void
check_and_convert_grammar (void)
{
  /* Grammar has been read.  Do some checking.  */
  if (nrules == 0)
    complain (NULL, fatal, _("no rules in the input grammar"));

  /* If the user did not define her ENDTOKEN, do it now. */
  if (!endtoken)
    {
      endtoken = symbol_get ("YYEOF", empty_loc);
      endtoken->content->class = token_sym;
      endtoken->content->number = 0;
      /* Value specified by POSIX.  */
      endtoken->content->user_token_number = 0;
      {
        symbol *alias = symbol_get ("$end", empty_loc);
        symbol_class_set (alias, token_sym, empty_loc, false);
        symbol_make_alias (endtoken, alias, empty_loc);
      }
    }

  /* Report any undefined symbols and consider them nonterminals.  */
  symbols_check_defined ();

  /* Find the start symbol if no %start.  */
  if (!start_flag)
    {
      symbol *start = find_start_symbol ();
      grammar_start_symbol_set (start, start->location);
    }

  /* Insert the initial rule, whose line is that of the first rule
     (not that of the start symbol):

     $accept: %start $end.  */
  {
    symbol_list *p = symbol_list_sym_new (accept, empty_loc);
    p->rhs_loc = grammar->rhs_loc;
    p->next = symbol_list_sym_new (startsymbol, empty_loc);
    p->next->next = symbol_list_sym_new (endtoken, empty_loc);
    p->next->next->next = symbol_list_sym_new (NULL, empty_loc);
    p->next->next->next->next = grammar;
    nrules += 1;
    nritems += 3;
    grammar = p;
  }

  aver (nsyms <= SYMBOL_NUMBER_MAXIMUM);
  aver (nsyms == ntokens + nvars);

  /* Assign the symbols their symbol numbers.  */
  symbols_pack ();

  /* Scan rule actions after invoking symbol_check_alias_consistency
     (in symbols_pack above) so that token types are set correctly
     before the rule action type checking.

     Before invoking grammar_rule_check_and_complete (in packgram
     below) on any rule, make sure all actions have already been
     scanned in order to set 'used' flags.  Otherwise, checking that a
     midrule's $$ should be set will not always work properly because
     the check must forward-reference the midrule's parent rule.  For
     the same reason, all the 'used' flags must be set before checking
     whether to remove '$' from any midrule symbol name (also in
     packgram).  */
  for (symbol_list *sym = grammar; sym; sym = sym->next)
    code_props_translate_code (&sym->action_props);

  /* Convert the grammar into the format described in gram.h.  */
  packgram ();

  /* The grammar as a symbol_list is no longer needed. */
  symbol_list_free (grammar);
}
