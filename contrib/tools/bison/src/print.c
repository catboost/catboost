/* Print information on generated parser, for bison,

   Copyright (C) 1984, 1986, 1989, 2000-2005, 2007, 2009-2015, 2018-2021
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

#include <config.h>

#include "print.h"

#include "system.h"

#include <bitset.h>
#include <mbswidth.h>

#include "closure.h"
#include "complain.h"
#include "conflicts.h"
#include "counterexample.h"
#include "files.h"
#include "getargs.h"
#include "gram.h"
#include "lalr.h"
#include "lr0.h"
#include "muscle-tab.h"
#include "reader.h"
#include "reduce.h"
#include "state.h"
#include "symtab.h"
#include "tables.h"

static bitset no_reduce_set;



/*---------------------------------------.
| *WIDTH := max (*WIDTH, strlen (STR)).  |
`---------------------------------------*/

static void
max_length (size_t *width, const char *str)
{
  size_t len = mbswidth (str, 0);
  if (len > *width)
    *width = len;
}

/*--------------------------------.
| Report information on a state.  |
`--------------------------------*/

static void
print_core (FILE *out, const state *s)
{
  const item_index *sitems = s->items;
  size_t snritems = s->nitems;
  /* Output all the items of a state, not only its kernel.  */
  if (report_flag & report_itemsets)
    {
      closure (sitems, snritems);
      sitems = itemset;
      snritems = nitemset;
    }

  if (!snritems)
    return;

  fputc ('\n', out);

  rule const *previous_rule = NULL;
  for (size_t i = 0; i < snritems; i++)
    {
      item_number *sp1 = ritem + sitems[i];
      rule const *r = item_rule (sp1);
      item_print (sp1, previous_rule, out);
      previous_rule = r;

      /* Display the lookahead tokens?  */
      if (report_flag & report_lookaheads
          && item_number_is_rule_number (*sp1))
        state_rule_lookaheads_print (s, r, out);
      fputc ('\n', out);
    }
}


/*------------------------------------------------------------.
| Report the shifts iff DISPLAY_SHIFTS_P or the gotos of S on |
| OUT.                                                        |
`------------------------------------------------------------*/

static void
print_transitions (const state *s, FILE *out, bool display_transitions_p)
{
  transitions *trans = s->transitions;
  size_t width = 0;

  /* Compute the width of the lookahead token column.  */
  for (int i = 0; i < trans->num; i++)
    if (!TRANSITION_IS_DISABLED (trans, i)
        && TRANSITION_IS_SHIFT (trans, i) == display_transitions_p)
      {
        symbol *sym = symbols[TRANSITION_SYMBOL (trans, i)];
        max_length (&width, sym->tag);
      }

  /* Nothing to report. */
  if (!width)
    return;

  fputc ('\n', out);
  width += 2;

  /* Report lookahead tokens and shifts.  */
  for (int i = 0; i < trans->num; i++)
    if (!TRANSITION_IS_DISABLED (trans, i)
        && TRANSITION_IS_SHIFT (trans, i) == display_transitions_p)
      {
        symbol *sym = symbols[TRANSITION_SYMBOL (trans, i)];
        const char *tag = sym->tag;
        const state *s1 = trans->states[i];

        fprintf (out, "    %s", tag);
        for (int j = width - mbswidth (tag, 0); j > 0; --j)
          fputc (' ', out);
        if (display_transitions_p)
          fprintf (out, _("shift, and go to state %d\n"), s1->number);
        else
          fprintf (out, _("go to state %d\n"), s1->number);
      }
}


/*--------------------------------------------------------.
| Report the explicit errors of S raised from %nonassoc.  |
`--------------------------------------------------------*/

static void
print_errs (FILE *out, const state *s)
{
  errs *errp = s->errs;
  size_t width = 0;

  /* Compute the width of the lookahead token column.  */
  for (int i = 0; i < errp->num; ++i)
    if (errp->symbols[i])
      max_length (&width, errp->symbols[i]->tag);

  /* Nothing to report. */
  if (!width)
    return;

  fputc ('\n', out);
  width += 2;

  /* Report lookahead tokens and errors.  */
  for (int i = 0; i < errp->num; ++i)
    if (errp->symbols[i])
      {
        const char *tag = errp->symbols[i]->tag;
        fprintf (out, "    %s", tag);
        for (int j = width - mbswidth (tag, 0); j > 0; --j)
          fputc (' ', out);
        fputs (_("error (nonassociative)\n"), out);
      }
}


/*-------------------------------------------------------------------.
| Report a reduction of RULE on LOOKAHEAD (which can be 'default').  |
| If not ENABLED, the rule is masked by a shift or a reduce (S/R and |
| R/R conflicts).                                                    |
`-------------------------------------------------------------------*/

static void
print_reduction (FILE *out, size_t width,
                 const char *lookahead,
                 rule *r, bool enabled)
{
  fprintf (out, "    %s", lookahead);
  for (int j = width - mbswidth (lookahead, 0); j > 0; --j)
    fputc (' ', out);
  if (!enabled)
    fputc ('[', out);
  if (r->number)
    fprintf (out, _("reduce using rule %d (%s)"), r->number,
             r->lhs->symbol->tag);
  else
    fprintf (out, _("accept"));
  if (!enabled)
    fputc (']', out);
  fputc ('\n', out);
}


/*-------------------------------------------.
| Report on OUT the reduction actions of S.  |
`-------------------------------------------*/

static void
print_reductions (FILE *out, const state *s)
{
  reductions *reds = s->reductions;
  if (reds->num == 0)
    return;

  rule *default_reduction = NULL;
  if (yydefact[s->number] != 0)
    default_reduction = &rules[yydefact[s->number] - 1];

  transitions *trans = s->transitions;

  bitset_zero (no_reduce_set);
  {
    int i;
    FOR_EACH_SHIFT (trans, i)
      bitset_set (no_reduce_set, TRANSITION_SYMBOL (trans, i));
  }
  for (int i = 0; i < s->errs->num; ++i)
    if (s->errs->symbols[i])
      bitset_set (no_reduce_set, s->errs->symbols[i]->content->number);

  /* Compute the width of the lookahead token column.  */
  size_t width = 0;
  if (default_reduction)
    width = mbswidth (_("$default"), 0);

  if (reds->lookaheads)
    for (int i = 0; i < ntokens; i++)
      {
        bool count = bitset_test (no_reduce_set, i);

        for (int j = 0; j < reds->num; ++j)
          if (bitset_test (reds->lookaheads[j], i))
            {
              if (! count)
                {
                  if (reds->rules[j] != default_reduction)
                    max_length (&width, symbols[i]->tag);
                  count = true;
                }
              else
                max_length (&width, symbols[i]->tag);
            }
      }

  /* Nothing to report. */
  if (!width)
    return;

  fputc ('\n', out);
  width += 2;

  bool default_reduction_only = true;

  /* Report lookahead tokens (or $default) and reductions.  */
  if (reds->lookaheads)
    for (int i = 0; i < ntokens; i++)
      {
        bool defaulted = false;
        bool count = bitset_test (no_reduce_set, i);
        if (count)
          default_reduction_only = false;

        for (int j = 0; j < reds->num; ++j)
          if (bitset_test (reds->lookaheads[j], i))
            {
              if (! count)
                {
                  if (reds->rules[j] != default_reduction)
                    {
                      default_reduction_only = false;
                      print_reduction (out, width,
                                       symbols[i]->tag,
                                       reds->rules[j], true);
                    }
                  else
                    defaulted = true;
                  count = true;
                }
              else
                {
                  default_reduction_only = false;
                  if (defaulted)
                    print_reduction (out, width,
                                     symbols[i]->tag,
                                     default_reduction, true);
                  defaulted = false;
                  print_reduction (out, width,
                                   symbols[i]->tag,
                                   reds->rules[j], false);
                }
            }
      }

  if (default_reduction)
    {
      char *default_reductions =
        muscle_percent_define_get ("lr.default-reduction");
      print_reduction (out, width, _("$default"), default_reduction, true);
      aver (STREQ (default_reductions, "most")
            || (STREQ (default_reductions, "consistent")
                && default_reduction_only)
            || (reds->num == 1 && reds->rules[0]->number == 0));
      (void) default_reduction_only;
      free (default_reductions);
    }
}


/*--------------------------------------------------------------.
| Report on OUT all the actions (shifts, gotos, reductions, and |
| explicit errors from %nonassoc) of S.                         |
`--------------------------------------------------------------*/

static void
print_actions (FILE *out, const state *s)
{
  /* Print shifts.  */
  print_transitions (s, out, true);
  print_errs (out, s);
  print_reductions (out, s);
  /* Print gotos.  */
  print_transitions (s, out, false);
}


/*----------------------------------.
| Report all the data on S on OUT.  |
`----------------------------------*/

static void
print_state (FILE *out, const state *s)
{
  fputs ("\n\n", out);
  fprintf (out, _("State %d"), s->number);
  fputc ('\n', out);
  print_core (out, s);
  print_actions (out, s);
  if ((report_flag & report_solved_conflicts) && s->solved_conflicts)
    {
      fputc ('\n', out);
      fputs (s->solved_conflicts, out);
    }
  if (has_conflicts (s)
      && (report_flag & report_cex
          || warning_is_enabled (Wcounterexamples)))
    {
      fputc ('\n', out);
      counterexample_report_state (s, out, "    ");
    }
}

/*-----------------------------------------.
| Print information on the whole grammar.  |
`-----------------------------------------*/

static void
print_terminal_symbols (FILE *out)
{
  /* TERMINAL (type #) : rule #s terminal is on RHS */
  fprintf (out, "%s\n\n", _("Terminals, with rules where they appear"));
  for (int i = 0; i < max_code + 1; ++i)
    if (token_translations[i] != undeftoken->content->number)
      {
        const symbol *sym = symbols[token_translations[i]];
        const char *tag = sym->tag;
        fprintf (out, "%4s%s", "", tag);
        if (sym->content->type_name)
          fprintf (out, " <%s>", sym->content->type_name);
        fprintf (out, " (%d)", i);

        for (rule_number r = 0; r < nrules; r++)
          for (item_number *rhsp = rules[r].rhs; *rhsp >= 0; rhsp++)
            if (item_number_as_symbol_number (*rhsp) == token_translations[i])
              {
                fprintf (out, " %d", r);
                break;
              }
        fputc ('\n', out);
      }
  fputs ("\n\n", out);
}


static void
print_nonterminal_symbols (FILE *out)
{
  fprintf (out, "%s\n\n", _("Nonterminals, with rules where they appear"));
  for (symbol_number i = ntokens; i < nsyms; i++)
    {
      const symbol *sym = symbols[i];
      const char *tag = sym->tag;
      bool on_left = false;
      bool on_right = false;

      for (rule_number r = 0; r < nrules; r++)
        {
          on_left |= rules[r].lhs->number == i;
          for (item_number *rhsp = rules[r].rhs; !on_right && 0 <= *rhsp; ++rhsp)
            on_right |= item_number_as_symbol_number (*rhsp) == i;
          if (on_left && on_right)
            break;
        }

      int column = 4 + mbswidth (tag, 0);
      fprintf (out, "%4s%s", "", tag);
      if (sym->content->type_name)
        column += fprintf (out, " <%s>",
                           sym->content->type_name);
      fprintf (out, " (%d)\n", i);

      if (on_left)
        {
          fprintf (out, "%8s%s", "", _("on left:"));
          for (rule_number r = 0; r < nrules; r++)
            if (rules[r].lhs->number == i)
              fprintf (out, " %d", r);
          fputc ('\n', out);
        }

      if (on_right)
        {
          fprintf (out, "%8s%s", "", _("on right:"));
          for (rule_number r = 0; r < nrules; r++)
            for (item_number *rhsp = rules[r].rhs; 0 <= *rhsp; ++rhsp)
              if (item_number_as_symbol_number (*rhsp) == i)
                {
                  fprintf (out, " %d", r);
                  break;
                }
          fputc ('\n', out);
        }
    }
}

void
print_results (void)
{
  /* We used to use just .out if SPEC_NAME_PREFIX (-p) was used, but
     that conflicts with Posix.  */
  FILE *out = xfopen (spec_verbose_file, "w");

  reduce_output (out);
  grammar_rules_partial_print (out,
                               _("Rules useless in parser due to conflicts"),
                               rule_useless_in_parser_p);
  conflicts_output (out);

  grammar_rules_print (out);
  print_terminal_symbols (out);
  print_nonterminal_symbols (out);

  /* Storage for print_reductions.  */
  no_reduce_set = bitset_create (ntokens, BITSET_FIXED);
  for (state_number i = 0; i < nstates; i++)
    print_state (out, states[i]);
  bitset_free (no_reduce_set);

  xfclose (out);
}
