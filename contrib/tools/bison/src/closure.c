/* Closures for Bison

   Copyright (C) 1984, 1989, 2000-2002, 2004-2005, 2007, 2009-2015,
   2018-2019 Free Software Foundation, Inc.

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

#include <bitset.h>
#include <bitsetv.h>

#include "closure.h"
#include "derives.h"
#include "getargs.h"
#include "gram.h"
#include "reader.h"
#include "symtab.h"

/* NITEMSET is the size of the array ITEMSET.  */
item_number *itemset;
size_t nitemset;

static bitset ruleset;

/* internal data.  See comments before set_fderives and set_firsts.  */
static bitsetv fderives = NULL;
static bitsetv firsts = NULL;

/* Retrieve the FDERIVES/FIRSTS sets of the nonterminals numbered Var.  */
#define FDERIVES(Var)   fderives[(Var) - ntokens]
#define FIRSTS(Var)   firsts[(Var) - ntokens]


/*-----------------.
| Debugging code.  |
`-----------------*/

static void
print_closure (char const *title, item_number const *array, size_t size)
{
  fprintf (stderr, "Closure: %s\n", title);
  for (size_t i = 0; i < size; ++i)
    {
      fprintf (stderr, "  %2d: .", array[i]);
      item_number *rp;
      for (rp = &ritem[array[i]]; *rp >= 0; ++rp)
        fprintf (stderr, " %s", symbols[*rp]->tag);
      fprintf (stderr, "  (rule %d)\n", -*rp - 1);
    }
  fputs ("\n\n", stderr);
}


static void
print_firsts (void)
{
  fprintf (stderr, "FIRSTS\n");
  for (symbol_number i = ntokens; i < nsyms; ++i)
    {
      fprintf (stderr, "  %s firsts\n", symbols[i]->tag);
      bitset_iterator iter;
      symbol_number j;
      BITSET_FOR_EACH (iter, FIRSTS (i), j, 0)
        fprintf (stderr, "    %s\n", symbols[j + ntokens]->tag);
    }
  fprintf (stderr, "\n\n");
}


static void
print_fderives (void)
{
  fprintf (stderr, "FDERIVES\n");
  for (symbol_number i = ntokens; i < nsyms; ++i)
    {
      fprintf (stderr, "  %s derives\n", symbols[i]->tag);
      bitset_iterator iter;
      rule_number r;
      BITSET_FOR_EACH (iter, FDERIVES (i), r, 0)
        {
          fprintf (stderr, "    %3d ", r);
          rule_rhs_print (&rules[r], stderr);
          fprintf (stderr, "\n");
        }
    }
  fprintf (stderr, "\n\n");
}

/*------------------------------------------------------------------.
| Set FIRSTS to be an NVARS array of NVARS bitsets indicating which |
| items can represent the beginning of the input corresponding to   |
| which other items.                                                |
|                                                                   |
| For example, if some rule expands symbol 5 into the sequence of   |
| symbols 8 3 20, the symbol 8 can be the beginning of the data for |
| symbol 5, so the bit [8 - ntokens] in first[5 - ntokens] (= FIRST |
| (5)) is set.                                                      |
`------------------------------------------------------------------*/

static void
set_firsts (void)
{
  firsts = bitsetv_create (nvars, nvars, BITSET_FIXED);

  for (symbol_number i = ntokens; i < nsyms; ++i)
    for (symbol_number j = 0; derives[i - ntokens][j]; ++j)
      {
        item_number sym = derives[i - ntokens][j]->rhs[0];
        if (ISVAR (sym))
          bitset_set (FIRSTS (i), sym - ntokens);
      }

  if (trace_flag & trace_sets)
    bitsetv_matrix_dump (stderr, "RTC: Firsts Input", firsts);
  bitsetv_reflexive_transitive_closure (firsts);
  if (trace_flag & trace_sets)
    bitsetv_matrix_dump (stderr, "RTC: Firsts Output", firsts);

  if (trace_flag & trace_sets)
    print_firsts ();
}

/*-------------------------------------------------------------------.
| Set FDERIVES to an NVARS by NRULES matrix of bits indicating which |
| rules can help derive the beginning of the data for each           |
| nonterminal.                                                       |
|                                                                    |
| For example, if symbol 5 can be derived as the sequence of symbols |
| 8 3 20, and one of the rules for deriving symbol 8 is rule 4, then |
| the [5 - NTOKENS, 4] bit in FDERIVES is set.                       |
`-------------------------------------------------------------------*/

static void
set_fderives (void)
{
  fderives = bitsetv_create (nvars, nrules, BITSET_FIXED);

  set_firsts ();

  for (symbol_number i = ntokens; i < nsyms; ++i)
    for (symbol_number j = ntokens; j < nsyms; ++j)
      if (bitset_test (FIRSTS (i), j - ntokens))
        for (rule_number k = 0; derives[j - ntokens][k]; ++k)
          bitset_set (FDERIVES (i), derives[j - ntokens][k]->number);

  if (trace_flag & trace_sets)
    print_fderives ();

  bitsetv_free (firsts);
}



void
new_closure (unsigned n)
{
  itemset = xnmalloc (n, sizeof *itemset);

  ruleset = bitset_create (nrules, BITSET_FIXED);

  set_fderives ();
}



void
closure (item_number const *core, size_t n)
{
  if (trace_flag & trace_sets)
    print_closure ("input", core, n);

  bitset_zero (ruleset);

  for (size_t c = 0; c < n; ++c)
    if (ISVAR (ritem[core[c]]))
      bitset_or (ruleset, ruleset, FDERIVES (ritem[core[c]]));

  /* core is sorted on item index in ritem, which is sorted on rule number.
     Compute itemset with the same sort.  */
  /* A bit index over RULESET. */
  rule_number ruleno;
  bitset_iterator iter;

  nitemset = 0;
  size_t c = 0;
  BITSET_FOR_EACH (iter, ruleset, ruleno, 0)
    {
      item_number itemno = rules[ruleno].rhs - ritem;
      while (c < n && core[c] < itemno)
        {
          itemset[nitemset] = core[c];
          nitemset++;
          c++;
        }
      itemset[nitemset] = itemno;
      nitemset++;
    };

  while (c < n)
    {
      itemset[nitemset] = core[c];
      nitemset++;
      c++;
    }

  if (trace_flag & trace_sets)
    print_closure ("output", itemset, nitemset);
}


void
free_closure (void)
{
  free (itemset);
  bitset_free (ruleset);
  bitsetv_free (fderives);
}
